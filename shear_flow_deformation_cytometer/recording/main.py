from PyQt5 import uic
from PyQt5 import QtWidgets, QtCore, QtGui
from SwitchButton import SwitchButton

from pypylon import pylon

import numpy as np
from pathlib import Path
import tifffile
import sys
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import config
import filename
from datetime import datetime

from shear_flow_deformation_cytometer.gui.QExtendedGraphicsView import QExtendedGraphicsView
from qimage2ndarray import array2qimage


# this is the class which produces the graphs needed for histogram
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.ioff()
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.yaxis.set_visible(False)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
        self.axes.set_facecolor('snow')  # color of the background inside the graph
        self.fig.set_facecolor('whitesmoke')  # color of the padding around graph


class InputSlideSpinSwitch(QtWidgets.QWidget):
    def __init__(self, parent, brexpslide, brexpspin, brexplay):
        super().__init__(parent)
        brexpslide.valueChanged.connect(brexpspin.setValue)
        brexpspin.valueChanged.connect(brexpslide.setValue)

        brexpswitch = SwitchButton(self, '', 15, '', 31, 50)
        brexplay.insertWidget(0, brexpswitch)

        self.toggled = brexpswitch.toggled
        self.slider = brexpslide
        self.spin_box = brexpspin
        self.switch = brexpswitch

    def value(self):
        return self.spin_box.value()

    def show(self):
        for widget in [self.slider, self.spin_box, self.switch]:
            widget.show()

    def hide(self):
        for widget in [self.slider, self.spin_box, self.switch]:
            widget.hide()


class Camera(QtCore.QObject):
    camera = None
    img = None
    mounted = False
    active = True

    is_recording = False
    record_do_stop = False

    signal_display = QtCore.Signal(np.ndarray)
    signal_finished_recording = QtCore.Signal()

    def __init__(self, parent, exp, gain, hist, view, sn, flip, master):
        super().__init__()
        self.record_thread = None
        self.parent = parent
        self.exp = exp
        self.gain = gain
        self.hist = hist
        self.view = view
        self.sn = sn
        self.flip = flip
        self.master = master

        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.view.origin)

        self.parent.timer.timeout.connect(self.update_view)
        self.parent.htimer.timeout.connect(self.update_hist)

        self.signal_display.connect(self.display_image)

    def set_active(self, active):
        self.active = active
        widgets = [self.view, self.hist, self.exp, self.gain, self.sn, self.parent.fl_props]
        for widget in widgets:
            if active:
                widget.show()
            else:
                widget.hide()

    def unmount(self):
        try:  # try to close if it is already open
            self.camera.Close()
        except:
            pass

        self.mounted = False

    def mount(self):
        if self.camera is not None:
            try: #try to close if it is already open
                self.camera.Close()
            except:
                pass

        try:
            binfo = pylon.DeviceInfo()
            binfo.SetSerialNumber(self.sn.currentText())
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(binfo))
            self.camera.Open()
            self.set_camera_static_parameters()
            self.sn.setStyleSheet('Background: rgb(170, 255, 127);')
            self.exp.switch.toggled.connect(lambda c: self.auto_exposure(c))
            self.gain.switch.toggled.connect(lambda c: self.auto_gain(c))
            self.auto_exposure(self.exp.switch.isChecked())
            self.auto_gain(self.gain.switch.isChecked())

            self.camera.ReverseX.SetValue(self.flip[0])
            self.camera.ReverseY.SetValue(self.flip[1])
            if self.master is True:
                self.camera.LineSelector.SetValue("Line3")
                if self.parent.exTswitch.isChecked():
                    self.set_slave()
            else:
                self.set_slave()
            self.mounted = True
        except Exception as err:
            print(err, file=sys.stderr)
            self.sn.setStyleSheet('Background: rgb(255, 170, 127);')
            self.mounted = False

    def live(self):
        if not self.active:
            return
        if self.mounted is False:
            self.mount()
        try:
            self.camera.StopGrabbing()
        except:
            pass

        self.set_line_mode()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def prepare_rec(self):
        if not self.active:
            return
        if self.mounted is False:
            self.mount()
        try:
            self.camera.StopGrabbing()
        except:
            pass

        if self.exp.spin_box.isEnabled():
            self.camera.ExposureTime.SetValue(self.exp.spin_box.value())
        if self.gain.spin_box.isEnabled():
            self.camera.Gain.SetValue(self.gain.spin_box.value())

        self.set_line_mode()
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    def set_line_mode(self):
        self.camera.AcquisitionFrameRate = self.parent.frate.value()
        if self.parent.exTswitch.isChecked() or self.master is False:
            self.camera.LineMode.SetValue("Input")
        else:
            self.camera.LineMode.SetValue("Output")
        if self.master is True:
            self.camera.LineSource.SetValue("ExposureActive")

    def record_stop(self):
        self.record_do_stop = True

    def record_sequence(self, output_path, frame_rate, total_frame_number, sk, callback_end=None, callback_frame=None):
        self.record_do_stop = False
        self.record_thread = threading.Thread(target=self.record_sequence_thread, args=(output_path, frame_rate, total_frame_number, sk, callback_end, callback_frame))

    def record_sequence_thread(self, output_path, frame_rate, total_frame_number, sk, callback_end, callback_frame=None):
        self.is_recording = True
        tif_writer = tifffile.TiffWriter(output_path, bigtiff=True)
        try:
            timestamp_start = None
            current_frame_number = 0
            dt = 1 / frame_rate * 1e3
            while current_frame_number < total_frame_number and not self.record_do_stop:
                # grab the next frame
                grab = self.camera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if grab.GrabSucceeded():
                    # get the image and timestamp
                    img = grab.GetArray()
                    timestamp = grab.GetTimeStamp() / 1000000
                    # store the first timestamp as start time
                    if timestamp_start is None:
                        timestamp_start = timestamp
                    # round the timestamp to the nearest frame number
                    frame_number = np.round((timestamp - timestamp_start)/dt + 0.5)

                    # if we missed some frames, fill them up with black frames
                    while current_frame_number < frame_number and current_frame_number < total_frame_number:
                        meta_data = {'timestamp': "nan"}
                        tif_writer.save(np.zeros_like(img), compression=0, metadata=meta_data, contiguous=False)
                        current_frame_number += 1

                    # if we don't have reached the end yet, store the current frame
                    if current_frame_number < total_frame_number:
                        meta_data = {'timestamp': str(timestamp)}
                        tif_writer.save(img, compression=0, metadata=meta_data, contiguous=False)
                        current_frame_number += 1

                    if current_frame_number % sk == 0 and callback_frame:
                        callback_frame(img)
                        self.display_image(img)
        finally:
            tif_writer.close()
            self.is_recording = False
            if callback_end is not None:
                callback_end()

    def display_image(self, img):
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(img)))

    def update_hist(self):
        if self.mounted is True and self.img is not None:
            self.hist.axes.clear()
            self.hist.axes.set_xlim([0 , 255])
            self.hist.axes.set_ylim([0 , 1.1])
            # self.img = plt.imread('1.jpg')

            y, x = np.histogram(self.img, bins=np.arange(0, 255, 10), density=True)
            x = x[:-1] + np.diff(x)
            y = y / y.max()
            self.hist.axes.plot(x, y)
            self.hist.axes.fill_between(x, 0, y, alpha=0.5)
            self.hist.draw()

        if self.mounted is True:
            if not self.exp.spin_box.isEnabled():
                self.exp.spin_box.setValue(int(self.camera.ExposureTime.GetValue()))
            if not self.gain.spin_box.isEnabled():
                self.gain.spin_box.setValue(int(self.camera.Gain.GetValue()))

    def update_view(self):
        if self.mounted:
            if not self.exp.switch.isChecked():
                self.camera.AcquisitionFrameRate = self.parent.frate.value()
            if self.exp.spin_box.isEnabled():
                self.camera.ExposureTime.SetValue(self.exp.spin_box.value())
            if self.gain.spin_box.isEnabled():
                self.camera.Gain.SetValue(self.gain.spin_box.value())

            if self.camera.IsGrabbing():
                grab = self.camera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if grab.GrabSucceeded():
                    self.img = grab.GetArray()
                    self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(self.img)))
                    self.view.setExtend(self.img.shape[1], self.img.shape[0])
                grab.Release()

    # function where repetetive camera settings is being applied. to avoid redundancy in the code
    def set_camera_static_parameters(self):
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.ExposureAuto = "Off"
        self.camera.GainAuto = "Off"
        self.camera.AutoTargetBrightness = 0.5
        self.camera.MaxNumBuffer = 21
        # cam.GetStreamGrabberParams().MaxTransferSize.SetValue(1048576)
        self.camera.ChunkModeActive = True
        self.camera.ChunkSelector = "Timestamp"
        self.camera.ChunkEnable = True
        self.camera.ChunkSelector = "Gain"
        self.camera.ChunkEnable = True
        self.camera.ChunkSelector = "ExposureTime"
        self.camera.ChunkEnable = True

    # set the settings specific to the slave camera
    def set_slave(self):
        self.camera.LineSelector.SetValue("Line3")
        self.camera.TriggerSelector.SetValue("FrameStart")
        self.camera.TriggerMode.SetValue("On")
        #self.camera.TriggerDelay.SetValue(30)
        self.camera.ExposureMode.SetValue("Timed")
        self.camera.TriggerActivation.SetValue("RisingEdge")
        self.camera.TriggerSource.SetValue("Line3")

    # turn the camera auto-exposure on or off
    def auto_exposure(self, active):
        if active is True:
            self.camera.ExposureAuto = "Continuous"
            self.exp.slider.setValue(int(self.camera.ExposureTime.GetValue()))
            self.exp.slider.setEnabled(False)
            self.exp.spin_box.setEnabled(False)
        else:
            self.camera.ExposureAuto = "Off"
            self.exp.slider.setEnabled(True)
            self.exp.spin_box.setEnabled(True)

    # turn the camera auto-gain on or off
    def auto_gain(self, active):
        if active is True:
            self.camera.GainAuto = "Continuous"
            self.gain.slider.setValue(int(self.camera.Gain.GetValue()))
            self.gain.slider.setEnabled(False)
            self.gain.spin_box.setEnabled(False)
        else:
            self.camera.GainAuto = "Off"
            self.gain.slider.setEnabled(True)
            self.gain.spin_box.setEnabled(True)



class MainWindow(QtWidgets.QMainWindow):
    isStopped = True
    recording_thread = None
    recording_finished = QtCore.Signal()
    recording_counter = QtCore.Signal(int)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowIcon(QtGui.QIcon('images/icon.png'))
        # loads the ui from .ui file
        self.ui = uic.loadUi('twoCamera.ui', self)
        self.setWindowTitle('Shear Flow Deformation Cytometer Recording')

        # button icons and styling
        self.counter.setAlignment(QtCore.Qt.AlignCenter)
        self.rec.setIcon(QtGui.QIcon('images/rec.png'))
        self.stop.setIcon(QtGui.QIcon('images/stop.png'))
        self.live.setIcon(QtGui.QIcon('images/play.png'))
        self.mount.setIcon(QtGui.QIcon('images/mount.png'))

        self.switchpage.clicked.connect(lambda: self.Swidget.setCurrentIndex((self.Swidget.currentIndex() + 1) %2) )

        self.NoteEdit.returnPressed.connect(self.EnterNote)

        # making the graphs for histograms and placing them inside window
        self.brhist = MplCanvas(self, width=4, height=2.5, dpi=70)
        self.brlay.insertWidget(3, self.brhist)

        self.flhist = MplCanvas(self, width=4, height=2.5, dpi=70)
        self.fllay.insertWidget(3, self.flhist)

        self.timer = QtCore.QTimer(self)
        self.htimer = QtCore.QTimer(self)

        self.brview = QExtendedGraphicsView()
        self.brview.setMinimumHeight(300)
        self.layout_views.addWidget(self.brview)

        self.flview = QExtendedGraphicsView()
        self.layout_views.addWidget(self.flview)

        self.cameras = [
            Camera(self,
                InputSlideSpinSwitch(self, self.brexpslide, self.brexpspin, self.brexplay),
                InputSlideSpinSwitch(self, self.brgainslide, self.brgainspin, self.brgainlay),
                self.brhist,
                self.brview,
                self.brsn,
                flip=[False, True],
                master=True,
            ),
            Camera(self,
                InputSlideSpinSwitch(self, self.flexpslide, self.flexpspin, self.flexplay),
                InputSlideSpinSwitch(self, self.flgainslide, self.flgainspin, self.flgainlay),
                self.flhist,
                self.flview,
                self.flsn,
                flip=[True, True],
                master=False,
            )
        ]

        self.twoCamSwitch = SwitchButton(self, 'On', 10, 'Off', 31, 60)
        self.cameraPar.addWidget(self.twoCamSwitch, 1, 1)
        self.twoCamSwitch.toggled.connect(self.TwoCamMode)
        self.TwoCamMode(False)

        # making external switch botton
        self.exTswitch = SwitchButton(self, 'On', 10, 'Off', 31, 60)
        self.cameraPar.addWidget(self.exTswitch, 2, 1)
        
        self.exTswitch.toggled.connect(self.ETrig)

        self.button_find_cams.clicked.connect(self.findCameras)
        self.findCameras()

        # reading default config file
        self.conf = config.SetupConfig('config.txt')  # config has its own class
        self.exTswitch.setChecked(self.conf.exTrig)
        self.conf.update(self) #
        self.saveCon.clicked.connect(self.SaveCon)


        # connects values of frame rate, duration is s and frames so that each change when the other changes
        self.fnum.setValue(int(self.duration.value() * self.frate.value()))
        self.duration.valueChanged.connect(lambda c: self.fnum.setValue(int(c * self.frate.value())))
        self.fnum.valueChanged.connect(lambda c: self.duration.setValue(float(c / self.frate.value())))
        self.fnum.valueChanged.connect(lambda c: self.duration.setValue(float(c / self.frate.value())))
        self.frate.valueChanged.connect(lambda c: self.fnum.setValue(int(c * self.duration.value())))
        # self.bar.setMaximum(self.fnum.value())
        # self.fnum.valueChanged.connect(lambda c: self.bar.setMaximum(c))

        ## conecting the push bottons to their functions
        self.mount.clicked.connect(self.Mount)
        self.live.clicked.connect(self.Live)
        self.rec.clicked.connect(self.Rec)
        self.stop.clicked.connect(self.Stop)

        ## connecting the browse botton to its function
        self.browse.clicked.connect(self.Browse)
        self.dpath = filename.Dpath() #default path function
        #creating the default path if it does not exist
        if os.path.exists(self.dpath)==False:
            os.mkdir(self.dpath)
        self.spath.setText(self.dpath)
        #reading the note file from default path
        notep = self.dpath + '\\' + 'Notes.txt'
        if os.path.isfile(notep):
            with open(notep) as Notes:
                lines = Notes.read()
                self.Notes.append(lines)
        # self.ctext.appendPlainText(self.Dpath)

        self.recording_finished.connect(self.on_recording_finished)
        self.recording_counter.connect(self.set_counter)

    def findCameras(self):
        for cam in self.cameras:
            cam.unmount()
        ## getting the list of connected camera's SN
        SN = []
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        for device in devices:
            SN.append( device.GetSerialNumber() )
        self.label_cams_found.setText(f"Cameras Found: {len(SN)}")

        #adding the serial numbers to list menus in ui
        for i in range(self.brsn.count()):
            self.brsn.removeItem(0)
        for i in range(self.flsn.count()):
            self.flsn.removeItem(0)
        self.brsn.addItems(SN)
        self.flsn.addItems(SN)
        if len(SN) > 1:
            self.flsn.setCurrentIndex(1) #poiting the fl to the second camera SN to avoid both pointing to the same camera
            self.twoCamSwitch.setEnabled(True)
        else:
            self.TwoCamMode(False)
            self.twoCamSwitch.setEnabled(False)

    ## stop either the live view or the recording   
    def Stop(self):
        try:
            self.htimer.stop()
            self.timer.stop()
        except:
            pass
        self.live.setStyleSheet('')
        self.isStopped = True
        
    ## function for mounting the cameras
    def Mount(self):
        for cam in self.cameras:
            cam.mount()

    ## function which starts the liveview
    def Live(self):
        self.isStopped = False
        for cam in self.cameras:
            cam.live()

        self.exTswitch.setEnabled(False)

        self.SaveCon()
        self.timer.start(int(1000/25))
        self.htimer.start(int(1000/5))
        self.live.setStyleSheet('Background: rgb(170, 255, 127);')

    ## function which starts the recording
    def Rec(self):
        try:
            self.htimer.stop()
            self.timer.stop()
            self.live.setStyleSheet('')
        except:
            pass
        path = self.spath.text()

        for cam in self.cameras:
            cam.prepare_rec()

        self.exTswitch.setEnabled(False)
        self.rec.setEnabled(False)

        self.rec.setStyleSheet('Background: rgb(255, 170, 127);')
        self.counter.setStyleSheet('color: rgb(255, 255, 255); background-color: rgb(0, 0, 0);')

        self.recording_thread = threading.Thread(target=self.Save)
        self.recording_thread.start()

    def on_recording_finished(self):
        # only when all cameras have finished
        for camera in self.cameras:
            if camera.active and camera.is_recording:
                return
        # self.bar.setValue(i)
        self.SaveNote()
        self.counter.setStyleSheet('color: rgb(0, 255, 0); background-color: rgb(0, 0, 0);')

        conp = filename.Conpath(self.brpath)
        self.conf.save(self)
        self.conf.savein(conp)
        self.exTswitch.setEnabled(True)
        self.rec.setEnabled(True)
        self.rec.setStyleSheet('')
        self.Live()

    ## funcation which runs the loop which saves the recording to the storage
    def Save(self):
        path = self.spath.text()
        Path(path).mkdir(parents=True, exist_ok=True)

        frate = self.frate.value()
        fnum = self.fnum.value()

        sk = np.clip(frate // 20, 1, np.Inf)

        paths = filename.path(path, ["", "_Fl"])
        self.cameras[0].record_sequence(paths[0], frate, fnum, sk, self.recording_finished, self.recording_counter.emit)
        if self.cameras[1].active:
            self.cameras[1].record_sequence(paths[1], frate, fnum, sk, self.recording_finished)

    def set_counter(self, value):
        self.counter.setText(str(value))

    ## saves the config file
    def SaveCon(self):
        self.conf.save(self)
        self.SaveNote()
        # self.pbar.setValue(200)

    def TwoCamMode(self, t):
        self.cameras[1].set_active(t)
        if t is False:
            self.cameras[1].unmount()
        else:
            if self.isStopped is False:
                self.cameras[1].live()

    ## function to enable and diable the external trigger. takes bolean as argument
    def ETrig(self, t):
        self.conf.exTrig = t
        if t:
            self.frate.setValue(500)
            self.frate.setEnabled(False)
            self.ctext.appendPlainText(str(self.frate.value()))
        else:
            self.frate.setEnabled(True)
            self.frate.setValue(self.conf.frate)

    ## function which opens the file dialog to ask for the saving directory
    def Browse(self):
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.ShowDirsOnly
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open a folder','', options=options)
        self.spath.setText(path)
        notep = path + '\\' + 'Notes.txt'
        with open(notep) as Notes:
            lines = Notes.read()
            self.Notes.append(lines)
        
    
    ## enters notes from line edit to the text box with the time of the note
    def EnterNote(self):
        now = datetime.now()
        time = now.strftime("[%H:%M] ")
        self.Notes.append(time + self.NoteEdit.text())
        self.NoteEdit.clear()

    ## save notes to selected path
    def SaveNote(self):
        path = self.spath.text() + '\\' + 'Notes.txt'
        with open(path, "w") as Notes:
            Notes.write(self.Notes.toPlainText())


def main():
    # These three lines tells windows to show the program as a singular item on taskbar
    import ctypes
    # set the app id to an arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('FAU.BaslerTwoCamera')

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    app.setStyle('Fusion')
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
