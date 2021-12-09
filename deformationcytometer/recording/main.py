from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication

from PyQt5 import uic
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from SwitchButton import *

from pypylon import pylon
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from datetime import datetime
import numpy as np
import math as m
import tifffile
import cv2 
import sys
import os
from pathlib import Path

#import pyqtgraph as pg

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

import config
import filename
from datetime import datetime
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
plt.rc('legend', fontsize=7)
plt.rc('axes', titlesize=10)
# These three lines tells windows to show the program as a singular item on taskbar
import ctypes
myappid = 'FAU.BaslerTwoCamera' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


#this is the class which produces the graphs needed for histogram
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.yaxis.set_visible(False)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
        self.axes.set_facecolor('snow') #color of the background inside the graph
        self.fig.set_facecolor('whitesmoke') #color of the pading aroung graph
        # plt.title('Histogram')

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowIcon(QtGui.QIcon('images/icon.png')) #window icon
        self.ui = uic.loadUi('twoCamera.ui', self) #loads the ui from .ui file
        self.setWindowTitle('Shear Flow Cytometer Recording') #window name

        ## botton icons and styling
        self.counter.setAlignment(QtCore.Qt.AlignCenter)
        self.rec.setIcon(QtGui.QIcon('images/rec.png'))
        self.stop.setIcon(QtGui.QIcon('images/stop.png'))
        self.live.setIcon(QtGui.QIcon('images/play.png'))
        self.mount.setIcon(QtGui.QIcon('images/mount.png'))
        # self.switchpage.setIcon(QtGui.QIcon('switch.png'))
        # self.switchpage.setStyleSheet('border : 0; background: transparent;')
        self.switchpage.clicked.connect(lambda: self.Swidget.setCurrentIndex((self.Swidget.currentIndex() + 1) %2) )

        self.NoteEdit.returnPressed.connect(self.EnterNote)

        ## connection brightfield gain and exp sliders and spin boxes
        self.brexpslide.valueChanged.connect(self.brexpspin.setValue)
        self.brgainslide.valueChanged.connect(self.brgainspin.setValue)
        self.brexpspin.valueChanged.connect(self.brexpslide.setValue)
        self.brgainspin.valueChanged.connect(self.brgainslide.setValue)
        ## connection fluorescent gain and exp sliders and spin boxes
        self.flexpslide.valueChanged.connect(self.flexpspin.setValue)
        self.flgainslide.valueChanged.connect(self.flgainspin.setValue)
        self.flexpspin.valueChanged.connect(self.flexpslide.setValue)
        self.flgainspin.valueChanged.connect(self.flgainslide.setValue)
        ## making auto gain and auto exp switches and placing them inside their layouts
        self.brexpswitch = SwitchButton(self, '', 15, '', 31, 50)
        self.brgainswitch = SwitchButton(self, '', 15, '', 31, 50)
        self.brexplay.insertWidget(0, self.brexpswitch)
        self.brgainlay.insertWidget(0 , self.brgainswitch)

        self.flexpswitch = SwitchButton(self, '', 15, '', 31, 50)
        self.flgainswitch = SwitchButton(self, '', 15, '', 31, 50)
        self.flexplay.insertWidget(0, self.flexpswitch)
        self.flgainlay.insertWidget(0 , self.flgainswitch)

        ## making external switch botton
        self.exTswitch = SwitchButton(self, 'On', 10, 'Off', 31, 60)
        self.cameraPar.addWidget(self.exTswitch , 2,1)
        
        self.exTswitch.toggled.connect(lambda c: self.ETrig(c))

        ## making the graphs for histograms and placing them inside window
        self.brhist = MplCanvas(self, width=4 , height=2.5,  dpi=70)
        self.brlay.insertWidget(2 , self.brhist)

        self.flhist = MplCanvas(self, width=4 , height=2.5 ,  dpi=70)
        self.fllay.insertWidget(3 , self.flhist)

        ## getting the list of connected camera's SN
        SN = []
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        for device in devices:
            SN.append( device.GetSerialNumber() ) 

        #adding the serial numbers to list menus in ui
        self.brsn.addItems(SN)
        self.flsn.addItems(SN)
        self.flsn.setCurrentIndex(1) #poiting the fl to the second camera SN to avoid both pointing to the same camera

        # values keeping the image reverse 
        self.brXflip = False
        self.brYflip = True
        self.flXflip = True
        self.flYflip = True
        ## reading default config file
        self.conf = config.SetupConfig('config.txt') #config has it's own class
        self.exTswitch.setChecked(self.conf.exTrig)
        self.conf.update(self) #
        self.saveCon.clicked.connect(self.SaveCon)


        ## connects values of frame rate, duration is s and frames so that each change when the other changes
        self.fnum.setValue(int(self.duration.value() * self.frate.value()))
        self.duration.valueChanged.connect(lambda c: self.fnum.setValue(int(c * self.frate.value())))
        self.fnum.valueChanged.connect(lambda c: self.duration.setValue(float(c / self.frate.value())))
        self.fnum.valueChanged.connect(lambda c: self.duration.setValue(float(c / self.frate.value())))
        self.frate.valueChanged.connect(lambda c: self.fnum.setValue(int(c * self.duration.value())))
        # self.bar.setMaximum(self.fnum.value())
        # self.fnum.valueChanged.connect(lambda c: self.bar.setMaximum(c))

        ## values holding the state of the camera
        self.bmount = False
        self.fmount = False
        ## conecting the push bottons to their functions
        self.mount.clicked.connect(self.Mount)
        self.live.clicked.connect(self.Live)
        self.rec.clicked.connect(self.Rec)
        self.stop.clicked.connect(self.Stop)

        ## adjusting some parameters of the imageview from pyqtgraph module
        self.brview.ui.histogram.hide()
        self.brview.ui.roiBtn.hide()
        self.brview.ui.menuBtn.hide()
        #self.brview.view.setDefaultPadding(padding=0)

        self.flview.ui.histogram.hide()
        self.flview.ui.roiBtn.hide()
        self.flview.ui.menuBtn.hide()
        #self.flview.view.setDefaultPadding(padding=0)

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
        try: #try to close if it is already open
            self.bcamera.Close()
        except: 
            pass

        try:
            self.fcamera.Close()
        except: 
            pass

        self.exTswitch.setEnabled(False)
        try:
            binfo = pylon.DeviceInfo()
            binfo.SetSerialNumber(self.brsn.currentText())
            self.bcamera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(binfo))
            self.bcamera.Open()
            self.camStaticPar(self.bcamera)
            self.brsn.setStyleSheet('Background: rgb(170, 255, 127);')
            self.brexpswitch.toggled.connect(lambda c: self.BautoExp(c))
            self.brgainswitch.toggled.connect(lambda c: self.BautoGain(c))
            self.BautoExp(self.brexpswitch.isChecked())
            self.BautoGain(self.brgainswitch.isChecked())

            self.bcamera.ReverseX.SetValue(self.brXflip)
            self.bcamera.ReverseY.SetValue(self.brYflip)
            self.bcamera.LineSelector.SetValue("Line3")
            if self.exTswitch.isChecked():
                self.slave(self.bcamera)
            self.bmount = True
        except:
            self.brsn.setStyleSheet('Background: rgb(255, 170, 127);')
            self.bmount = False

        try:
            finfo = pylon.DeviceInfo()
            finfo.SetSerialNumber(self.flsn.currentText())
            self.fcamera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(finfo))
            self.fcamera.Open()
            self.camStaticPar(self.fcamera)
            self.flsn.setStyleSheet('Background: rgb(170, 255, 127);')
            self.flexpswitch.toggled.connect(lambda c: self.FautoExp(c))
            self.flgainswitch.toggled.connect(lambda c: self.FautoGain(c))
            self.FautoExp(self.flexpswitch.isChecked())
            self.FautoGain(self.flgainswitch.isChecked())

            self.fcamera.ReverseX.SetValue(self.flXflip)
            self.fcamera.ReverseY.SetValue(self.flYflip)
            self.slave(self.fcamera)
            self.fmount = True
        except:
            self.flsn.setStyleSheet('Background: rgb(255, 170, 127);')
            self.fmount = False

    ## recurrent function to update the imageview liveview windows
    def updateview(self):
        if self.bmount:
            if not self.exTswitch.isChecked():
                self.bcamera.AcquisitionFrameRate = self.frate.value()
            if self.brexpspin.isEnabled():
                self.bcamera.ExposureTime.SetValue(self.brexpspin.value())
            if self.brgainspin.isEnabled():
                self.bcamera.Gain.SetValue(self.brgainspin.value())


            if self.bcamera.IsGrabbing():
                bgrab = self.bcamera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if bgrab.GrabSucceeded():
                    self.bimg = bgrab.GetArray()               
                    self.brview.setImage(self.bimg.T , autoRange=False , autoLevels=False , levels=(0, 255))
                bgrab.Release()

        if self.fmount:
            if not self.exTswitch.isChecked():
                self.fcamera.AcquisitionFrameRate = self.frate.value()
            if self.flexpspin.isEnabled():
                self.fcamera.ExposureTime.SetValue(self.flexpspin.value())
            if self.flgainspin.isEnabled():
                self.fcamera.Gain.SetValue(self.flgainspin.value())

            if self.fcamera.IsGrabbing():
                fgrab = self.fcamera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if fgrab.GrabSucceeded():
                    self.fimg = fgrab.GetArray()               
                    self.flview.setImage(self.fimg.T , autoRange=False , autoLevels=False , levels=(0, 255))
                fgrab.Release()

    ## function which updates both histograms the 
    def update_hist(self):  
        if self.fmount:
            self.update_flhist()
        if self.bmount:
            self.update_brhist()

        if not self.brexpspin.isEnabled():
            self.brexpspin.setValue(int(self.bcamera.ExposureTime.GetValue() ) )
        if not self.brgainspin.isEnabled():
            self.brgainspin.setValue(int (self.bcamera.Gain.GetValue()) )

        if not self.flexpspin.isEnabled():
            self.flexpspin.setValue(int(self.fcamera.ExposureTime.GetValue() ) )
        if not self.flgainspin.isEnabled():
            self.flgainspin.setValue(int (self.fcamera.Gain.GetValue()) )

    ## function which starts the liveview
    def Live(self):
        self.isStopped = False
        if not self.bmount and not self.fmount:
            self.Mount()
        try:
            self.fcamera.StopGrabbing()
        except:
            pass
        try:
            self.bcamera.StopGrabbing()
        except:
            pass

        self.exTswitch.setEnabled(False)
        if self.fmount:
            self.fcamera.AcquisitionFrameRate = self.frate.value()
            self.fcamera.LineMode.SetValue("Input")
            self.fcamera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        if self.bmount:
            self.bcamera.AcquisitionFrameRate = self.frate.value()

            if self.exTswitch.isChecked():
                self.bcamera.LineMode.SetValue("Input")
            else:
                self.bcamera.LineMode.SetValue("Output")
            self.bcamera.LineSource.SetValue("ExposureActive")
            self.bcamera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


        self.SaveCon()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateview)
        self.timer.start(int(1000/25))
        self.live.setStyleSheet('Background: rgb(170, 255, 127);')

        self.htimer = QtCore.QTimer(self)
        self.htimer.timeout.connect(self.update_hist)
        self.htimer.start(int(1000/5))

    ## function which starts the recording
    def Rec(self):
        try:
            self.htimer.stop()
            self.timer.stop()
            self.live.setStyleSheet('')
        except:
            pass
        path = self.spath.text()

        if not self.bmount and not self.fmount:
            self.Mount()
        try:
            self.fcamera.StopGrabbing()
        except:
            pass
        try:
            self.bcamera.StopGrabbing()
        except:
            pass

        self.exTswitch.setEnabled(False)
        self.rec.setEnabled(False)
        if self.fmount:
            if self.flexpspin.isEnabled():
                self.fcamera.ExposureTime.SetValue(self.flexpspin.value())
            if self.flgainspin.isEnabled():
                self.fcamera.Gain.SetValue(self.flgainspin.value())
            self.fcamera.AcquisitionFrameRate = self.frate.value()
            self.fcamera.LineMode.SetValue("Input")
            self.fcamera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            # self.fcamera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        if self.bmount:
            if self.brexpspin.isEnabled():
                self.bcamera.ExposureTime.SetValue(self.brexpspin.value())
            if self.brgainspin.isEnabled():
                self.bcamera.Gain.SetValue(self.brgainspin.value())
            self.bcamera.AcquisitionFrameRate = self.frate.value()
            if self.exTswitch.isChecked():
                self.bcamera.LineMode.SetValue("Input")
            else:
                self.bcamera.LineMode.SetValue("Output")
            self.bcamera.LineSource.SetValue("ExposureActive")
            self.bcamera.StartGrabbing(pylon.GrabStrategy_OneByOne)


        self.rec.setStyleSheet('Background: rgb(255, 170, 127);')
        self.counter.setStyleSheet('color: rgb(255, 255, 255); background-color: rgb(0, 0, 0);')
        self.Save()
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
        self.brpath = filename.Bpath(path)
        self.flpath = filename.Fpath(path)
        frate = self.frate.value()
        fnum = self.fnum.value()

        if frate < 20:
            sk = 1
        else:
            sk = frate // 20

        if self.fmount:
            Ftif = tifffile.TiffWriter(self.flpath, bigtiff=True)
        if self.bmount:
            Btif = tifffile.TiffWriter(self.brpath, bigtiff=True)
        ran = np.arange(1 , fnum+1)
        for i in ran:
            if self.isStopped:
                self.isStopped = False
                break
            if self.bmount:
                bgrab = self.bcamera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if bgrab.GrabSucceeded():
                    bimg = bgrab.GetArray()
                    btimestamp = bgrab.GetTimeStamp() //1000000
                    bmetad = {'timestamp': str(btimestamp)}
                    Btif.save(bimg, compression=0, metadata=bmetad, contiguous=False)
                    if i%sk == 0:
                        QApplication.processEvents()
                        self.brview.setImage(bimg.T , autoRange=False , autoLevels=False , levels=(0, 255))
                        self.counter.setText(str(i))

            if self.fmount:
                fgrab = self.fcamera.RetrieveResult(3000, pylon.TimeoutHandling_Return)
                if fgrab.GrabSucceeded():
                    fimg = fgrab.GetArray()
                    ftimestamp = fgrab.GetTimeStamp() //1000000
                    fmetad = {'timestamp': str(ftimestamp)}
                    Ftif.save(fimg, compression=0, metadata=fmetad, contiguous=False)
                    if i%sk == 0:
                        self.flview.setImage(fimg.T , autoRange=False , autoLevels=False , levels=(0, 255))

        # self.bar.setValue(i)
        if self.fmount:
            Ftif.close()
        if self.bmount:
            Btif.close()
        self.SaveNote()
        self.counter.setStyleSheet('color: rgb(0, 255, 0); background-color: rgb(0, 0, 0);')


    ## saves the config file
    def SaveCon(self):
        self.conf.save(self)
        self.SaveNote()
        # self.pbar.setValue(200)
        
    ##updates fluorescent histogram
    def update_flhist(self):
        self.flhist.axes.clear()
        self.flhist.axes.set_xlim([0 , 255])
        self.flhist.axes.set_ylim([0 , 1.1])
        # self.fimg = plt.imread('1.jpg')

        y, x = np.histogram(self.fimg, bins=np.arange(0, 255, 10), density=True)
        x = x[:-1] + np.diff(x)

        #imi = pg.ImageItem(self.fimg)
        #x , y = imi.getHistogram(bins='auto', step='auto', perChannel=False, targetImageSize=200, targetHistogramSize=500)
        y = y / y.max()
        self.flhist.axes.plot(x , y)
        self.flhist.axes.fill_between(x , 0 , y , alpha=0.5)
        self.flhist.draw()
    ## update brightfield histogram
    def update_brhist(self):
        self.brhist.axes.clear()
        self.brhist.axes.set_xlim([0 , 255])
        self.brhist.axes.set_ylim([0 , 1.1])
        # self.bimg = plt.imread('1.jpg')

        y, x = np.histogram(self.bimg, bins=np.arange(0, 255, 10), density=True)
        x = x[:-1] + np.diff(x)
        #imi = pg.ImageItem(self.bimg)
        #x , y = imi.getHistogram(bins='auto', step='auto', perChannel=False, targetImageSize=200, targetHistogramSize=500)
        y = y / y.max()
        self.brhist.axes.plot(x , y)
        self.brhist.axes.fill_between(x , 0 , y , alpha=0.5)
        self.brhist.draw()
    
    ## turns the brightfield camera autoexp on or off
    def BautoExp(self , t):
        if t == True:
            self.bcamera.ExposureAuto = "Continuous"
            self.brexpslide.setValue(int(self.bcamera.ExposureTime.GetValue() ) )
            self.brexpslide.setEnabled(False)
            self.brexpspin.setEnabled(False)
        if t == False:
            self.bcamera.ExposureAuto = "Off"
            self.brexpslide.setEnabled(True)
            self.brexpspin.setEnabled(True)
    ## turns the brightfield camera autogain on or off
    def BautoGain(self , t):
        if t == True:
            self.bcamera.GainAuto = "Continuous"
            self.brgainslide.setValue(int (self.bcamera.Gain.GetValue()) )
            self.brgainslide.setEnabled(False)
            self.brgainspin.setEnabled(False)
        if t == False:
            self.bcamera.GainAuto = "Off"
            self.brgainslide.setEnabled(True)
            self.brgainspin.setEnabled(True)
    ## turns the fluorescent camera autoexp on or off
    def FautoExp(self , t):
        if t == True:
            self.fcamera.ExposureAuto = "Continuous"
            self.flexpslide.setValue(int(self.fcamera.ExposureTime.GetValue() ) )
            self.flexpslide.setEnabled(False)
            self.flexpspin.setEnabled(False)
        if t == False:
            self.fcamera.ExposureAuto = "Off"
            self.flexpslide.setEnabled(True)
            self.flexpspin.setEnabled(True)
    ## turns the fluorescent camera autogain on or off
    def FautoGain(self , t):
        if t == True:
            self.fcamera.GainAuto = "Continuous"
            self.flgainslide.setValue(int (self.fcamera.Gain.GetValue()) )
            self.flgainslide.setEnabled(False)
            self.flgainspin.setEnabled(False)
        if t == False:
            self.fcamera.GainAuto = "Off"
            self.flgainslide.setEnabled(True)
            self.flgainspin.setEnabled(True)

    ##function where repetetive camera settings is being applied. to avoid redundency in the code
    def camStaticPar(self , cam):
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionMode.SetValue("Continuous")
        cam.ExposureAuto = "Off"
        cam.GainAuto = "Off"
        cam.AutoTargetBrightness = 0.5
        cam.MaxNumBuffer = 21
        # cam.GetStreamGrabberParams().MaxTransferSize.SetValue(1048576)
        cam.ChunkModeActive = True
        cam.ChunkSelector = "Timestamp"
        cam.ChunkEnable = True
        cam.ChunkSelector = "Gain"
        cam.ChunkEnable = True
        cam.ChunkSelector = "ExposureTime"
        cam.ChunkEnable = True

    ## set the settings specific to the slave camera
    def slave(self, cam):
        cam.LineSelector.SetValue("Line3")
        cam.TriggerSelector.SetValue("FrameStart")
        cam.TriggerMode.SetValue("On")
        #self.fcamera.TriggerDelay.SetValue(30)
        cam.ExposureMode.SetValue("Timed")
        cam.TriggerActivation.SetValue("RisingEdge")
        cam.TriggerSource.SetValue("Line3")

    ## function to enable and diable the external trigger. takes bolean as argument
    def ETrig(self , t):
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
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly
        path = QFileDialog.getExistingDirectory(self, 'Open a folder','', options=options)
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
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    app.setStyle('Fusion')
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()








