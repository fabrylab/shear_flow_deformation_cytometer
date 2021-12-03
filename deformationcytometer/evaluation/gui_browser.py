import sys

# Setting the Qt bindings for QtPy
import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers
import matplotlib
from pathlib import Path
import numpy as np
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import yaml
import QExtendedGraphicsView

from deformationcytometer.evaluation.helper_functions import getMeta, load_all_data_new, plot_velocity_fit, plotDensityScatter, plot_density_hist, plotBinnedData, stress_strain_fit, get2Dhist_k_alpha, getGp1Gp2fit_k_alpha, getGp1Gp2fit3_k_alpha, get2Dhist_k_alpha

def kill_thread(thread):
    """
    thread: a threading.Thread object
    """
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)

def pathParts(path):
    if path.parent == path:
        return [path]
    return pathParts(path.parent) + [path]


class ImageView(QtWidgets.QWidget):
    data_loaded_event = QtCore.Signal(str, object, object, object)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QExtendedGraphicsView.QExtendedGraphicsView()
        self.view.setMinimumWidth(300)
        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.view.origin)
        layout.addWidget(self.view)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.sliderChanged)
        layout.addWidget(self.slider)
        self.ellipses = []
        self.pen = QtGui.QPen(QtGui.QColor("magenta"), 3)
        self.pen.setCosmetic(True)
        self.data_loaded_event.connect(self.data_loaded)

    thread = None
    def selected(self, filename):
        import threading
        if self.thread is not None:
            kill_thread(self.thread)
            self.thread = None
        self.im = None
        self.filename = filename
        if self.filename.endswith(".tif"):
            self.thread = threading.Thread(target=self.load_image, args=(filename,), daemon=True)
            self.thread.start()
        self.disable()

    def load_image(self, filename):
        print("loading", filename, "...")
        im = imageio.get_reader(filename)
        im.get_data(0)
        data, config = load_all_data_new(filename, do_excude=False, do_group=False)
        self.data_loaded_event.emit(filename, im, data, config)
        print("loaded")

    def data_loaded(self, filename, im, data, config):
        if filename == self.filename:
            self.im = im
            self.data = data
            self.config = config
            self.thread = None
            self.slider.setRange(0, im.get_length()-1)
            self.slider.setEnabled(True)
            self.sliderChanged()

    def disable(self):
        self.slider.setEnabled(False)
        self.deleteEllipses()
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(np.zeros((4, 4)))))

    def sliderChanged(self):
        if self.im is None:
            return
        frame = self.slider.value()
        im = self.im.get_data(frame)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view.setExtend(im.shape[1], im.shape[0])

        data = self.data.query(f"frames == {frame}")
        self.deleteEllipses()
        for i, d in data.iterrows():
            self.addEllipse(d.x, d.y, d.long_axis / self.config["pixel_size"], d.short_axis / self.config["pixel_size"], d.angle,)

    def deleteEllipses(self):
        for ellipse in self.ellipses:
            ellipse.scene().removeItem(ellipse)
        self.ellipses = []

    def addEllipse(self, x, y, w, h, angle):
        ellipse = QtWidgets.QGraphicsEllipseItem(x - w/2, y - h/2, w, h, self.view.origin,)
        ellipse.setTransformOriginPoint(x, y)
        ellipse.setRotation(angle)
        ellipse.setPen(self.pen)
        self.ellipses.append(ellipse)

class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        try:
            self.setWindowIcon(qta.icon("mdi.folder-pound-outline"))
        except Exception:
            pass

        # QSettings
        self.settings = QtCore.QSettings("DeformationCytometer", "DeformationCytometer")

        self.setMinimumWidth(1200)
        self.setMinimumHeight(400)
        self.setWindowTitle("DeformationCytometer Viewer")

        hlayout = QtWidgets.QHBoxLayout(self)

        self.browser = Browser()
        #hlayout.addWidget(self.browser)

        self.plot = MeasurementPlot()
        #hlayout.addWidget(self.plot)

        self.text = MetaDataEditor()
        #hlayout.addWidget(self.text)
        self.view = ImageView()

        self.splitter_filebrowser = QtWidgets.QSplitter()
        self.splitter_filebrowser.addWidget(self.browser)
        self.splitter_filebrowser.addWidget(self.plot)
        self.splitter_filebrowser.addWidget(self.text)
        self.splitter_filebrowser.addWidget(self.view)
        hlayout.addWidget(self.splitter_filebrowser)
        #splitter_filebrowser.setStretchFactor(0, 2)
        #splitter_filebrowser.setStretchFactor(1, 4)

        self.browser.signal_selection_changed.connect(self.selected)

    def selected(self, name):
        self.view.selected(name)
        self.text.selected(name)
        self.plot.selected(name)


class MeasurementPlot(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.hlayout = QtWidgets.QVBoxLayout(self)
        self.hlayout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibWidget(self)
        plt.clf()
        self.hlayout.addWidget(self.canvas)
        self.tools = NavigationToolbar(self.canvas, self)
        self.hlayout.addWidget(self.tools)

    def selected(self, name):
        plt.clf()
        if name.endswith(".tif"):
            data, config = load_all_data_new(name, do_excude=False)

            pair_2dmode = get2Dhist_k_alpha(data)
            #pair_2dmode = getGp1Gp2fit3_k_alpha(data)

            from scipy.special import gamma
            def fit(omega, k, alpha):
                omega = np.array(omega)
                G = k * (1j * omega) ** alpha * gamma(1 - alpha)# + 1j * omega * mu
                return np.real(G), np.imag(G)

            plt.subplot(3, 3, 1)
            plot_velocity_fit(data)
            plt.text(0, 0, f"eta0: {data.iloc[0].eta0:.2f}\ntau: {data.iloc[0].tau:.4f}\ndelta: {data.iloc[0].delta:.3f}", ha="left", va="bottom")

            plt.subplot(3, 3, 2)
            plt.axline([0,0], slope=1, color="k")
            colors = np.array([matplotlib.colors.to_rgba("C0")]*len(data.tt_r2))
            colors[:, 3] = data.tt_r2
            plt.scatter(data.omega, data.omega_weissenberg, s=1, color=colors)
            plt.xlabel("measured angular frequency (rad/s)")
            plt.ylabel("fitted angular frequency (rad/s)")

            plt.subplot(3, 3, 3)
            plotDensityScatter(data.stress, data.epsilon)
            plotBinnedData(data.stress, data.epsilon, bins=np.arange(0, 300, 10))
            stress, strain = stress_strain_fit(data, pair_2dmode[0], pair_2dmode[1])
            plt.plot(stress, strain, "-k")
            plt.xlabel("stress (Pa)")
            plt.ylabel("strain")

            plt.subplot(3, 3, 4)
            plt.loglog(data.omega_weissenberg, data.w_Gp1, "o", alpha=0.25, ms=1)
            plt.loglog(data.omega_weissenberg, data.w_Gp2, "o", alpha=0.25, ms=1)

            xx = [10**np.floor(np.log10(np.min(data.w_Gp1))), 10**np.ceil(np.log10(np.max(data.w_Gp1)))]
            plt.plot(xx, fit(xx, *pair_2dmode)[0], "k-", lw=1.)
            plt.plot(xx, fit(xx, *pair_2dmode)[1], "k--", lw=1.)

            plt.ylabel("G' / G'' (Pa)")
            plt.xlabel("angular frequency (rad/s)")
            plt.xlim(*np.percentile(data.omega_weissenberg, [0.1, 99.9]))

            ax = plt.subplot(3, 3, 5)
            plt.cla()
            plt.xlim(0, 4)
            plot_density_hist(np.log10(data.w_k_cell), color="C0")
            plt.axvline(np.log10(pair_2dmode[0]), color="k")
            plt.xlabel("stiffness k (Pa)")
            plt.ylabel("relative density")
            ax.xaxis.set_major_formatter(lambda x, pos : f'$10^{{{int(x)}}}$' if x % 1 == 0 else f'$10^{{{x}}}$')

            plt.subplot(3, 3, 6)
            plt.cla()
            plt.xlim(0, 1)
            plot_density_hist(data.w_alpha_cell, color="C1")
            plt.ylabel("relative density")
            plt.xlabel("fluidity $\\alpha$")
            plt.axvline(pair_2dmode[1], color="k")

            ax = plt.subplot(3, 3, 7)
            plt.cla()
            plotDensityScatter(np.log10(data.w_k_cell), data.w_alpha_cell)
            plt.axvline(np.log10(pair_2dmode[0]), color="k"); plt.axhline(pair_2dmode[1], color="k", label="2dmode")
            #plt.legend()
            plt.xlabel("stiffness k (Pa)")
            plt.ylabel("fluidity $\\alpha$")
            ax.xaxis.set_major_formatter(lambda x, pos : f'$10^{{{int(x)}}}$' if x % 1 == 0 else f'$10^{{{x}}}$')

            plt.xlim(1, 3)
            plt.ylim(0, .5)

            plt.tight_layout()
            #plt.plot(data.rp, data.vel)




        self.canvas.draw()


class MetaDataEditor(QtWidgets.QWidget):
    yaml_file = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.label0 = QtWidgets.QLabel("Config")
        hlayout.addWidget(self.label0)
        self.text0 = QtWidgets.QPlainTextEdit()
        self.text0.setReadOnly(True)
        self.text0.setToolTip("Config Contents")
        hlayout.addWidget(self.text0)

        self.label = QtWidgets.QLabel("Meta data with inherited meta data from parent folders")
        hlayout.addWidget(self.label)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setToolTip("Meta data from parent folders")
        hlayout.addWidget(self.text)

        self.label = QtWidgets.QLabel("Edit Meta data for this file/folder")
        hlayout.addWidget(self.label)
        self.text2 = QtWidgets.QPlainTextEdit()
        self.text2.textChanged.connect(self.save)
        self.text2.setToolTip("Meta data from current folder/file. Can be edited and will be automatically saved")
        hlayout.addWidget(self.text2)

        self.name = QtWidgets.QLineEdit()
        self.name.setReadOnly(True)
        self.name.setToolTip("The current folder/file.")
        hlayout.addWidget(self.name)

    def save(self):
        if self.yaml_file is not None:
            with open(self.yaml_file, "w") as fp:
                fp.write(self.text2.toPlainText())

    def selected(self, name):
        meta = getMeta(name)
        self.name.setText(name)

        self.text.setPlainText(yaml.dump(meta))

        self.config_file = None
        if name.endswith(".tif"):
            config_file = Path(name.replace(".tif", "_config.txt"))
            self.text0.setPlainText(Path(config_file).read_text())
        else:
            self.text0.setPlainText("")

        self.yaml_file = None
        if name.endswith(".tif"):
            yaml_file = Path(name.replace(".tif", "_meta.yaml"))
        else:
            yaml_file = Path(name) / "meta.yaml"

        if yaml_file.exists():
            with yaml_file.open() as fp:
                self.text2.setPlainText(fp.read())
        else:
            self.text2.setPlainText("")
        self.yaml_file = yaml_file


class Browser(QtWidgets.QTreeView):
    signal_selection_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings("fabrylab", "flowcytometer browser")

        # self.setCentralWidget(self.frame)
        #hlayout = QtWidgets.QVBoxLayout(self)

        """ browser"""
        self.dirmodel = QtWidgets.QFileSystemModel()
        # Don't show files, just folders
        # self.dirmodel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)
        self.dirmodel.setNameFilters(["*.tif"])
        self.dirmodel.setNameFilterDisables(False)
        self.folder_view = self#QtWidgets.QTreeView(parent=self)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.activated[QtCore.QModelIndex].connect(self.clicked)
        # self.folder_view.selected[QtCore.QModelIndex].connect(self.clicked)

        # Don't show columns for size, file type, and last modified
        self.folder_view.setHeaderHidden(True)
        self.folder_view.hideColumn(1)
        self.folder_view.hideColumn(2)
        self.folder_view.hideColumn(3)

        self.selectionModel = self.folder_view.selectionModel()

        #hlayout.addWidget(self.folder_view)

        if self.settings.value("browser/path"):
            self.set_path(self.settings.value("browser/path"))
        else:
            self.set_path(r"\\131.188.117.96\biophysDS")

        self.setAcceptDrops(True)

        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(False)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            url = Path(url.path())
            if url.is_dir() or url.suffix == ".tif":
                event.accept()
                return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():
            url = Path(url.path())
            if url.is_dir() or url.suffix == ".tif":
                self.set_path(url)

    def set_path(self, path):
        path = Path(path)
        self.dirmodel.setRootPath(str(path.parent))
        for p in pathParts(path):
            self.folder_view.expand(self.dirmodel.index(str(p)))
        self.folder_view.setCurrentIndex(self.dirmodel.index(str(path)))
        print("scroll to ", str(path), self.dirmodel.index(str(path)))
        self.folder_view.scrollTo(self.dirmodel.index(str(path)))

    def clicked(self, index):
        # get selected path of folder_view
        index = self.selectionModel.currentIndex()
        dir_path = self.dirmodel.filePath(index)
        print(dir_path)
        self.settings.setValue("browser/path", dir_path)
        self.signal_selection_changed.emit(dir_path)

        if dir_path.endswith(".npz"):
            print("################# load", dir_path)
            self.loadFile(dir_path)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # set an application id, so that windows properly stacks them in the task bar
    if sys.platform[:3] == 'win':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('fabrybiophysics.deformationcytometer_browser')  # arbitrary string
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
