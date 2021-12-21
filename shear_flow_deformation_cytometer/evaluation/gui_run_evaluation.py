import sys

# Setting the Qt bindings for QtPy
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from gui import QtShortCuts


""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        try:
            self.setWindowIcon(qta.icon("mdi.folder-pound-outline"))
        except Exception:
            pass

        # QSettings
        self.settings = QtCore.QSettings("fabrylab", "DeformationCytometer")

        #self.setMinimumWidth(1200)
        #self.setMinimumHeight(400)
        self.setWindowTitle("DeformationCytometer Evaluate")

        with QtShortCuts.QVBoxLayout(self):
            self.input = QtShortCuts.QInputFolder(settings=self.settings, settings_key="evaluate_folder")
            self.network = QtShortCuts.QInputFilename(settings=self.settings, settings_key="custom_network")
            self.irregularity_threshold = QtShortCuts.QInputNumber('irregularity', 1.06, settings=self.settings, settings_key="irregularity_threshold")
            self.solidity_threshold = QtShortCuts.QInputNumber('solidity', 0.96, settings=self.settings, settings_key="solidity_threshold")
            self.r_min = QtShortCuts.QInputNumber('r_min', 6, settings=self.settings, settings_key="r_min")
            self.button_run = QtShortCuts.QPushButton("run", self.run)

    def run(self):
        import subprocess
        subprocess.run([
            sys.executable,
            '../detection/detect_cells_multiprocess_pipe_batch.py',
            self.input.value(),
            "-n", self.network.value(),
            "-r", str(self.irregularity_threshold.value()),
            "-s", str(self.solidity_threshold.value()),
            "--rmin", str(self.r_min.value()),
        ])


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
