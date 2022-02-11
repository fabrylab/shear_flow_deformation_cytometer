import sys
from pathlib import Path
# Setting the Qt bindings for QtPy
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from shear_flow_deformation_cytometer.gui import QtShortCuts
from collections import defaultdict

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

tooltip_strings = defaultdict(str)
tooltip_strings["choose folder"] = "Select folder containing data to evaluate (.tiff)"
tooltip_strings["choose file"] = "Select a neural network weights file (.h5)"
tooltip_strings["irregularity"] = "Set the irregularity threshold to filter cells"
tooltip_strings["solidity"] = "Set the solidity threshold to filter cells"
tooltip_strings["r_min"] = "Set a minimum radius to filter cells"
tooltip_strings["force"] = "If activated reevaluates .tiff files even if corresponding .csv exists"


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        try:
            self.setWindowIcon(qta.icon("mdi.folder-pound-outline"))
        except Exception:
            pass

        # QSettings
        self.settings = QtCore.QSettings("fabrylab", "DeformationCytometer")

        self.setMinimumWidth(500)
        #self.setMinimumHeight(400)
        self.setWindowTitle("DeformationCytometer Evaluate")

        with QtShortCuts.QVBoxLayout(self):
            self.input = QtShortCuts.QInputFolder(name = 'Input folder   ',settings=self.settings, settings_key="evaluate_folder",tooltip=tooltip_strings["choose folder"])
            self.network = QtShortCuts.QInputFilename(name='Network (.h5)',settings=self.settings, settings_key="custom_network",tooltip=tooltip_strings["choose file"], existing=True)
            self.irregularity_threshold = QtShortCuts.QInputNumber('irregularity', 1.06, settings=self.settings, settings_key="irregularity_threshold",tooltip=tooltip_strings["irregularity"])
            self.solidity_threshold = QtShortCuts.QInputNumber('solidity', 0.96, settings=self.settings, settings_key="solidity_threshold",tooltip=tooltip_strings["solidity"])
            self.r_min = QtShortCuts.QInputNumber('r_min', 6, settings=self.settings, settings_key="r_min",tooltip=tooltip_strings["r_min"])
            self.force = QtShortCuts.QInputBool("Force reevaluation",False,tooltip=tooltip_strings["force"])
            self.button_run = QtShortCuts.QPushButton("run", self.run)

    def run(self):
        import subprocess
        self.close()
        subprocess.run([
            sys.executable,
            Path(__file__).parent / '../detection/detect_cells_multiprocess_pipe_batch.py',
            self.input.value(),
            "-n", self.network.value(),
            "-r", str(self.irregularity_threshold.value()),
            "-s", str(self.solidity_threshold.value()),
            "--rmin", str(self.r_min.value()),
            "-f", str(self.force.value())
        ])

def main():
    app = QtWidgets.QApplication(sys.argv)
    # set an application id, so that windows properly stacks them in the task bar
    if sys.platform[:3] == 'win':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'fabrybiophysics.deformationcytometer_browser')  # arbitrary string
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
