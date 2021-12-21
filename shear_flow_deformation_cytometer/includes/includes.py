import configparser
import copy
import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtCore import QSettings
import imageio
import numpy as np
import pandas as pd
import tqdm
import argparse
from shear_flow_deformation_cytometer.detection.includes.UNETmodel import weights_url
class Dialog(QFileDialog):
    def __init__(self, title="open file", filetype="", mode="file", settings_name="__"):
        super().__init__()

        self.settings_name = settings_name
        self.load_seetings()
        self.filter = filetype
        self.mode = mode
        self.title = title

    def openFile(self):
        if self.mode == "file":
            file = self.getOpenFileName(caption=self.title, filter=self.filter)[0]
        elif self.mode == "dir":
            file = self.getExistingDirectory(caption=self.title)
        else:
            return
        self.update_settings(file)
        return file

    def load_seetings(self):
        self.settings = QSettings("deformation_cytometer", self.settings_name)
        if "location" in self.settings.allKeys():
            self.location = self.settings.value("location")
            self.setDirectory(self.location)

    def update_settings(self, file):
        try:
           self.settings.setValue("location", os.path.split(file)[0])
        except Exception as e:
            print(e)

def read_args_pipeline():
    # defining and reading command line arguments for detect_cells.py
    network_weight = weights_url
    file = None
    irregularity_threshold = 1.3
    solidity_threshold = 0.7
    # read arguments if arguments are provided and if not executed from the pycharm console
    if len(sys.argv) > 1 and not sys.argv[0].endswith("pydevconsole.py"):
        parser = argparse.ArgumentParser()
        parser.add_argument('file', default=None, help='specify an input file or folder') # positional argument
        parser.add_argument('-n', '--network_weight', default="", help='provide an external the network weight file')
        parser.add_argument('-r', '--irregularity_filter', type=float, default=1.06, help='cells with larger irregularity (deviation from'
                                                                                'elliptical shape) are excluded')
        parser.add_argument('-s', '--solidity_filter', type=float, default=0.96, help='cells with smaller solidity are excluded')
        args = parser.parse_args()
        file = args.file
        network_weight = args.network_weight if args.network_weight.endswith(".h5") else None
        irregularity_threshold = args.irregularity_filter
        solidity_threshold = args.solidity_filter

    return file, network_weight, irregularity_threshold, solidity_threshold

def read_args_detect_cells():
    # defining and reading command line arguments for detect_cells.py
    network_weight = weights_url
    file = None
    # read arguments if arguments are provided and if not executed from the pycharm console
    if len(sys.argv) > 1 and not sys.argv[0].endswith("pydevconsole.py"):
        parser = argparse.ArgumentParser()
        parser.add_argument('file', default=None, help='specify an input file or folder') # positional argument
        parser.add_argument('-n', '--network_weight', default=None, help='provide an external the network weight file')
        args = parser.parse_args()
        file = args.file
        network_weight = args.network_weight if args.network_weight.endswith(".h5") else None

    return file, network_weight

def read_args_evaluate():
    # defining and reading command line arguments for strain_vs_stress_clean.py
    file = None
    irregularity_threshold = 1.06
    solidity_threshold = 0.96
    # read arguments if arguments are provided and if not executed from the pycharm console
    if len(sys.argv) > 1 and not sys.argv[0].endswith("pydevconsole.py"):
        parser = argparse.ArgumentParser()
        parser.add_argument('file', default=None, help='specify an input file or folder') # positional argument
        parser.add_argument('-r', '--irregularity_filter', type=float, default=1.06, help='cells with larger irregularity (deviation from'
                                                                                'elliptical shape) are excluded')
        parser.add_argument('-s', '--solidity_filter', type=float, default=0.96, help='cells with smaller solidity are excluded')

        args = parser.parse_args()
        file = args.file
        irregularity_threshold = args.irregularity_filter
        solidity_threshold = args.solidity_filter

    return file, irregularity_threshold, solidity_threshold

def read_args_tank_treading():
    # defining and reading command line arguments for strain_vs_stress_clean.py
    file = None
    # read arguments if arguments are provided and if not executed from the pycharm console
    if len(sys.argv) > 1 and not sys.argv[0].endswith("pydevconsole.py"):
        parser = argparse.ArgumentParser()
        parser.add_argument('file', default=None, help='specify an input file or folder') # positional argument
        args = parser.parse_args()
        file = args.file

    return file



def getInputFile(filetype="video file (*.tif *.avi)", settings_name="", video=None):


    if video is None:
        if len(sys.argv) >= 2:
            return sys.argv[1]
        else:
            # select video file
            app = QApplication(sys.argv)
            video = Dialog(title="select the data file", filetype=filetype,
                           mode="file", settings_name=settings_name).openFile()
            if video == '':
                print('empty')
                sys.exit()
    return video


def getInputFolder(settings_name=""):
    # if there are command line parameters, we use the provided folder
    if len(sys.argv) >= 2:
        parent_folder = sys.argv[1]
    # if not we ask for a folder
    else:
        # %% select video file
        app = QApplication(sys.argv)
        parent_folder = Dialog(title="select the data folder", mode="dir", settings_name=settings_name).openFile()
        if parent_folder == '':
            print('empty')
            sys.exit()
    return parent_folder

# %% open and read the config file
def getConfig(configfile):
    configfile = str(configfile)
    if configfile.endswith("_result.txt"):
        configfile = configfile.replace("_result.txt", "_config.txt")
    if configfile.endswith("_evaluated_new.csv"):
        configfile = configfile.replace("_evaluated_new.csv", "_config.txt")
    if configfile.endswith(".tif"):
        configfile = configfile.replace(".tif", "_config.txt")
    if configfile.endswith("_addon_evaluated.csv"):
        configfile = configfile.replace("_addon_evaluated.csv", "_addon_config.txt")
    if not Path(configfile).exists():
        raise IOError(f"Config file {configfile} does not exist.")

    config = configparser.ConfigParser()
    config.read(configfile)

    config_data = {}
    # print("config", config, configfile)

    config_data["file_data"] = configfile.replace("_config.txt", "_result.txt")
    config_data["file_tif"] = configfile.replace("_config.txt", ".tif")
    config_data["file_config"] = configfile

    config_data["magnification"] = float(config['MICROSCOPE']['objective'].split()[0])
    config_data["coupler"] = float(config['MICROSCOPE']['coupler'].split()[0])
    config_data["camera_pixel_size"] = float(config['CAMERA']['camera pixel size'].split()[0])
    config_data["pixel_size"] = config_data["camera_pixel_size"] / (
                config_data["magnification"] * config_data["coupler"])  # in meter
    config_data["px_to_um"] = config_data["pixel_size"]
    config_data["pixel_size_m"] = config_data["pixel_size"] * 1e-6  # in um
    config_data["channel_width_px"] = float(config['SETUP']['channel width'].split()[0]) / config_data[
        "pixel_size"]  # in pixels
    config_data["imaging_pos_mm"] = float(config['SETUP']['imaging position after inlet'].split()[0]) * 10  # in mm

    config_data["pressure_pa"] = float(config['SETUP']['pressure'].split()[0]) * 1000  # applied pressure (in Pa)

    config_data["channel_width_m"] = float(config['SETUP']['channel width'].split()[0]) * 1e-6
    config_data["channel_length_m"] = float(config['SETUP']['channel length'].split()[0]) * 1e-2

    config_data["cell_treatment"] = config['CELL']['treatment']
    return config_data


def getData(datafile):
    if str(datafile).endswith(".tif"):
        datafile = str(datafile).replace(".tif", "_result.txt")
    datafile = str(datafile)
    # %% import raw data
    data = np.genfromtxt(datafile, dtype=float, skip_header=2)

    data = pd.DataFrame({
        "frames": data[:, 0].astype(int),
        "x": data[:, 1],
        "y": data[:, 2],
        "rp": data[:, 3],
        "long_axis": data[:, 4],
        "short_axis": data[:, 5],
        "angle": data[:, 6],
        "irregularity": data[:, 7],
        "solidity": data[:, 8],
        "sharpness": data[:, 9],
        "timestamp": data[:, 10],
    })
    return data


# %%  compute average (flatfield) image
def getFlatfield(video, flatfield, force_recalculate=False):
    if os.path.exists(flatfield) and not force_recalculate:
        im_av = np.load(flatfield)
    else:
        vidcap = imageio.get_reader(video)
        print("compute average (flatfield) image")
        count = 0
        progressbar = tqdm.tqdm(vidcap)
        progressbar.set_description("computing flatfield")
        for image in progressbar:
            if len(image.shape) == 3:
                image = image[:, :, 0]
            if count == 0:
                im_av = copy.deepcopy(image)
                im_av = np.asarray(im_av)
                im_av.astype(float)
            else:
                im_av = im_av + image.astype(float)
            count += 1
        im_av = im_av / count
        try:
            np.save(flatfield, im_av)
        except PermissionError as err:
            print(err)
    return im_av


def convertVideo(input_file, output_file=None, rotate=True):
    if output_file is None:
        basefile, extension = os.path.splitext(input_file)
        new_input_file = basefile + "_raw" + extension
        os.rename(input_file, new_input_file)
        output_file = input_file
        input_file = new_input_file

    if input_file.endswith(".tif"):
        vidcap = imageio.get_reader(input_file)
        video = imageio.get_writer(output_file)
        count = 0
        for im in vidcap:
            print(count)
            if len(im.shape) == 3:
                im = im[:, :, 0]
            if rotate:
                im = im.T
                im = im[::-1, ::]

            video.append_data(im)
            count += 1

        return

    vidcap = cv2.VideoCapture(input_file)
    video = imageio.get_writer(output_file, quality=7)
    count = 0
    success = True
    while success:
        success, im = vidcap.read()
        print(count)
        if success:
            if len(im.shape) == 3:
                im = im[:, :, 0]
            if rotate:
                im = im.T
                im = im[::-1, ::]

            video.append_data(im)
            count += 1
