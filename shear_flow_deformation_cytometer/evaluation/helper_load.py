import glob
import json
import sys
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd

from shear_flow_deformation_cytometer.includes.includes import getConfig
from shear_flow_deformation_cytometer.evaluation.helper_functions import filterCenterCells

path_string = Union[str, Path]


def check_config_changes(config, evaluation_version, solidity_threshold, irregularity_threshold):
    # checking if evaluation version, solidity or regularity thresholds have changed or if flag
    # for new network evaluation is set to True

    checks = {"evaluation_version": evaluation_version, "solidity": solidity_threshold,
              "irregularity": irregularity_threshold, "network_evaluation_done": True}

    for key, value in checks.items():
        if key not in config:
            return True
        else:
            if config[key] != value:
                return True
    return False

def process_paths(name: Union[path_string, List[path_string]], filter_func: callable = None,
                  file_name: str = None) -> list:
    results = []
    # if the function is called with a list or tuple, iterate over those and join the results
    if isinstance(name, (tuple, list)):
        for n in name:
            results.extend(process_paths(n, filter_func=filter_func, file_name=file_name))
    else:
        # add the file_name pattern if there is one and if it is not already there
        if file_name is not None and Path(name).suffix != Path(file_name).suffix:
            name = Path(name) / "**" / file_name
        # if it is a glob pattern, add all matching elements
        if "*" in str(name):
            results = glob.glob(str(name), recursive=True)
        # or add the name directly
        elif Path(name).exists():
            results = [name]

        # filter results if a filter is provided
        if filter_func is not None:
            results = [n for n in results if filter_func(n)]

        # if nothing was found, try to give a meaningful error message
        if len(results) == 0:
            # get a list of all parent folders
            name = Path(name).absolute()
            hierarchy = []
            while name.parent != name:
                hierarchy.append(name)
                name = name.parent
            # iterate over the parent folders, starting from the lowest
            for path in hierarchy[::-1]:
                # check if the path exists (or is a glob pattern with matches)
                if "*" in str(path):
                    exists = len(glob.glob(str(path)))
                else:
                    exists = path.exists()
                # if it does not exist, we have found our problem
                if not exists:
                    target = f"No file/folder \"{path.name}\""
                    if "*" in str(path.name):
                        target = f"Pattern \"{path.name}\" not found"
                    source = f"in folder \"{path.parent}\""
                    if "*" in str(path.parent):
                        source = f"in any folder matching the pattern \"{path.parent}\""
                    print(f"WARNING: {target} {source}", file=sys.stderr)
                    break
    return results

def get_meta(filename: path_string, cached_meta_files: dict = None):
    import yaml
    if cached_meta_files is None:
        cached_meta_files = {}
    filename = Path(filename)
    # if the data is not cached yet
    if filename not in cached_meta_files:
        # get meta data from parent folder
        cached_meta_files[filename] = {}
        if filename.parent != filename:
            cached_meta_files[filename] = get_meta(filename.parent).copy()

        # find meta data filename
        if str(filename).endswith(".tif"):
            yaml_file = Path(str(filename).replace(".tif", "_meta.yaml"))
        elif str(filename).endswith("_evaluated_new.csv"):
            yaml_file = Path(str(filename).replace("_evaluated_new.csv", "_meta.yaml"))
        elif str(filename).endswith("_evaluated.csv"):
            yaml_file = Path(str(filename).replace("_evaluated.csv", "_meta.yaml"))
        else:
            yaml_file = filename / "meta.yaml"

        # load data from file and join with parent meta data
        if yaml_file.exists():
            with yaml_file.open() as fp:
                data = yaml.load(fp, Loader=yaml.SafeLoader)
            if data is not None:
                cached_meta_files[filename].update(data)

    # return the metadata
    return cached_meta_files[filename]


ureg = None


def convert_old_csv_to_new(data, config):
    data2 = pd.DataFrame()
    print(data.columns)

    if "frames" in data.columns:
        data2["frame"] = data.frames  # renamed to frame (without the plural s)
    data2["timestamp"] = data.timestamp
    data2["x"] = data.x
    data2["y"] = data.y
    data2["radial_position"] = data.rp  # renamed to radial_position
    data2["long_axis"] = data.long_axis
    data2["short_axis"] = data.short_axis
    data2["long_axis_px"] = data.long_axis / config["pixel_size"]
    data2["short_axis_px"] = data.short_axis / config["pixel_size"]
    data2["angle"] = data.angle
    data2["irregularity"] = data.irregularity
    data2["solidity"] = data.solidity
    # data2["sharpness"] = data.sharpness
    data2["measured_velocity"] = data.velocity  # renamed
    data2["cell_id"] = data.cell_id  # renamed
    data2["tt"] = data.tt
    data2["tt_r2"] = data.tt_r2
    data2["tt_omega"] = data.omega
    # data2["velocity_gradient"] = data.velocity_gradient
    # data2["velocity_fitted"] = data.velocity_fitted
    # data2["imaging_pos_mm"] = data.imaging_pos_mm
    data2["stress"] = data.stress
    # data2["stress_center"] = data.stress_center

    data2["strain"] = data.strain
    data2["area"] = data.area
    data2["pressure"] = data.pressure
    # data2["vel_fit_error"] = data.vel_fit_error
    data2["vel"] = data.vel
    data2["vel_grad"] = data.vel_grad
    data2["eta"] = data.eta
    data2["eta0"] = data.eta0
    data2["delta"] = data.delta
    data2["tau"] = data.tau
    data2["tt_mu1"] = data.mu1
    data2["tt_eta1"] = data.eta1
    data2["tt_Gp1"] = data.Gp1
    data2["tt_Gp2"] = data.Gp2
    data2["tt_k_cell"] = data.k_cell
    data2["tt_alpha_cell"] = data.alpha_cell
    data2["tt_epsilon"] = data.epsilon
    if "omega_weissenberg" not in data:
        def func(x, a, b):
            return x / 2 * 1 / (1 + (a * x) ** b)

        x = [0.113, 0.45]

        data2["omega"] = func(np.abs(data.vel_grad), *x)
    else:
        data2["omega"] = data.omega_weissenberg
    data2["Gp1"] = data.w_Gp1
    data2["Gp2"] = data.w_Gp2
    data2["k"] = data.w_k_cell
    data2["alpha"] = data.w_alpha_cell

    return data2


def load_all_data_new(input_path: Union[path_string, List[path_string]], pressure=None, do_group=True, add_units=False,
                      do_excude=True, cache=False) -> (pd.DataFrame, dict):
    import re
    import configparser

    if cache is True:
        import hashlib
        hash = hashlib.md5(str(input_path).encode()).hexdigest()
        cache_file = f"tmp_{hash}.csv"
        if Path(cache_file).exists():
            data = pd.read_csv(cache_file)
            return data, {}

    # convert paths when input is a tif file
    if isinstance(input_path, list):
        for i, path in enumerate(input_path):
            if str(path).endswith(".tif"):
                input_path[i] = str(path).replace(".tif", "_evaluated.csv")
    else:
        if str(input_path).endswith(".tif"):
            input_path = str(input_path).replace(".tif", "_evaluated.csv")

    unit_matcher = re.compile(r"(\d*\.?\d+)([^\d]+)$")

    def filter(file):
        try:
            config = getConfig(file)
        except OSError as err:
            print(err, file=sys.stderr)
            return False
        data_pressure = config['pressure_pa'] / 100_000
        if pressure is not None and data_pressure != pressure:
            print("filtered due to pressure")
            return False
        return True

    paths = process_paths(input_path, filter, "*_evaluated.csv")
    # paths = get_folders(input_path, pressure=pressure, repetition=repetition)
    data_list = []
    config = {}
    for index, file in enumerate(paths):
        if str(file).endswith('_addon_evaluated.csv'):
            output_file = Path(str(file))
            output_config_file = Path(str(output_file).replace("_addon_evaluated.csv", "_evaluated_config_new.txt"))
            output_config_file_raw = Path(str(output_file).replace("_addon_evaluated.csv", "_addon_config.txt"))
        if str(file).endswith('evaluated_new.csv'):
            output_file = Path(str(file))
            output_config_file = Path(str(output_file).replace("_evaluated_new.csv", "_evaluated_config_new.txt").replace(
                    "_evaluated_new_hand2.csv", "_evaluated_config_new_hand2.txt"))
            output_config_file_raw = Path(str(output_file).replace("_evaluated_new.csv", "_config.txt").replace("_evaluated_new_hand2.csv",
                                                                                      "_config.txt"))
        if str(file).endswith('evaluated.csv'):
            output_file = Path(str(file))
            output_config_file = Path(str(output_file).replace("_evaluated.csv", "_evaluated.json"))
            output_config_file_raw = Path(str(output_file).replace("_evaluated_new.csv", "_config.txt"))
        if str(file).endswith('.tif'):
            output_file = Path(str(file).replace(".tif", "_evaluated.csv"))
            output_config_file = Path(str(output_file).replace(".tif", "_evaluated.json"))
            output_config_file_raw = Path(str(output_file).replace(".tif", "_config.txt"))
        # measurement_datetime = datetime.datetime.strptime(Path(output_file).name[:19], "%Y_%m_%d_%H_%M_%S")
        # measurement_datetime = Path(output_file).name[:19]

        print("config", output_config_file)
        with output_config_file.open("r") as fp:
            config = json.load(fp)
            config["channel_width_m"] = 0.00019001261833616293

        data = pd.read_csv(output_file, skiprows=[1])

        if str(output_file).endswith('evaluated_new.csv'):
            data = convert_old_csv_to_new(data, config)
        if do_group is True:
            data = data.groupby(['cell_id'], as_index=False).mean()
            # filter the cells in the center
            data = filterCenterCells(data)

        data["filename"] = output_file.name
        # data["datetime"] = measurement_datetime
        # data["time_after_harvest"] = float(config_raw["CELL"]["time after harvest"].strip(" mins").strip(" min"))

        if add_units is True:
            import pint, pint_pandas
            global ureg

            if ureg is None:
                ureg = pint.UnitRegistry()
                ureg.define('frame = []')
                ureg.setup_matplotlib(True)
                ureg.define(f'cam_pixel = {config["pixel_size_m"]} * m = px')

        # add meta data
        meta = get_meta(output_file)
        for key, value in meta.items():
            if isinstance(value, str) and unit_matcher.match(value):
                if add_units is True:
                    try:
                        value = ureg(value)
                    except pint.errors.UndefinedUnitError:
                        value = float(unit_matcher.match(value).groups()[0])
                else:
                    value = float(unit_matcher.match(value).groups()[0])
            data[key] = value

        if "exclude" in meta and do_excude is True:
            if meta["exclude"] is True:
                print("excluding", output_file)
                continue
        data_list.append(data)

    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

    if add_units is True:
        import pint, pint_pandas

        units = {
            "timestamp": "ms",
            # "datetime": "dimensionless",
            # "time_after_harvest": "min",
            "frame": "frame",
            "x": "cam_pixel",
            "y": "cam_pixel",
            "rp": "µm",
            "long_axis": "µm",
            "short_axis": "µm",
            "angle": "deg",
            "irregularity": "dimensionless",
            "solidity": "dimensionless",
            "sharpness": "dimensionless",
            "velocity": "mm/s",
            "cell_id": "dimensionless",
            "tt": "rad/s",
            "tt_r2": "dimensionless",
            "omega": "rad/s",
            "velocity_gradient": "1/s",
            "velocity_fitted": "mm/s",
            "imaging_pos_mm": "mm",
            "stress": "Pa",
            "stress_center": "Pa",
            "strain": "dimensionless",
            "area": "µm**2",
            "pressure": "bar",
            "vel": "m/s",
            "vel_grad": "1/s",
            "eta": "Pa*s",
            "eta0": "Pa*s",
            "delta": "dimensionless",
            "tau": "s",
            "mu1": "Pa",
            "eta1": "Pa*s",
            "Gp1": "Pa",
            "Gp2": "Pa",
            "k_cell": "Pa",
            "alpha_cell": "dimensionless",
            "epsilon": "dimensionless",
            "w_Gp1": "Pa",
            "w_Gp2": "Pa",
            "w_k_cell": "Pa",
            "w_alpha_cell": "dimensionless",
        }
        for key, value in units.items():
            if value is not None and key in data:
                data[key] = pint_pandas.PintArray(data[key].values, getattr(ureg, value))

    if cache is True:
        data.to_csv(cache_file, index_label=False)

    return data, config
