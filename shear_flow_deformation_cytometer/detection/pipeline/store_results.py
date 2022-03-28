from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *


class ResultCombiner:
    def __init__(self, data_storage, output="_evaluated.csv", output_config="_evaluated.json"):
        self.data_storage = data_storage
        self.output = output
        self.output_config = output_config

    def init(self):
        self.filenames = {}

    def __call__(self, block):
        import sys
        # if the file is finished, store the results
        if block["filename"] not in self.filenames:
            self.filenames[block["filename"]] = dict(cached={}, next_index=-1, cell_count=0, cells=[], progressbar=None,
                                                     config=dict())

        file = self.filenames[block["filename"]]

        if file["progressbar"] is None and block["type"] != "end":
            import tqdm
            file["progressbar"] = tqdm.tqdm(total=block["image_count"], smoothing=0)
            file["progress_count"] = 0

        if block["type"] == "start" or block["type"] == "end":
            return

        log("6combine", "prepare", 1, block["index"])

        file["cells"].append(block["cells"])
        file["cell_count"] += len(block["cells"])
        file["progress_count"] += block["end_index"] - block["index"]
        file["config"] = block["config"]
        file["progressbar"].update(block["end_index"] - block["index"])
        file["progressbar"].set_description(f"cells {file['cell_count']}")
        self.data_storage.deallocate(block["data_info"])
        self.data_storage.deallocate(block["mask_info"])

        if file["progress_count"] == block["image_count"]:
            #try:
            self.save(block)
            #except Exception as err:
            #    print(err, file=sys.stderr)
            file["progressbar"].close()
            del self.filenames[block["filename"]]

        log("6combine", "prepare", 0, block["index"])
    def save(self, block):
        evaluation_version = 9
        from pathlib import Path

        import pandas as pd
        import numpy as np
        import json
        from shear_flow_deformation_cytometer.evaluation.helper_functions import correctCenter, filterCells, getStressStrain, \
            apply_velocity_fit, get_cell_properties, match_cells_from_all_data, add_units
        from shear_flow_deformation_cytometer.evaluation.overview_plot import overview_plot
        from shear_flow_deformation_cytometer.includes.fit_velocity import improved_fit
        from shear_flow_deformation_cytometer.evaluation.fluorescence_intensity import get_fluorescence_intensity

        filename = block["filename"]
        image_width = block["data_info"]["shape"][2]

        file = self.filenames[filename]

        data = pd.concat(file["cells"])
        data.reset_index(drop=True, inplace=True)

        config = file["config"]

        # find the center of the channel with a rough velocity fit
        correctCenter(data, config)

        # reset the indices
        data.reset_index(drop=True, inplace=True)

        getStressStrain(data, config)

        data["area"] = data.long_axis * data.short_axis * np.pi
        data["pressure"] = config["pressure_pa"] * 1e-5
        data.to_csv(r"\\131.188.117.96\biophysDS2\nstroehlein\bloodcells\2021.11.02\1.5 percent alginate\0.2 bar\tmp.csv")
        # apply the shear thinning velocity fit
        data, p = improved_fit(data, config)

        # do matching of velocities again
        try:
            match_cells_from_all_data(data, config, image_width)
        except AttributeError as err:
            print("Err", err)
            pass

        get_cell_properties(data)

        overview_plot(filename, data)

        # optionally add fluorescence data
        data = get_fluorescence_intensity(str(filename), data, optional=True)

        data = add_units(data, config)

        output_file = Path(str(filename)[:-4] + self.output)
        output_config_file = Path(str(filename)[:-4] + self.output_config)
        config["evaluation_version"] = evaluation_version
        data.to_csv(output_file, index=False)

        with output_config_file.open("w") as fp:
            json.dump(config, fp, indent=0)
