from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *


class ProcessTankTreading:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def __call__(self, block):
        from shear_flow_deformation_cytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader
        import numpy as np
        import pandas as pd
        pd.options.mode.chained_assignment = 'raise'
        if block["type"] == "start" or block["type"] == "end":
            return block

        log("5tt", "prepare", 1, block["index"])

        data_storage_numpy = self.data_storage.get_stored(block["data_info"])

        class CachedImageReader:
            def get_data(self, index):
                return data_storage_numpy[int(index) - block["index"]]

        image_reader = CachedImageReader()
        cells = block["cells"]
        cells["tt"] = np.nan
        cells["tt_r2"] = np.nan
        cells["tt_omega"] = np.nan
        row_indices = block["row_indices"]

        for i, index in enumerate(range(block["index"], block["end_index"] - 1)):
            cells1 = cells.iloc[row_indices[i+0]:row_indices[i+1]]
            cells2 = cells.iloc[row_indices[i+1]:row_indices[i+2]]

            for i, (index, d2) in enumerate(cells2.iterrows()):
                if np.isnan(d2.measured_velocity):
                    continue
                try:
                    d1 = cells1[cells1.cell_id == d2.cell_id].iloc[0]
                except IndexError:
                    continue
                d = pd.DataFrame([d2, d1])

                crops, shifts, valid = getCroppedImages(image_reader, d)

                if len(crops) <= 1:
                    continue

                crops = crops[valid]
                shifts = shifts[valid]

                time = (d.timestamp - d.iloc[0].timestamp) * 1e-3

                speed, r2 = doTracking(crops, data0=d, times=np.array(time), pixel_size=block["config"]["pixel_size"])

                cells2.iat[i, cells2.columns.get_loc("tt")] = speed * 2 * np.pi
                cells2.iat[i, cells2.columns.get_loc("tt_r2")] = r2
                if r2 > 0.2:
                    cells2.iat[i, cells2.columns.get_loc("tt_omega")] = speed * 2 * np.pi

        log("5tt", "prepare", 0, block["index"])
        return block
