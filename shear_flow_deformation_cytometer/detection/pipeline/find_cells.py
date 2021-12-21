from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *

class ProcessFindCells:
    def __init__(self, irregularity_threshold, solidity_threshold, data_storage, r_min, write_clickpoints_markers, hollow_masks=True):
        self.irregularity_threshold = irregularity_threshold
        self.solidity_threshold = solidity_threshold
        self.data_storage = data_storage
        self.r_min = r_min
        self.write_clickpoints_markers = write_clickpoints_markers
        self.hollow_masks = hollow_masks

    def __call__(self, block):
        import time
        predict_start_first = time.time()
        import pandas as pd
        from pathlib import Path
        from shear_flow_deformation_cytometer.detection.includes.regionprops import mask_to_cells_edge, mask_to_cells_edge2
        from shear_flow_deformation_cytometer.evaluation.helper_functions import filterCells
        import numpy as np

        output_path = Path(block["filename"][:-4] + "_result_new.csv")

        if block["type"] != "image":
            if block["type"] == "start":
                # add ellipse marker type
                if self.write_clickpoints_markers:
                    import clickpoints
                    with clickpoints.DataFile(block["filename"][:-4] + ".cdb") as cdb:
                        cdb.setMarkerType("cell", "#FF0000", mode=cdb.TYPE_Ellipse)
                # delete an existing outputfile
                if output_path.exists():
                    output_path.unlink()
            return block

        data_storage_mask_numpy = self.data_storage.get_stored(block["mask_info"])

        log("3find_cells", "detect", 1, block["index"])
        new_cells = []
        row_indices = [0]
        for mask, timestamp, index in zip(data_storage_mask_numpy, block["timestamps"], range(block["index"], block["index"] + data_storage_mask_numpy.shape[0])):
            cells = mask_to_cells_edge2(mask, None, block["config"], self.r_min, frame_data={"frame": index, "timestamp": timestamp}, hollow_masks=self.hollow_masks)
            row_indices.append(row_indices[-1]+len(cells))
            new_cells.extend(cells)

        new_cells = pd.DataFrame(new_cells,
                                 columns=["frame", "timestamp", "x", "y", "radial_position",
                                          "long_axis", "short_axis",
                                          "long_axis_px", "short_axis_px",
                                          "angle", "irregularity", "solidity", "velocity", "cell_id"])

        if not output_path.exists():
            with output_path.open("w") as fp:
                new_cells.to_csv(fp, index=False, header=True)
        else:
            with output_path.open("a") as fp:
                new_cells.to_csv(fp, index=False, header=False)

        # filter cells according to solidity and irregularity
        new_cells = filterCells(new_cells, solidity_threshold=self.solidity_threshold,
                                irregularity_threshold=self.irregularity_threshold)

        if self.write_clickpoints_markers:
            import clickpoints
            with clickpoints.DataFile(block["filename"][:-4] + ".cdb") as cdb:
                for i, d in new_cells.iterrows():
                    cdb.setEllipse(frame=int(d.frames), x=d.x, y=d.y,
                                   width=d.long_axis / block["config"]["pixel_size"],
                                   height=d.short_axis / block["config"]["pixel_size"],
                                   angle=d.angle, type="cell")

        block["config"]["solidity"] = self.solidity_threshold
        block["config"]["irregularity"] = self.irregularity_threshold

        #new_cells.set_index("frame", inplace=True)

        block["cells"] = new_cells
        block["row_indices"] = row_indices
        #del data["mask"]

        log("3find_cells", "detect", 0, block["index"])
        return block
