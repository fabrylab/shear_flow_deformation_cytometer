from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *


class ProcessPairData:
    def __call__(self, block):
        from shear_flow_deformation_cytometer.detection.includes.regionprops import matchVelocities

        if block["type"] == "end" or block["type"] == "start":
            return block

        log("4vel", "prepare", 1, block["index"])

        cells = block["cells"]
        row_indices = block["row_indices"]
        next_id = 1 + block["index"] * 1000

        _, next_id = matchVelocities(cells.iloc[0:0],
                                     cells.iloc[row_indices[0]:row_indices[1]],
                                     next_id, block["config"])

        for i, index in enumerate(range(block["index"], block["end_index"] - 1)):
            _, next_id = matchVelocities(cells.iloc[row_indices[i]:row_indices[i+1]],
                                         cells.iloc[row_indices[i+1]:row_indices[i+2]],
                                         next_id, block["config"])

        log("4vel", "prepare", 0, block["index"])
        return block
