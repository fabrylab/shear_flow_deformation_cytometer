from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *


class ProcessCopyImages:
    def __init__(self, data_storage: "JoinedDataStorage"):
        self.data_storage = data_storage

    def __call__(self, filename):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        from shear_flow_deformation_cytometer.detection import pipey
        from shear_flow_deformation_cytometer.detection.includes.regionprops import preprocess, getTimestamp
        from shear_flow_deformation_cytometer.includes.includes import getConfig
        from pathlib import Path
        import clickpoints
        import numpy as np

        import shutil

        Path("tmp").mkdir(exist_ok=True)
        target_file = Path("tmp") / Path(filename).name
        shutil.copy(filename, target_file)

        return filename, target_file

