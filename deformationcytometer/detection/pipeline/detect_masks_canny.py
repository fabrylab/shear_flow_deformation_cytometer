from deformationcytometer.detection.includes.pipe_helpers import *



class ProcessDetectMasksBatchCanny:
    """
    Takes images and groups them into batches to feed them into the neural network to create masks.
    """
    unet = None
    batch = None

    def __init__(self, batch_size, network_weights, data_storage, data_storage_mask, write_clickpoints_masks):
        # store the batch size
        self.batch_size = batch_size
        self.network_weights = network_weights
        self.data_storage = data_storage
        self.data_storage_mask = data_storage_mask
        self.write_clickpoints_masks = write_clickpoints_masks

    def __call__(self, block):
        import time
        predict_start_first = time.time()
        from deformationcytometer.detection.includes.UNETmodel import UNet
        import numpy as np
        import cv2
        from skimage.filters import gaussian
        from skimage.morphology import area_opening
        from skimage import feature
        from scipy.ndimage import generate_binary_structure, binary_fill_holes
        from skimage import morphology
        from deformationcytometer.detection.includes.regionprops import preprocess, getTimestamp

        if block["type"] == "start" or block["type"] == "end":
            yield block
            return

        log("2detect", "prepare", 1, block["index"])


        data_storage_numpy = self.data_storage.get_stored(block["data_info"])
        data_storage_mask_numpy = self.data_storage.get_stored(block["mask_info"])

        for i, im in enumerate(data_storage_numpy):
            f = im
            ff = f / f.max() * 255
            ffl = ff
            ffl = np.uint8(ffl / ffl.max() * 255)
            fban = gaussian(f, sigma=1) - gaussian(f, sigma=6)
            fban = fban - fban.min()
            fban = np.uint8(fban / fban.max() * 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            gradient = cv2.morphologyEx(fban, cv2.MORPH_GRADIENT, kernel)
            fban = np.uint8(gradient / gradient.max() * 255)
            edges = feature.canny(fban, sigma=2, low_threshold=0.99, high_threshold=0.99, use_quantiles=True)
            struct = generate_binary_structure(2, 1)
            ffil = binary_fill_holes(edges, structure=struct).astype(int)
            ffil = np.uint8(ffil * 255)
            mask = area_opening(ffil, area_threshold=600, connectivity=1)
            import matplotlib.pyplot as plt
            data_storage_mask_numpy[i] = mask

        import clickpoints
        if self.write_clickpoints_masks:
            with clickpoints.DataFile(block["filename"][:-4] + ".cdb") as cdb:
                # iterate over all images and return them
                for mask, index in zip(data_storage_mask_numpy, range(block["index"], block["end_index"])):
                    cdb.setMask(frame=index, data=mask.astype(np.uint8))

        block["config"].update({"network": self.network_weights})

        log("2detect", "prepare", 0, block["index"])
        yield block