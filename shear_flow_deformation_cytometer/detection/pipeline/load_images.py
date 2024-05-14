import tifffile

from shear_flow_deformation_cytometer.detection.includes.pipe_helpers import *


rounding_enabled = False


class ProcessLoadImages:
    def __init__(self, data_storage: "JoinedDataStorage", batch_size, write_clickpoints_file=False):
        self.data_storage = data_storage
        self.write_clickpoints_file = write_clickpoints_file
        self.batch_size = batch_size

    def __call__(self, filename, copy_of_file=None):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        import sys
        from shear_flow_deformation_cytometer.detection import pipey
        from shear_flow_deformation_cytometer.detection.includes.regionprops import preprocess, getTimestamp
        from shear_flow_deformation_cytometer.includes.includes import getConfig
        import numpy as np

        class reader2:
            def __init__(self, reader):
                self.reader = reader

            def __len__(self):
                return len(self.reader)//2

            def __iter__(self):
                for i, im in enumerate(self.reader):
                    if i % 2 == 0:
                        yield im

            def get_meta_data(self, index):
                return self.reader.get_meta_data(index*2)

            def close(self):
                self.reader.close()

        log("1load_images", "prepare", 1)

        # open the image reader
        #reader = reader2(imageio.get_reader(copy_of_file or filename))
        try:
            reader_tiff = tifffile.TiffReader(copy_of_file or filename).pages
            reader = imageio.get_reader(copy_of_file or filename)
        except Exception as err:
            print(err, file=sys.stderr)
            return
        # get the config file
        config = getConfig(filename)
        # framerate
        dt = 1 / config["frame_rate"] * 1e3
        # get the total image count
        image_count = len(reader_tiff)

        if self.write_clickpoints_file:
            import clickpoints
            cdb = clickpoints.DataFile(filename[:-4]+".cdb", "w")
            cdb.setMaskType("prediction", color="#FF00FF", index=1)
        yield dict(filename=filename, index=-1, type="start", image_count=image_count)
        log("1load_images", "prepare", 0)

        data_storage_numpy = None

        log("1load_images", "read", 1)
        images = []
        timestamps = []
        start_batch_index = 0
        timestamp_start = None
        timestamp_previous = None
        log("1load_images", "read", 1, 0)

        # iterate over all images in the file
        for image_index, im in enumerate(reader_tiff):
            im = im.asarray()
            # ensure image has only one channel
            if len(im.shape) == 3:
                im = im[:, :, 0]
            # get the timestamp from the file
            timestamp = float(getTimestamp(reader, image_index, dt))
            if timestamp_start is None:
                timestamp_start = timestamp
            timestamp -= timestamp_start

            if rounding_enabled:
                timestamp = np.round(timestamp / dt) * dt

                if timestamp_previous:
                    if timestamp - timestamp_previous < dt * 0.9:
                        timestamp += dt
                timestamp_previous = timestamp

            if self.write_clickpoints_file:
                cdb.setImage(filename, frame=image_index)#, timestamp=timestamp)

            images.append(im)
            timestamps.append(timestamp)

            if image_index == image_count-1 or len(images) == self.batch_size:

                info = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.float32)
                info_mask = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.uint8)
                data_storage_numpy = self.data_storage.get_stored(info)
                for i, im in enumerate(images):
                    data_storage_numpy[i] = im

                log("1load_images", "read", 0, start_batch_index)
                yield dict(filename=filename, index=start_batch_index, end_index=start_batch_index+len(images), type="image", timestamps=timestamps,
                           data_info=info, mask_info=info_mask,
                           config=config, image_count=image_count)
                if image_index != image_count-1:
                    log("1load_images", "read", 1, start_batch_index+len(images))
                images = []
                timestamps = []
                start_batch_index = image_index+1

            if image_index == image_count - 1:
                break

        reader.close()
        if copy_of_file is not None:
            copy_of_file.unlink()
        yield dict(filename=filename, index=image_count, type="end")




class ProcessLoadImagesClickpoints:
    def __init__(self, data_storage: "JoinedDataStorage", batch_size):
        self.data_storage = data_storage
        self.batch_size = batch_size

    def __call__(self, filename, copy_of_file=None):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        import sys
        from shear_flow_deformation_cytometer.detection import pipey
        from shear_flow_deformation_cytometer.detection.includes.regionprops import preprocess, getTimestamp
        from shear_flow_deformation_cytometer.includes.includes import getConfig
        import clickpoints
        import numpy as np

        log("1load_images", "prepare", 1)



        # open the image reader
        #reader = reader2(imageio.get_reader(copy_of_file or filename))
        try:
            cdb = clickpoints.DataFile(copy_of_file or filename)
        except Exception as err:
            print(err, file=sys.stderr)
            return

        # get the total image count
        image_count = cdb.getImageCount()

        yield dict(filename=filename, index=-1, type="start", image_count=image_count)
        log("1load_images", "prepare", 0)

        data_storage_numpy = None

        log("1load_images", "read", 1)
        images = []
        timestamps = []
        start_batch_index = 0
        timestamp_start = None
        log("1load_images", "read", 1, 0)

        # iterate over all images in the file
        for image_index, im in enumerate(cdb.getImages()):
            im = im.data_thp1
            # ensure image has only one channel
            if len(im.shape) == 3:
                im = im[:, :, 0]
            # get the timestamp from the file
            timestamp = 0#float(getTimestamp(reader, image_index))
            if timestamp_start is None:
                timestamp_start = timestamp
            timestamp -= timestamp_start

            images.append(im)
            timestamps.append(timestamp)

            if image_index == image_count-1 or len(images) == batch_size:

                info = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.float32)
                info_mask = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.uint8)
                data_storage_numpy = self.data_storage.get_stored(info)
                for i, im in enumerate(images):
                    data_storage_numpy[i] = im

                log("1load_images", "read", 0, start_batch_index)
                yield dict(filename=filename, index=start_batch_index, end_index=start_batch_index+len(images), type="image", timestamps=timestamps,
                           data_info=info, mask_info=info_mask,
                           config=config, image_count=image_count)
                if image_index != image_count-1:
                    log("1load_images", "read", 1, start_batch_index+len(images))
                images = []
                timestamps = []
                start_batch_index = image_index+1

            if image_index == image_count - 1:
                break

        reader.close()
        if copy_of_file is not None:
            copy_of_file.unlink()

        yield dict(filename=filename, index=image_count, type="end")



class ProcessLoadImagesClickpointsAndMasks:
    def __init__(self, data_storage: "JoinedDataStorage", batch_size, type="polygon"):
        self.data_storage = data_storage
        self.batch_size = batch_size
        self.type = type

    def __call__(self, filename, copy_of_file=None):
        """
        Loads an .tif file stack and yields all the images.
        """
        import imageio
        import sys
        from shear_flow_deformation_cytometer.detection import pipey
        from shear_flow_deformation_cytometer.detection.includes.regionprops import preprocess, getTimestamp
        from shear_flow_deformation_cytometer.includes.includes import getConfig
        import clickpoints
        import numpy as np

        log("1load_images", "prepare", 1)



        # open the image reader
        #reader = reader2(imageio.get_reader(copy_of_file or filename))
        try:
            cdb = clickpoints.DataFile(copy_of_file or filename)
        except Exception as err:
            print(err, file=sys.stderr)
            return

        # get the config file
        config = getConfig(filename)

        # get the total image count
        image_count = cdb.getImageCount()

        yield dict(filename=filename, index=-1, type="start", image_count=image_count)
        log("1load_images", "prepare", 0)

        data_storage_numpy = None

        log("1load_images", "read", 1)
        images = []
        masks = []
        timestamps = []
        start_batch_index = 0
        timestamp_start = None
        log("1load_images", "read", 1, 0)

        # iterate over all images in the file
        for image_index, img in enumerate(cdb.getImages()):
            im = img.data_thp1
            # ensure image has only one channel
            if len(im.shape) == 3:
                im = im[:, :, 0]

            mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            if self.type == "polygon":
                for obj in img.polygons:
                    mask[obj.getPixels()] = 1
                from skimage.morphology import binary_erosion
                mask = mask - binary_erosion(binary_erosion(mask))
            elif self.type == "mask":
                from skimage.morphology import binary_erosion
                if img.mask is not None:
                    mask = img.mask.data_thp1
                mask = mask - binary_erosion(binary_erosion(mask))
            masks.append(mask)

            # get the timestamp from the file
            timestamp = 0#float(getTimestamp(reader, image_index))
            if timestamp_start is None:
                timestamp_start = timestamp
            timestamp -= timestamp_start

            images.append(im)
            timestamps.append(timestamp)

            if image_index == image_count-1 or len(images) == self.batch_size:
                info = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.float32)
                info_mask = self.data_storage.allocate([len(images)]+list(images[0].shape), dtype=np.uint8)
                data_storage_numpy = self.data_storage.get_stored(info)
                data_storage_numpy_mask = self.data_storage.get_stored(info_mask)
                for i, im in enumerate(images):
                    data_storage_numpy[i] = im
                    data_storage_numpy_mask[i] = masks[i]

                log("1load_images", "read", 0, start_batch_index)
                yield dict(filename=filename, index=start_batch_index, end_index=start_batch_index+len(images), type="image", timestamps=timestamps,
                           data_info=info, mask_info=info_mask,
                           config=config, image_count=image_count)
                if image_index != image_count-1:
                    log("1load_images", "read", 1, start_batch_index+len(images))
                images = []
                masks = []
                timestamps = []
                start_batch_index = image_index+1

            if image_index == image_count - 1:
                break

        cdb.db.close()
        if copy_of_file is not None:
            copy_of_file.unlink()

        yield dict(filename=filename, index=image_count, type="end")