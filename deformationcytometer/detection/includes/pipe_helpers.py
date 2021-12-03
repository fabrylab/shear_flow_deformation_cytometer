import multiprocessing


def log(name, name2, onoff, index=0):
    import os, time
    with open(f"log_{name}_{os.getpid()}.txt", "a") as fp:
        fp.write(f"{time.time()} \"{name2}\" {onoff} {index}\n")


def clear_logs():
    import glob, os
    files = glob.glob("log_*.txt")
    for file in files:
        os.remove(file)


class FileFinished: pass


def to_filelist(paths, reevaluate=False):
    import glob
    from pathlib import Path
    from deformationcytometer.includes.includes import getConfig

    if not isinstance(paths, list):
        paths = [paths]
    files = []
    for path in paths:
        if path.endswith(".tif") or path.endswith(".cdb"):
            if "*" in path:
                files.extend(glob.glob(path, recursive=True))
            else:
                files.append(path)
        else:
            files.extend(glob.glob(path + "/**/*.tif", recursive=True))
    files2 = []
    for filename in files:
        if reevaluate or not Path(str(filename)[:-4] + "_evaluated_config_new.txt").exists():
            # check if the config file exists
            try:
                config = getConfig(filename)
            except (OSError, ValueError) as err:
                print(err)
                continue
            files2.append(filename)
        else:
            print(filename, "already evaluated")
    return files2


class get_items:
    def __init__(self, reevaluate):
        self.reevaluate = reevaluate

    def __call__(self, d):
        from deformationcytometer.detection import pipey
        d = to_filelist(d, self.reevaluate)
        for x in d:
            yield x
        yield pipey.STOP


import numpy as np
class JoinedDataStorage:
    def __init__(self, count):
        self.image_data = DataBlock(count, dtype=np.float32)
        self.mask_data = DataBlock(count, dtype=np.uint8)

    def get_stored(self, info):
        if info["dtype"] == np.float32:
            return self.image_data.get_stored(info)
        return self.mask_data.get_stored(info)

    def allocate(self, shape, dtype=np.float32):
        import time
        while True:
            try:
                if dtype == np.float32:
                    return self.image_data.allocate(shape)
                return self.mask_data.allocate(shape)
            except ValueError:
                time.sleep(1)

    def deallocate(self, info):
        if info["dtype"] == np.float32:
            return self.image_data.deallocate(info)
        return self.mask_data.deallocate(info)


class DataBlock:
    def __init__(self, count, dtype):
        self.max_size = 540 * 720 * count
        self.default_dtype = dtype
        self.data_storage = multiprocessing.Array({np.float32: "f", np.uint8: "B"}[dtype], 540 * 720 * count)
        self.data_storage_allocated = multiprocessing.Array("L", 100*2)

    def get_stored(self, info):
        return np.frombuffer(self.data_storage.get_obj(), count=int(np.prod(info["shape"])), dtype=info["dtype"], offset=info["offset"]).reshape(info["shape"])

    def allocate(self, shape, dtype=None):
        dtype = dtype or self.default_dtype

        def getOverlap(a, b):
            return max(0, min(a[1], b[1]) - max(a[0], b[0]))

        allocated_blocks = np.frombuffer(self.data_storage_allocated.get_obj(), dtype=np.uint32).reshape(-1, 2)
        start_frames = sorted([b[0] for b in allocated_blocks if b[1]])
        end_frames = sorted([b[1] for b in allocated_blocks if b[1]])

        count = np.prod(shape)
        bytes = np.dtype(dtype).itemsize

        start = 0
        for s, e in zip(start_frames, end_frames):
            if not getOverlap([start, start+count*bytes], [s, e]):
                break
            start = e
        if start+count*bytes > self.max_size:
            raise ValueError()
        for i, (s, e) in enumerate(allocated_blocks):
            if e == 0:
                allocated_blocks[i] = [start, start+count*bytes]
                break

        return dict(shape=shape, offset=start, dtype=dtype, name="data_storage", allocation_index=i)

    def deallocate(self, info):
        allocated_blocks = np.frombuffer(self.data_storage_allocated.get_obj(), dtype=np.uint32).reshape(-1, 2)
        allocated_blocks[info["allocation_index"], :] = 0