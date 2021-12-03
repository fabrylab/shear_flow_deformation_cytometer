# -*- coding: utf-8 -*-

import numpy as np
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import os
import imageio
import json
from pathlib import Path
import tqdm
import pandas as pd
from scipy.ndimage import shift
import skimage.registration

from deformationcytometer.includes.includes import getInputFile, read_args_tank_treading
from deformationcytometer.evaluation.helper_functions import getConfig, getData, getVelocity, correctCenter
from deformationcytometer.evaluation.helper_functions import fit_func_velocity
import scipy as sp
import scipy.optimize
import tifffile
from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader

file = read_args_tank_treading()
video = getInputFile(settings_name="extract_cell_snippets.py", video=file)
print(video)

config = getConfig(video)
config["channel_width_m"] = 0.00019001261833616293

data = getData(video)
getVelocity(data, config)

correctCenter(data, config)

data = data[(data.solidity > 0.96) & (data.irregularity < 1.06)]
data.reset_index(drop=True, inplace=True)

ids = pd.unique(data["cell_id"])

image_reader = CachedImageReader(video)

results = []
for id in tqdm.tqdm(ids):
    d = data[data.cell_id == id]

    crops, shifts, valid = getCroppedImages(image_reader, d)

    if len(crops) <= 1:
        continue

    crops = crops[valid]
    shifts = shifts[valid]

    time = (d.timestamp - d.iloc[0].timestamp) * 1e-3

    speed, r2 = doTracking(crops, data0=d, times=np.array(time), pixel_size=config["pixel_size"])
    results.append([id, speed, r2])

data = pd.DataFrame(results, columns=["id", "tt", "tt_r2"])
data.to_csv(video[:-4]+"_tt.csv")
