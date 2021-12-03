# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:28:22 2020
@author: Ben Fabry
"""
# this program reads the frames of an avi video file, averages all images,
# and stores the normalized image as a floating point numpy array 
# in the same directory as the extracted images, under the name "flatfield.npy"
#
# The program then loops again through all images of the video file,
# identifies cells, extracts the cell shape, fits an ellipse to the cell shape,
# and stores the information on the cell's centroid position, long and short axis,
# angle (orientation) of the long axis, and bounding box widht and height
# in a text file (result_file.txt) in the same directory as the video file.

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
settings_name = "extrac_cell_snippets.py"
def angles_in_ellipse(
        a,
        b):
    assert(a < b)
    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    print("circumference", sp.special.ellipeinc(2.0 * np.pi, e), e)
    num = 20
    num = np.round(sp.special.ellipeinc(2.0 * np.pi, e))
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles

r_min = 5   #cells smaller than r_min (in um) will not be analyzed
file = read_args_tank_treading()
video = getInputFile(settings_name=settings_name, video=file)

#%%
config = getConfig(video)
config["channel_width_m"] = 0.00019001261833616293

data = getData(video)
getVelocity(data, config)

# take the mean of all values of each cell
#data = data.groupby(['cell_id']).mean()

correctCenter(data, config)
#exit()


data = data[(data.solidity > 0.96) & (data.irregularity < 1.06)]
#data = data[(data.solidity > 0.98) & (data.irregularity < 1.02)]
data.reset_index(drop=True, inplace=True)

ids = pd.unique(data["cell_id"])

image_reader = imageio.get_reader(video)
import matplotlib.pyplot as plt
i = 0
target_folder = Path(video[:-4])#Path(video).parent / "_".join(str(Path(video).name).split("_")[:6])
print("target_folder", target_folder)
target_folder.mkdir(exist_ok=True)

def getCroppedImages(image_reader, cells, w=60, h=40, o=5, o2=15):
    crops = []
    shifts = []
    valid = []
    im0 = None
    shift0 = None
    # iterate over all cells
    for index, cell in enumerate(cells.itertuples()):
        # get the image
        im = image_reader.get_data(cell.frames)
        # get the cell position
        y = int(round(cell.y))
        x = int(round(cell.x))
        # crop the image
        crop = im[y - h - o:y + h + o, x - w - o:x + w + o]
        # if the image does not have the full size, skip it (e.g. it is at the border)
        if crop.shape[0] != h * 2 + o * 2 or crop.shape[1] != w * 2 + o * 2:
            crops.append(np.ones([h*2, w*2])*np.nan)
            shifts.append([0, 0])
            valid.append(False)
            continue

        # if it is the first image, we cannot do any image registration
        if im0 is None:
            # we just move it by the float point part of the cell position
            shift_px = [cell.y - y, cell.x - x]
            #print(shift_px)
            shifts.append([0, 0])
            shift0 = np.array(shift_px)
        else:
            # try to register the image
            try:
                shift_px, error, diffphase = skimage.registration.phase_cross_correlation(im0[o2:-o2, o2:-o2], crop[o2:-o2, o2:-o2], upsample_factor=100)
            except ValueError:
                # if it is not successfully, skip the image
                crops.append(np.ones(h*2, w*2) * np.nan)
                shifts.append([0, 0])
                valid.append(False)
                continue
            #print(shift_px, type(shift_px))
            shifts.append(-np.array([cell.y - y, cell.x - x])+shift_px)
        # shift the image by the offset
        crop = shift(crop, [shift_px[0], shift_px[1]])
        # store the image if we don't have an image yet
        if im0 is None:
            im0 = crop
        # crop the image to remove unfilled borders
        crop = crop[o:-o, o:-o]
        # filter the image
        crop = scipy.ndimage.gaussian_laplace(crop.astype("float"), sigma=1)
        # append it to the list
        crops.append(crop)
        valid.append(True)
    # normalize the image stack
    crops = np.array(crops)
    crops -= np.nanmin(crops)
    crops /= np.nanmax(crops)
    crops *= 255
    crops = crops.astype(np.uint8)
    return crops, shifts, valid


def getVelGrad(r):
    p0, p1, p2 = config["vel_fit"]
    r = r
    p0 = p0 * 1e3
    r0 = config["channel_width_m"]*0.5*1e6#100e-6
    return - (p1 * p0 * (np.abs(r) / r0) ** p1) / r


with open(target_folder / "output.csv", "w") as fp:
    for id in ids:
        d = data[data.cell_id == id]

        crops, shifts, valid = getCroppedImages(image_reader, d)

        if len(crops) <= 1:
            continue

        tiffWriter = tifffile.TiffWriter(target_folder / f"{id:05d}.tif", bigtiff=True)
        for index, cell in enumerate(d.itertuples()):
            if not valid[index]:
                continue

            tiffWriter.save(crops[index], compress=0, metadata={}, contiguous=False)
            #imageio.imwrite(target_folder / f"{id:05d}_{index:05d}.tif", crops[index])

            rp = cell.rp+shifts[index][1]*config['pixel_size']
            grad = getVelGrad(rp)

            if 0:
                plt.subplot(121)
                plt.plot(data.rp*1e-6, data.velocity*1e-3, "o")
                vel = fit_func_velocity(config)
                dx = 1
                x = np.arange(-100, 100, dx)*1e-6
                v = vel(x*1e6)*1e-3
                plt.plot(x, v, "r+")
                plt.axhline(0, color="k", lw=0.8)

                plt.subplot(122)
                grad = np.diff(v)/np.diff(x)# * 1e3
                plt.plot(data.rp*1e-6, data.velocity_gradient, "o")
                plt.plot(data.rp*1e-6, getVelGrad(data.rp), "s")
                plt.plot(x[:-1]+0.5*np.diff(x), grad, "-+")
                plt.show()

            if i == 0:
                fp.write(f"""i,id,index,x,y,rp,long_axis,short_axis,angle,irregularity,solidity,sharpness,timestamp,velocity,grad\n""")
            fp.write(f"""{i},{cell.cell_id},{index},{cell.x+shifts[index][0]},{cell.y+shifts[index][1]},{rp},{cell.long_axis},{cell.short_axis},{cell.angle},{cell.irregularity},{cell.solidity},{cell.sharpness},{cell.timestamp},{cell.velocity},{grad}\n""")

            i += 1
            print(id, cell)
