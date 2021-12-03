import qtawesome
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage.draw
import imageio
import re
from pathlib import Path
import pylustrator
#pylustrator.start()
settings_name = "extract_track_from_snippets_dense_flow"
np.seterr(divide='ignore', invalid='ignore')

from deformationcytometer.includes.includes import getInputFile
from deformationcytometer.evaluation.helper_functions import getStressStrain, getConfig
import skimage.registration
import scipy.special

def getPerimeter(a, b):
    from scipy.special import ellipe

    # eccentricity squared
    e_sq = 1.0 - b ** 2 / a ** 2
    # circumference formula
    perimeter = 4 * a * ellipe(e_sq)

    return perimeter

def getEllipseArcSegment(angle, a, b):
    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    perimeter = scipy.special.ellipeinc(2.0 * np.pi, e)
    return scipy.special.ellipeinc(angle, e)/perimeter*2*np.pi# - sp.special.ellipeinc(angle-0.1, e)

def getArcLength(points, major_axis, minor_axis, ellipse_angle, center):
    p = points - np.array(center)#[None, None]
    alpha = np.deg2rad(ellipse_angle)
    p = p @ np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    distance_from_center = np.linalg.norm(p, axis=-1)
    angle = np.arctan2(p[..., 0], p[..., 1])
    angle = np.arctan2(np.sin(angle) / (major_axis / 2), np.cos(angle) / (minor_axis / 2))
    angle = np.unwrap(angle)

    r = np.linalg.norm([major_axis / 2 * np.sin(angle), minor_axis / 2 * np.cos(angle)], axis=0)

    length = getEllipseArcSegment(angle, minor_axis/2, major_axis/2)
    return length, distance_from_center/r
file = read_args_tank_treading()
video = getInputFile(settings_name=settings_name, video=file)
print("file:", video)
#video = r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_15_alginate2%_NIH_tanktreading_1\2\2020_09_15_10_30_43.tif"#getInputFile()
id = 8953

id = 5084

id = 4115

config = getConfig(video)

target_folder = Path(video[:-4])

data = pd.read_csv(target_folder / "output.csv")
print(data)
getStressStrain(data, config)

cell_ids = pd.unique(data.id)

pixel_size = 0.34500000000000003

#print("cell_ids", cell_ids)

def getImageStack(cell_id):
    return np.array([im for im in imageio.get_reader(target_folder / f"{cell_id:05}.tif")])

def getMask(d1, im):
    rr, cc = skimage.draw.ellipse(40, 60, d1.short_axis / pixel_size / 2, d1.long_axis / pixel_size / 2, im.shape,
                                  -d1.angle * np.pi / 180)
    mask = np.zeros(im.shape, dtype="bool")
    mask[rr, cc] = 1
    return mask

def getCenterLine(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    def func(x, m):
        return x*m
    import scipy.optimize
    p, popt = scipy.optimize.curve_fit(func, x, y, [1])
    return p[0], 0

def doPlotting():
    mask = getMask(data0.iloc[0], image_stack[0])
    inside = mask[y, x]
    if 0 and i == 0:
        plt.subplot(231)
        plt.imshow(images[i])
        plt.subplot(232)
        plt.imshow(images[i + 1])
        plt.subplot(233)
        plt.imshow(images[i] - images[i + 1])
        plt.subplot(234)
        plt.imshow(flow[1])
        plt.subplot(235)
        plt.imshow(flow[0])
        plt.subplot(236)
        plt.imshow(np.sqrt(flow[1] ** 2 + flow[0] ** 2))

    if 0 and i == 0:
        plt.figure(3)
        for i in np.arange(len(x))[inside]:
            plt.plot([x[i], x[i] + flow[1][i]], [y[i], y[i] + flow[0][i]], "r")
            plt.plot([x[i]], [y[i]], "ro")
            # plt.text(x[i], y[i], f"{projected_speed[i]:.3f}")
        plt.axis("equal")
        # x -= 30
        # y -= 20
        # distance = np.sqrt(x[inside]**2 + y[inside]**2)
        plt.figure(2)
        plt.subplot(121)

    if 0:
        plt.figure(10)
        p, = plt.plot(distance_to_center[indices_middle], projected_speed[indices_middle] / dt / (perimeter_pixels),
                      "o")

    a = np.arange(0, 1, 0.1)
    if 0:
        plt.figure(10)
        plt.plot(a, m * a + t, "-k")
        print("slope", m)
        plt.title(f"{m:.2f} {r2:2f}")
        plt.show()
        plt.cla()

#def plot_flow_field(flow, data):
#    from pyTFM.plotting import show_quiver
#    fig, ax = show_quiver(flow[0], flow[1])
 #   ax.plot(data["x"], data["y"], "o")


def doTracking(images, data0, times):

    data_x = []
    data_y = []

    perimeter_pixels = getPerimeter(data0.long_axis.mean() / pixel_size / 2, data0.short_axis.mean() / pixel_size / 2)

    for i in range(len(images)-1):
        dt = times[i+1]-times[i]
        flow = skimage.registration.optical_flow_tvl1(images[i], images[i+1], attachment=30)

        x, y = np.meshgrid(np.arange(flow[0].shape[1]), np.arange(flow[0].shape[0]), sparse=False, indexing='xy')
        x = x.flatten()
        y = y.flatten()
        flow = flow.reshape(2, -1)

        ox, oy = [60, 40]
        distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        projected_speed = ((x - ox) * flow[0] - (y - oy) * flow[1]) / distance

        angle, distance_to_center = getArcLength(np.array([x, y]).T, data0.long_axis.mean() / pixel_size,
                                                 data0.short_axis.mean() / pixel_size,
                                                 data0.angle.mean(), [ox, oy])

        indices_middle = (distance_to_center < 0.7) & ~np.isnan(projected_speed)

        data_x.extend(distance_to_center[indices_middle])
        data_y.extend(projected_speed[indices_middle]/dt/perimeter_pixels)

    m, t = getCenterLine(data_x, data_y)

    cr = np.corrcoef(data_y, m*np.array(data_x))
    r2 = np.corrcoef(data_y, m*np.array(data_x))[0, 1] ** 2

    return m, r2

def getLine(x, a):
    try:
        m, t = np.polyfit(x, a, deg=1)
    except (np.linalg.LinAlgError, TypeError):
        m, t = np.nan, np.nan
    return m, t

def joinImages(images):
    c, h, w = images.shape
    skip = int(np.ceil(c/10))
    c = images[::skip].shape[0]
    return images[::skip].transpose(1, 0, 2).reshape(h, w*c), c, skip


import tqdm
with open(target_folder / "speeds_new.txt", "w") as fp:
    #for cell_id in [id]:
    for cell_id in tqdm.tqdm(cell_ids): # [15195]:#

    #for cell_id in [15195]:#
        data0 = data[data.id == cell_id]

        print(cell_id, len(data0))

        if len(data0) < 2:
            print("too short")
            continue

        image_stack = getImageStack(cell_id)

        time = (data0.timestamp - data0.iloc[0].timestamp) * 1e-3

        speed, r2 = doTracking(image_stack, data0=data0, times=np.array(time))

        fp.write(f"{cell_id} {data0.grad.mean()/(2*np.pi)} {speed} {data0.rp.mean()} {r2}\n")
