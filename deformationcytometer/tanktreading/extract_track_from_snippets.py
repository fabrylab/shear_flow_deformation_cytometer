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
pylustrator.start()

from scripts.helper_functions import getStressStrain, getConfig, getInputFile

import scipy.special

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

video = getInputFile()

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

def doTracking(images, mask):
    lk_params = dict(winSize=(8, 8), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 0.03),
                     flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                     )

    for index, im in enumerate(images):
        # if it is the first image find features
        if index == 0:
            # find the features
            try:
                p0 = cv2.goodFeaturesToTrack(im, 20, 0.1, 5)[:, 0, :]
            except TypeError:
                print("error", cv2.goodFeaturesToTrack(im, 20, 0.1, 5))
                return None
            x, y = p0.astype(np.int).T
            inside = mask[y, x]
            p0 = p0[inside]

            # initialize the arrays
            tracks = np.ones((p0.shape[0], len(images), 2)) * np.nan
            active = np.ones(p0.shape[0], dtype=np.bool)
            # define the "results" of this tracking step
            st = np.ones(p0.shape[0], dtype=np.bool)
            p1 = p0
        else:
            # track the current points
            p1, st, err = cv2.calcOpticalFlowPyrLK(image_last, im, p0, None, **lk_params)
            st = st[:, 0].astype(np.bool)
            err = err[:, 0]

            # filter valid tracks (i.e. not out of bounds of the image)
            valid = (p1[:, 0] > 0) * (p1[:, 0] < im.shape[1]) * (p1[:, 1] > 0) * (p1[:, 1] < im.shape[0])
            x, y = p1.astype(np.int).T
            inside = mask[y, x]
            st = valid & st & inside #& (err < 0.2)
            active[active] = active[active] & st

        # add the found and active points to the track array
        tracks[active, index, :] = p1[st]
        # store the current points for the next iteration
        p0 = p1[st]
        image_last = im
        # if no points are left, stop
        if len(p0) == 0:
            break

    return tracks

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


def makePlot(cell_id, data0, images, time, tracks, distance_to_center, angle, speeds, slopes):

    plt.subplot(231)
    im, c, skip = joinImages(images)
    plt.imshow(im, cmap="gray")
    #for index, track in enumerate(tracks.transpose(1, 0, 2)[::skip]):
    #    plt.plot(track[:, 0]+images[0].shape[1]*index, track[:, 1], "+", ms=1)
    for index, track in enumerate(tracks):
        points = track[::skip]
        index = np.arange(points.shape[0])
        plt.plot(track[:, 0]+images[0].shape[1]*index, track[:, 1], "+", ms=1)

    plt.subplot(234)
    la, sa, a = data0.long_axis.mean() / pixel_size, data0.short_axis.mean() / pixel_size, data0.angle.mean()
    for i in range(tracks.shape[0]):
        plt.plot(tracks[i, :, 0], tracks[i, :, 1], "o-", ms=1)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=center, width=la, height=sa, angle=a, edgecolor='r', fc='None', lw=0.5, zorder=2)
    plt.gca().add_patch(ellipse)
    plt.gca().axis("equal")

    plt.subplot(232)
    plt.subplot(233)
    for d, a, m, t in zip(distance_to_center, angle, speeds, slopes):

        plt.subplot(232)
        plt.plot(time, (m * time + t) / np.pi * 180, "k-")
        plt.plot(time, a / np.pi * 180, "o", ms=2)
        plt.xlabel("time (ms)")
        plt.ylabel("angle (deg)")

        plt.subplot(233)
        plt.plot(d[0] * pixel_size, -m / (np.pi * 2), "o", ms=2)
        plt.xlabel("distance from center (µm)")
        plt.ylabel("rotation frequency (1/s)")

    plt.title(f"$\\gamma=${data0.grad.mean() / (2 * np.pi):.2} $\\omega=${-np.nanmedian(speeds)/(np.pi * 2):.2}")
    plt.axhline(-np.nanmedian(speeds) / (2 * np.pi), color="k", ls="--")

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(16.250000/2.54, 7.490000/2.54, forward=True)
    plt.figure(1).axes[0].set_position([0.051398, 0.775448, 0.878050, 0.141110])
    plt.figure(1).axes[0].set_xlim(-0.5, 1079.5)
    plt.figure(1).axes[0].set_xticklabels([""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[0].set_xticks([np.nan])
    plt.figure(1).axes[0].set_ylim(79.5, -0.5)
    plt.figure(1).axes[0].set_yticklabels([""], fontsize=10)
    plt.figure(1).axes[0].set_yticks([np.nan])
    plt.figure(1).axes[1].set_position([0.049804, 0.185454, 0.226657, 0.471913])
    plt.figure(1).axes[1].spines['right'].set_visible(False)
    plt.figure(1).axes[1].spines['top'].set_visible(False)
    plt.figure(1).axes[2].set_position([0.368933, 0.185454, 0.210172, 0.471913])
    plt.figure(1).axes[2].spines['right'].set_visible(False)
    plt.figure(1).axes[2].spines['top'].set_visible(False)
    plt.figure(1).axes[3].set_position([0.668273, 0.185454, 0.226657, 0.419706])
    plt.figure(1).axes[3].spines['right'].set_visible(False)
    plt.figure(1).axes[3].spines['top'].set_visible(False)
    #% end: automatic generated code from pylustrator
    plt.savefig(target_folder / f"fit_{cell_id}.png", dpi=300)
    print(target_folder / f"fit_{cell_id}.png")
    #plt.show()
    plt.clf()


import tqdm
with open(target_folder / "speeds.txt", "w") as fp:
    for cell_id in tqdm.tqdm(cell_ids): # [15195]:#
    #for cell_id in [15195]:#
        data0 = data[data.id == cell_id]

        print(cell_id, len(data0))

        if len(data0) < 5:
            print("to short")
            continue

        image_stack = getImageStack(cell_id)

        #print(image_stack.shape)
        #plt.imshow(image_stack[0])
        #plt.show()

        tracks = doTracking(image_stack, mask=getMask(data0.iloc[0], image_stack[0]))

        if tracks is None:
            print("No features to track")
            continue

        center = np.array([60, 40])

        distance_to_center = np.linalg.norm(tracks - center, axis=2)
        angle = np.arctan2(tracks[:, :, 1] - center[1], tracks[:, :, 0] - center[0])
        angle = np.unwrap(angle, axis=1)

        angle, distance_to_center = getArcLength(tracks, data0.long_axis.mean() / pixel_size, data0.short_axis.mean() / pixel_size,
                                 data0.angle.mean(), center)

        time = (data0.timestamp - data0.iloc[0].timestamp) * 1e-3

        if time.shape[0] != angle.shape[1]:
            print("missmatch", len(time), len(angle), tracks.shape, angle.shape, time.shape)
            continue

        fits = np.array([getLine(time, a) for a in angle])
        speeds = fits[:, 0]
        slopes = fits[:, 1]

        try:
            makePlot(cell_id, data0, image_stack, time, tracks, distance_to_center, angle, speeds, slopes)
        except ValueError:
            pass

        fp.write(f"{cell_id} {data0.grad.mean()/(2*np.pi)} {-np.median(speeds)/(2*np.pi)} {data0.rp.mean()}\n")
