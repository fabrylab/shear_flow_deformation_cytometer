
import numpy as np
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology, binary_fill_holes, binary_closing
from skimage.measure import label, regionprops
import os
import imageio
import json
from pathlib import Path
import time

from skimage.measure import label, regionprops


def preprocess(img):
    #if len(img.shape) == 3:
    #    img = img[:, :, 0]
    return (img - np.mean(img, axis=(0, 1))) / np.std(img, axis=(0, 1)).astype(np.float32)

def batch_iterator(reader, batch_size, preprocess):
    batch_images = None
    batch_image_indices = []
    image_count = len(reader)

    for image_index, im in enumerate(reader):
        # first iteration
        if batch_images is None:
            batch_images = np.zeros([batch_size, im.shape[0], im.shape[1]], dtype=np.float32)

        # add a new image to the batch
        batch_images[len(batch_image_indices)] = preprocess(im)
        batch_image_indices.append(image_index)

        # when the batch is full or when the video is finished
        if len(batch_image_indices) == batch_size or image_index == image_count - 1:
            # yield the images
            yield batch_images[:len(batch_image_indices)], batch_image_indices
            batch_image_indices = []

def getTimestamp(vidcap, image_index):
    if image_index >= len(vidcap):
        image_index = len(vidcap) - 1
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename + ext)


def mask_to_cells_edge(prediction_mask, im, config, r_min, frame_data, edge_dist=15, return_mask=False):
    r_min_pix = r_min / config["pixel_size_m"] / 1e6
    edge_dist_pix = edge_dist / config["pixel_size_m"] / 1e6
    cells = []
    # TDOD: consider first applying binary closing operations to avoid impact of very small gaps in the cell border
    filled = fill_voids.fill(prediction_mask)
    # iterate over all detected regions
    for region in regionprops(label(filled)):  # region props are based on the original image
        # checking if the anything was filled up by extracting the region form the original image
        # if no significant region was filled, we skip this object
        yc, xc = np.split(region.coords, 2, axis=1)
        if np.sum(~prediction_mask[yc.flatten(), xc.flatten()]) < 10:
            if return_mask:
                prediction_mask[yc.flatten(), xc.flatten()] = False
            continue
        elif return_mask:
            prediction_mask[yc.flatten(), xc.flatten()] = True

        a = region.major_axis_length / 2
        b = region.minor_axis_length / 2
        r = np.sqrt(a * b)

        if region.orientation > 0:
            ellipse_angle = np.pi / 2 - region.orientation
        else:
            ellipse_angle = -np.pi / 2 - region.orientation

        Amin_pixels = np.pi * (r_min_pix) ** 2  # minimum region area based on minimum radius
        # filtering cells close to left and right image edge
        # usually cells do not come close to upper and lower image edge
        x_pos = region.centroid[1]
        dist_to_edge =  np.min([x_pos, prediction_mask.shape[1] - x_pos])

        if region.area >= Amin_pixels and dist_to_edge > edge_dist_pix:  # analyze only regions larger than 100 pixels,
            # and only of the canny filtered band-passed image returend an object

            # the circumference of the ellipse
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

            if 0:
                # %% compute radial intensity profile around each ellipse
                theta = np.arange(0, 2 * np.pi, np.pi / 8)

                i_r = np.zeros(int(3 * r))
                for d in range(0, int(3 * r)):
                    # get points on the circumference of the ellipse
                    x = d / r * a * np.cos(theta)
                    y = d / r * b * np.sin(theta)
                    # rotate the points by the angle fo the ellipse
                    t = ellipse_angle
                    xrot = (x * np.cos(t) - y * np.sin(t) + region.centroid[1]).astype(int)
                    yrot = (x * np.sin(t) + y * np.cos(t) + region.centroid[0]).astype(int)
                    # crop for points inside the iamge
                    index = (xrot < 0) | (xrot >= im.shape[1]) | (yrot < 0) | (yrot >= im.shape[0])
                    x = xrot[~index]
                    y = yrot[~index]
                    # average over all these points
                    i_r[d] = np.mean(im[y, x])

                # define a sharpness value
                sharp = (i_r[int(r + 2)] - i_r[int(r - 2)]) / 5 / np.std(i_r)
            sharp = 0

            # %% store the cells
            yy = region.centroid[0] - config["channel_width_px"] / 2
            yy = yy * config["pixel_size_m"] * 1e6

            data = {}

            data.update(frame_data)
            data.update({
                          "x": region.centroid[1],  # x_pos
                          "y": region.centroid[0],  # y_pos
                          "rp": yy,                  # RadialPos
                          "long_axis": float(format(region.major_axis_length)) * config["pixel_size_m"] * 1e6,  # LongAxis
                          "short_axis": float(format(region.minor_axis_length)) * config["pixel_size_m"] * 1e6,  # ShortAxis
                          "angle": np.rad2deg(ellipse_angle),  # angle
                          "irregularity": region.perimeter / circum,  # irregularity
                          "solidity": region.solidity,  # solidity
                          "sharpness": sharp,  # sharpness
            })
            cells.append(data)
    if return_mask:
        return cells, prediction_mask
    else:
        return cells


def mask_to_cells_edge2(prediction_mask, im, config, r_min, frame_data, edge_dist=15, return_mask=False, hollow_masks=True):
    r_min_pix = r_min / config["pixel_size"]
    edge_dist_pix = edge_dist / config["pixel_size"]
    cells = []
    # iterate over all detected regions
    from skimage.morphology import dilation
    for region in regionprops(label(dilation(prediction_mask)), cache=True):  # region props are based on the original image
        # checking if the anything was filled up by extracting the region form the original image
        # if no significant region was filled, we skip this object
        if hollow_masks is True and region.filled_area - region.area < 10:
            continue
        # get the offset for the filled image
        oy, ox = region.bbox[:2]
        # and from now on just use the properties of the filled version
        region = regionprops(region.filled_image.astype(np.uint8))[0]

        a = region.major_axis_length / 2
        b = region.minor_axis_length / 2
        r = np.sqrt(a * b)

        if region.orientation > 0:
            ellipse_angle = np.pi / 2 - region.orientation
        else:
            ellipse_angle = -np.pi / 2 - region.orientation

        Amin_pixels = np.pi * (r_min_pix) ** 2  # minimum region area based on minimum radius
        # filtering cells close to left and right image edge
        # usually cells do not come close to upper and lower image edge
        x_pos = region.centroid[1] + ox
        dist_to_edge =  np.min([x_pos, prediction_mask.shape[1] - x_pos])

        if region.area >= Amin_pixels and dist_to_edge > edge_dist_pix:  # analyze only regions larger than 100 pixels,

            # the circumference of the ellipse
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

            # %% store the cells
            yy = region.centroid[0] + oy - config["channel_width_px"] / 2
            yy = yy * config["pixel_size"]

            data = {}
            data.update(frame_data)
            data.update({
                  "x": region.centroid[1] + ox,
                  "y": region.centroid[0] + oy,
                  "radial_position": yy,
                  "long_axis": region.major_axis_length * config["pixel_size"],
                  "short_axis": region.minor_axis_length * config["pixel_size"],
                  "long_axis_px": region.major_axis_length,
                  "short_axis_px": region.minor_axis_length,
                  "angle": np.rad2deg(ellipse_angle),
                  "irregularity": region.perimeter / circum,
                  "solidity": region.solidity,
            })
            cells.append(data)
    if return_mask:
        return cells, prediction_mask
    else:
        return cells




def mask_to_cells(prediction_mask, im, config, r_min, frame_data, edge_dist=15):
    r_min_pix = r_min / config["pixel_size_m"] / 1e6
    edge_dist_pix = edge_dist / config["pixel_size_m"] / 1e6
    cells = []
    labeled = label(prediction_mask)

    # iterate over all detected regions
    for region in regionprops(labeled, im, coordinates='rc'):  # region props are based on the original image
        a = region.major_axis_length / 2
        b = region.minor_axis_length / 2
        r = np.sqrt(a * b)

        if region.orientation > 0:
            ellipse_angle = np.pi / 2 - region.orientation
        else:
            ellipse_angle = -np.pi / 2 - region.orientation

        Amin_pixels = np.pi * (r_min_pix) ** 2  # minimum region area based on minimum radius
        # filtering cells close to left and right image edge
        # usually cells do not come close to upper and lower image edge
        x_pos = region.centroid[1]
        dist_to_edge =  np.min([x_pos, prediction_mask.shape[1] - x_pos])

        if region.area >= Amin_pixels and dist_to_edge > edge_dist_pix:  # analyze only regions larger than 100 pixels,
            # and only of the canny filtered band-passed image returend an object

            # the circumference of the ellipse
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

            # %% compute radial intensity profile around each ellipse
            theta = np.arange(0, 2 * np.pi, np.pi / 8)

            i_r = np.zeros(int(3 * r))
            for d in range(0, int(3 * r)):
                # get points on the circumference of the ellipse
                x = d / r * a * np.cos(theta)
                y = d / r * b * np.sin(theta)
                # rotate the points by the angle fo the ellipse
                t = ellipse_angle
                xrot = (x * np.cos(t) - y * np.sin(t) + region.centroid[1]).astype(int)
                yrot = (x * np.sin(t) + y * np.cos(t) + region.centroid[0]).astype(int)
                # crop for points inside the iamge
                index = (xrot < 0) | (xrot >= im.shape[1]) | (yrot < 0) | (yrot >= im.shape[0])
                x = xrot[~index]
                y = yrot[~index]
                # average over all these points
                i_r[d] = np.mean(im[y, x])

            # define a sharpness value
            sharp = (i_r[int(r + 2)] - i_r[int(r - 2)]) / 5 / np.std(i_r)

            # %% store the cells
            yy = region.centroid[0] - config["channel_width_px"] / 2
            yy = yy * config["pixel_size_m"] * 1e6

            data = {}
            data.update(frame_data)
            data.update({
                          "x": region.centroid[1],  # x_pos
                          "y": region.centroid[0],  # y_pos
                          "rp": yy,                  # RadialPos
                          "long_axis": float(format(region.major_axis_length)) * config["pixel_size_m"] * 1e6,  # LongAxis
                          "short_axis": float(format(region.minor_axis_length)) * config["pixel_size_m"] * 1e6,  # ShortAxis
                          "angle": np.rad2deg(ellipse_angle),  # angle
                          "irregularity": region.perimeter / circum,  # irregularity
                          "solidity": region.solidity,  # solidity
                          "sharpness": sharp,  # sharpness
            })
            cells.append(data)
    return cells


def save_cells_to_file(result_file, cells):
    result_file = Path(result_file)
    output_path = result_file.parent

    with result_file.open('w') as f:
        f.write(
            'Frame' + '\t' + 'x_pos' + '\t' + 'y_pos' + '\t' + 'RadialPos' + '\t' + 'LongAxis' + '\t' + 'ShortAxis' + '\t' + 'Angle' + '\t' + 'irregularity' + '\t' + 'solidity' + '\t' + 'sharpness' + '\t' + 'timestamp' + '\n')
        f.write('Pathname' + '\t' + str(output_path) + '\n')
        for cell in cells:
            f.write("\t".join([
                str(cell["frames"]),
                str(cell["x"]),
                str(cell["y"]),
                str(cell["rp"]),
                str(cell["long_axis"]),
                str(cell["short_axis"]),
                str(cell["angle"]),
                str(cell["irregularity"]),
                str(cell["solidity"]),
                str(cell["sharpness"]),
                str(cell["timestamp"]),
              ])+"\n")
    print(f"Save {len(cells)} cells to {result_file}")


def matchVelocities(last_frame_cells, new_cells, next_cell_id, config):

    if len(last_frame_cells) != 0 and len(new_cells) != 0:
        conditions = (
            # radial pos
                (np.abs(np.array(last_frame_cells.radial_position)[:, None] - np.array(new_cells.radial_position)[None, :]) < 1) &
                # long_axis
                (np.abs(np.array(last_frame_cells.long_axis)[:, None] - np.array(new_cells.long_axis)[None, :]) < 1) &
                # short axis
                (np.abs(np.array(last_frame_cells.short_axis)[:, None] - np.array(new_cells.short_axis)[None, :]) < 1) &
                # angle
                (np.abs(np.array(last_frame_cells.angle)[:, None] - np.array(new_cells.angle)[None, :]) < 5) &
                # positive velocity
                (np.abs(np.array(last_frame_cells.x)[:, None] < np.array(new_cells.x)[None, :]))  # &
        )
        indices = np.argmax(conditions, axis=0)
        found = conditions[indices, np.arange(conditions.shape[1])]
        for i in range(len(indices)):
            if found[i]:
                j = indices[i]
                c1 = new_cells.iloc[i]
                c2 = last_frame_cells.iloc[j]
                dt = c1.timestamp - c2.timestamp
                v = (c1.x - c2.x) * config["pixel_size"] / dt
                new_cells.iat[i, new_cells.columns.get_loc("measured_velocity")] = v
                new_cells.iat[i, new_cells.columns.get_loc("cell_id")] = c2.cell_id
            else:
                new_cells.iat[i, new_cells.columns.get_loc("cell_id")] = next_cell_id
                next_cell_id += 1

    if len(last_frame_cells) == 0 and len(new_cells) != 0:
        for index in range(len(new_cells)):
            new_cells.iat[index, new_cells.columns.get_loc("cell_id")] = next_cell_id
            next_cell_id += 1

    return new_cells, next_cell_id
