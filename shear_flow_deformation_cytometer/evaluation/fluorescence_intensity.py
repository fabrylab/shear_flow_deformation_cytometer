import numpy as np
import imageio
from pathlib import Path
import pandas as pd
from skimage.draw import ellipse


def get_fluo_flatfield(frames_with_cells, vidcap):
    sum_frames = 0
    num_frames = 0

    for image_index in frames_with_cells:
        image_index = int(image_index)
        im = vidcap.get_data(image_index)
        num_frames = num_frames + 1
        im_np = np.array(im, dtype=np.float32)
        sum_frames = sum_frames + im_np

    flatfield_correction = sum_frames / num_frames
    flatfield_correction = flatfield_correction / np.mean(flatfield_correction)
    return flatfield_correction


def get_fluorescence_intensity(filename, data, optional=False):
    if filename.endswith("_B.tif"):
        video = filename.replace("_B.tif", "_Fl.tif")
    else:
        video = filename.replace(".tif", "_Fl.tif")

    if not Path(video).exists() and optional:
        return data

    video_reader = imageio.get_reader(video)

    flatfield_correction = get_fluo_flatfield(data.frame, video_reader)

    im_corrected = None
    im_index = None
    new_data = []
    # iterate over all cells
    for index, cell_data in data.iterrows():
        # get the image if it is not the same as from the last cell
        if im_index != cell_data.frame:
            im_index = int(cell_data.frame)
            imm = video_reader.get_data(im_index)
            im_corrected = imm / flatfield_correction

        # get the pixel positions of the cell ellipse
        rr, cc = ellipse(cell_data.y, cell_data.x,
                         cell_data.short_axis_px / 2, cell_data.long_axis_px / 2,
                         rotation=-cell_data.angle * np.pi / 180)
        # and get the pixel values
        cell_pixels = im_corrected[rr, cc]

        # calculate various statistics of these pixel values
        new_data.append(dict(
            mean_intensity=np.nanmean(cell_pixels),
            integral_intensity=np.sum(cell_pixels),
            max_intensity=np.max(cell_pixels),
            percent90_intensity=np.percentile(cell_pixels, 90),
            std_intensity=np.std(cell_pixels),
        ))
    # add the new values to the dataframe and return it
    return pd.concat([data, pd.DataFrame(new_data)], axis=1)


if __name__ == "__main__":
    from shear_flow_deformation_cytometer.evaluation.helper_load import load_all_data_new

    filename = r"\\131.188.117.96\biophysDS2\nstroehlein\EA hy 926\2022.2.16\with gadolidium\5 bar\2022_02_16_13_41_23_B.tif"
    data, config = load_all_data_new(filename)
    data = get_fluorescence_intensity(filename, data, config)
    print(data)