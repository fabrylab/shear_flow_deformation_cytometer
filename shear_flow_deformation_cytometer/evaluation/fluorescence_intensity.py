import numpy as np
import imageio
from pathlib import Path
import pandas as pd
from skimage.draw import ellipse

leave_black_spots_out = True

def load_fluo_flatfield(vidcap):
    sum_frames = 0
    num_frames = 0
    image_count = len(vidcap)
    for image_index in range(image_count):
        image_index = int(image_index)
        im = vidcap.get_data(image_index)
        num_frames = num_frames + 1
        im_np = np.array(im, dtype=np.float32)
        sum_frames = sum_frames + im_np

    flatfield = sum_frames / num_frames
    flatfield = flatfield / np.mean(flatfield)
    return flatfield

def get_fluo_flatfield(vidcap):
    sum_frames = 0
    num_frames = 0
    image_count = len(vidcap)

    for image_index in range(image_count):
        image_index = int(image_index)
        im = vidcap.get_data(image_index)
        num_frames = num_frames + 1
        im_np = np.array(im, dtype=np.float32)
        sum_frames = sum_frames + im_np

    flatfield_correction = sum_frames / num_frames
    flatfield_correction = flatfield_correction / np.mean(flatfield_correction)
    #print(num_frames)
    return flatfield_correction


def get_fluorescence_intensity(filename, data, optional=False):
    if filename.endswith("_B.tif"):
        video = filename.replace("_B.tif", "_Fl.tif")
    else:
        video = filename.replace(".tif", "_Fl.tif")

    if not Path(video).exists() and optional:
        return data

    video_reader = imageio.get_reader(video)
    #flatfield_reader = imageio.get_reader("//131.188.117.96/biophysDS/khast/2022.3.17/flatfield/2022_03_17_10_43_09_Fl.tif")

    flat_correction = get_fluo_flatfield(video_reader)
    #print(np.mean(flat_correction), np.median(flat_correction))
    #flatfield_correction = load_fluo_flatfield(flatfield_reader)

    im_corrected = None
    im_index = None
    new_data = []
    # iterate over all cells
    for index, cell_data in data.iterrows():
        # get the image if it is not the same as from the last cell
        if im_index != cell_data.frame:
            im_index = int(cell_data.frame)
            #print(im_index)
            imm = video_reader.get_data(im_index)
            #imm = imm - background_correction
            #imm[imm<0] = 0
            #im_corrected = imm/ flatfield_correction
            im_corrected = imm / flat_correction
            if leave_black_spots_out == True:
                im_corrected= np.ma.masked_where(flat_correction < np.median(flat_correction), im_corrected)
                #im_corrected =  mask_black_spots.data
        # get the pixel positions of the cell ellipse
        rr, cc = ellipse(cell_data.y, cell_data.x,
                         cell_data.short_axis_px / 2, cell_data.long_axis_px / 2,
                         rotation=-cell_data.angle * np.pi / 180)
        # and get the pixel values
        #cell_pixels1 = im_corrected1[rr, cc]
        cell_pixels = im_corrected[rr, cc]
        #print(cell_pixels, cell_pixels1)
        #cell_pixels = cell_pixels.compressed()
        #print(cell_pixels1)
        #print(cell_pixels)

        # calculate various statistics of these pixel values
        black_spot_inside = np.ma.is_masked(cell_pixels)
        #print(index, black_spot_inside)
        if black_spot_inside == False:
            #print(cell_pixels.mean())
            new_data.append(dict(
                mean_intensity= cell_pixels.mean(), #np.nanmean(cell_pixels),
                integral_intensity= cell_pixels.sum(), #np.nansum(cell_pixels),
                max_intensity= cell_pixels.max(), #np.nanmax(cell_pixels),
                #percent90_intensity= #np.nanpercentile(cell_pixels, 90),
                std_intensity= cell_pixels.std() ,#np.nanstd(cell_pixels),
            ))
        else:
            new_data.append(dict(
                mean_intensity= None,  # np.nanmean(cell_pixels),
                integral_intensity= None,  # np.nansum(cell_pixels),
                max_intensity= None,  # np.nanmax(cell_pixels),
                # percent90_intensity= #np.nanpercentile(cell_pixels, 90),
                std_intensity= None,  # np.nanstd(cell_pixels),
            ))
    # add the new values to the dataframe and return it
    return pd.concat([data, pd.DataFrame(new_data)], axis=1)


if __name__ == "__main__":
    from shear_flow_deformation_cytometer.evaluation.helper_load import load_all_data_new

    filename = r"\\131.188.117.96\biophysDS2\nstroehlein\EA hy 926\2022.2.16\with gadolidium\5 bar\2022_02_16_13_41_23_B.tif"
    data, config = load_all_data_new(filename)
    data = get_fluorescence_intensity(filename, data, config)
    print(data)
