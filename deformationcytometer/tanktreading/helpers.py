import numpy as np
import scipy.optimize
import scipy.special
import skimage.registration
import skimage.registration
from scipy.ndimage import morphology
from scipy.ndimage import shift
import imageio


class CachedImageReader:
    def __init__(self, video, cache_count=10):
        self.image_reader = imageio.get_reader(video)
        self.frames = {}
        self.cache_count = cache_count

    def get_data(self, index):
        if index not in self.frames:
            self.frames[index] = self.image_reader.get_data(index)
        if len(self.frames) >= self.cache_count:
            if np.min(list(self.frames.keys())) != index:
                del self.frames[np.min(list(self.frames.keys()))]
        return self.frames[index]


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
    return scipy.special.ellipeinc(angle, e) / perimeter * 2 * np.pi  # - sp.special.ellipeinc(angle-0.1, e)


def getArcLength(points, major_axis, minor_axis, ellipse_angle, center):
    p = points - np.array(center)  # [None, None]
    alpha = np.deg2rad(ellipse_angle)
    p = p @ np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    distance_from_center = np.linalg.norm(p, axis=-1)
    angle = np.arctan2(p[..., 0], p[..., 1])
    angle = np.arctan2(np.sin(angle) / (major_axis / 2), np.cos(angle) / (minor_axis / 2))
    angle = np.unwrap(angle)

    r = np.linalg.norm([major_axis / 2 * np.sin(angle), minor_axis / 2 * np.cos(angle)], axis=0)

    length = getEllipseArcSegment(angle, minor_axis / 2, major_axis / 2)
    return length, distance_from_center / r


def getCroppedImages(image_reader, cells, w=60, h=40, o=5, o2=15):
    crops = []
    shifts = []
    valid = []
    im0 = None
    shift0 = None
    # iterate over all cells
    for index, cell in enumerate(cells.itertuples()):
        # get the image
        im = image_reader.get_data(cell.frame)
        # get the cell position
        y = int(round(cell.y))
        x = int(round(cell.x))
        # crop the image
        crop = im[y - h - o:y + h + o, x - w - o:x + w + o]
        # if the image does not have the full size, skip it (e.g. it is at the border)
        if crop.shape[0] != h * 2 + o * 2 or crop.shape[1] != w * 2 + o * 2:
            crops.append(np.ones([h * 2, w * 2]) * np.nan)
            shifts.append([0, 0])
            valid.append(False)
            continue

        # if it is the first image, we cannot do any image registration
        if im0 is None:
            # we just move it by the float point part of the cell position
            shift_px = [cell.y - y, cell.x - x]
            # print(shift_px)
            shifts.append([0, 0])
            shift0 = np.array(shift_px)
        else:
            # try to register the image
            try:
                shift_px, error, diffphase = skimage.registration.phase_cross_correlation(im0[o2:-o2, o2:-o2],
                                                                                          crop[o2:-o2, o2:-o2],
                                                                                          upsample_factor=100)
            except ValueError:
                # if it is not successfully, skip the image
                crops.append(np.ones(h * 2, w * 2) * np.nan)
                shifts.append([0, 0])
                valid.append(False)
                continue
            # print(shift_px, type(shift_px))
            shifts.append(-np.array([cell.y - y, cell.x - x]) + shift_px)
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
    if np.sum(~np.isnan(crops)) == 0:
        return [], [], []
    crops -= np.nanmin(crops)
    crops /= np.nanmax(crops)
    crops *= 255
    crops = crops.astype(np.uint8)
    return crops, np.array(shifts), np.array(valid).astype(np.bool)


def getCenterLine(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    def func(x, m):
        return x * m

    import scipy.optimize
    p, popt = scipy.optimize.curve_fit(func, x, y, [1])
    return p[0], 0


def doTracking(images, data0, times, pixel_size):
    data_x = []
    data_y = []

    np.seterr(divide='ignore', invalid='ignore')

    perimeter_pixels = getPerimeter(data0.long_axis.mean() / pixel_size / 2, data0.short_axis.mean() / pixel_size / 2)

    for i in range(len(images) - 1):
        dt = times[i + 1] - times[i]
        flow = skimage.registration.optical_flow_tvl1(images[i], images[i + 1], attachment=30)

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
        data_y.extend(projected_speed[indices_middle] / dt / perimeter_pixels)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    i = np.isfinite(data_x) & np.isfinite(data_y)
    data_x = data_x[i]
    data_y = data_y[i]

    if len(data_y) == 0:
        return 0, 0
    m, t = getCenterLine(data_x, data_y)

    cr = np.corrcoef(data_y, m * np.array(data_x))
    r2 = np.corrcoef(data_y, m * np.array(data_x))[0, 1] ** 2

    return m, r2
