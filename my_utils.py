"""
Utilitie functions to manage georeferenced data and other basic operations.
"""

import logging
import numpy as np
from PIL import Image
from osgeo import gdal

def from_raster_to_np_array(raster_path):
    '''
    Takes a raster file(.TIFF) and returns it as a numpy array.
    Args:
        raster_path = path to raster file; may be a RGB raster(image) or a single channel raster(mask);
    Return:
        img_np = numpy array of shape (heigt, width, n° of channels);
    '''
    raster = gdal.Open(raster_path)
    num_channels = raster.RasterCount
    
    if num_channels>=3:
        band1 = raster.GetRasterBand(1) # Red channel
        band2 = raster.GetRasterBand(2) # Green channel
        band3 = raster.GetRasterBand(3) # Blue channel
        
        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        img_np = np.dstack((b1, b2, b3))
    elif num_channels==1:
        band1 = raster.GetRasterBand(1)
        img_np = band1.ReadAsArray()
        
    return img_np


def load_image_as_np_array(path):
    """
    Takes a path to an image file and returns it as a numpy array;
    Does not work with raster data (load_raster_as_np_array alternative function);
    Args:
        path = path to image file; may be a RGB image or a single channel image(mask);
    Return:
        numpy array of shape (heigt, width, n° of channels); 
    """
    with open(path, 'rb') as file:
        PIL_image = Image.open(file)
        PIL_image_width, PIL_image_height = PIL_image.size
        
        if PIL_image.mode == 'RGB':
            return np.array(np.reshape(PIL_image.getdata(), (PIL_image_height, PIL_image_width, 3)))
        elif PIL_image.mode == "L":
            return np.array(np.reshape(PIL_image.getdata(), (PIL_image_height, PIL_image_width)))
        else:
            print('PIL_image.mode =! "RGB" OR "L"')


def extract_masks_from_cluster(clustered_mask, bool_array=False, ch_first=False):
    '''
    Extract unique arrays each containing a single instance from a "clustered mask" and stores them as a numpy array.
    
    Args:
        clustered_mask: An array of shape (height, width) containing an unique value[int] for each instace and zeros elsewhere;
        bool_array: set this to True if you want mask pixels containing bool values instead of original values[int];
        ch_first: set this to True if you want output shape to be (n° of instances, height, width);
    Return:
        masks: numpy array of shape (height, width, n° of instances);
        
    Obs.: It seems the clustered mask obtained via ArcGIS doesn't consistently give sorted values 
    for each instance ([1, 2, 3, 4...]), which means np.unique(clustered_mask) might give something
    like ([0, 48, 50, 51, 53...]);
    ''' 
    masks_list = []
    
    for i in np.delete(np.unique(clustered_mask), 0):    # removing BC mask
        if bool_array:
            mask = np.where(clustered_mask==i, True, False)
            masks_list.append(mask)
        else:
            mask = np.where(clustered_mask==i, i, 0)
            masks_list.append(mask)
    
    if ch_first:
        return np.array(masks_list)
    else:
        return np.dstack(masks_list)
    

def extract_bboxes_from_mask(mask):
    """
    Compute bounding boxes from masks.
    Args:
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: 
        bbox array [num_instances, (x1, y1, x2, y2)].
    
    Copied and adapted from matterpot's Mask RCNN repo.
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
            # logging.warning("extract_bboxes_from_mask func returned no bbox!")
        boxes[i] = np.array([x1, y1, x2, y2])       # ORIGINAL: [y1, x1, y2, x2]
    return boxes.astype(np.float32)
    
    
def rle_encode(mask):
    """
    Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    
    Copied from matterpot's Mask RCNN repo.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """
    Decodes an RLE encoded list of space separated
    numbers and returns a binary mask.
    
    Copied from matterpot's Mask RCNN repo.
    """
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask
