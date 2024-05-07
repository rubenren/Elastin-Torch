import numpy as np
import torch
from PIL import Image, ImageOps

def add_margin(pil_img, top, bottom, left, right, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def shatter_img(in_img, size=352):
    img_len = len(np.asarray(in_img))
    img_wid = len(np.asarray(in_img)[0])
    # pad our image beforehand to make it easier for overlaying
    bottom_pad = size - img_len % size
    right_pad = size - img_wid % size
    img = np.asarray(add_margin(in_img, 0, bottom_pad, 0, right_pad, 0))
    
    img_len = len(img)
    img_wid = len(img[0])
    image_grid = []
    for i in range(size, img_len + 1, size):
        temp = []
        for j in range(size, img_wid + 1, size):
            temp.append(img[i-size:i,j-size:j])
        image_grid.append(temp)
    return image_grid

def align_datum(image_path = '', mask_path = '', color = False, size=512, ):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    if not color:
        image = ImageOps.grayscale(image)
        mask = ImageOps.grayscale(mask)
    images = shatter_img(image, size=size)
    masks = shatter_img(mask, size=size)

    aligned_lists = {'path':[], 'i':[], 'j':[], 'data':[], 'mask':[]}
    for i in range(len(images)):
        for j in range(len(images[0])):
            aligned_lists['path'].append(image_path)
            aligned_lists['i'].append(i)
            aligned_lists['j'].append(j)
            aligned_lists['data'].append(images[i][j] / 255)
            aligned_lists['mask'].append(masks[i][j] / 255)

    return aligned_lists