import cv2
import numpy as np
from PIL import Image
import os
import pandas
import pygame


def make_background():
    image=' '
    file_without_extension = image.split('.')[0]
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    trans_mask = image[:, :, 3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(file_without_extension + '.jpeg', new_img)


def shift():
    image = ''
    img = cv2.imread(image)
    file_without_extension = image.split('.')[0]
    arr_translation = [[15, -15], [-15, 15], [-15, -15],
                       [15, 15]]
    arr_caption = ['15-15', '-1515', '-15-15', '1515']
    for i in range(4):
        transform = cv2.AffineTransform(
            translation=tuple(arr_translation[i]))
        warp_image = cv2.warp(img, transform, mode="wrap")
        img_convert = cv2.convertScaleAbs(warp_image,
                                          alpha=(255.0))
        cv2.imwrite(file_without_extension +
                    arr_caption[i] + '.jpeg', img_convert)

def rotate():
    image = ''
    img = Image.open(image)
    file_without_extension = image.split('.')[0]
    angles = np.ndarray((2,),
        buffer=np.array([-13, 13]), dtype=int)
    for angle in angles:
        transformed_image = cv2.transform.rotate(np.array(img),
                                                 angle, cval=255, preserve_range=True).astype(np.uint8)
        cv2.imwrite(file_without_extension +
                    str(angle) + '.jpeg', transformed_image)

def balancing():
    arr_len_files = []
    for path in 'Cycrillic':
        name_path = 'C:%Users%kigab%PycharmProjects%neuro'+path+'/'
        files=os.listdir(name_path)
        arr_len_files.append(len(files))

    min_value=min(arr_len_files)
    for path in 'Cycrillic':
        folder = 'C:%Users%kigab%PycharmProjects%neuro'+path
        arr = []
        for the_file in os.listdir(folder):
            arr.append(folder + '/' + the_file)
        d = 0
        k = len(arr)
        for i in arr:
            os.remove(i)
            d += 1
            if d == k - min_value:
                break