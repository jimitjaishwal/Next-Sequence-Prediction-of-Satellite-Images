import os
import numpy as np
from PIL import Image

from os import listdir
from PIL import Image

from skimage.io import imread
from skimage.transform import resize 

from os.path import isfile, isdir


def get_clean_data(path = 'data'):
    data_dir = os.path.join(os.getcwd(), path)
    image_files = [os.path.join(data_dir, f) for f in listdir(data_dir) if isfile(os.path.join(data_dir, f))]
    
    def get_image_id(path):
        names = []
        for i, name in enumerate(path):
            image_id = name[name.find('data')+5:name.find('.')]
            names.append(image_id)
        return names
    
    def get_info_from_image(index):
        image_path = image_files[index]
        image_obj = Image.open(image_path)
        coords = (480, 14, 580, 27)
        corped_image = image_obj.crop(coords)
        return np.array(corped_image)
    
    image_ids = get_image_id(image_files)
    croped_image = get_info_from_image(2)
    
    invalid_image_paths = []
    
    for i, image_path in enumerate(image_files):
        
        try:
            IMG = Image.open(image_path)
            img = get_info_from_image(i)
            if np.min(croped_image == img) == True:
                coord = (0, 0, 480, 480)
                cropped_image = IMG.crop(coord)
                dirs = os.path.join(os.getcwd(), 'New Data',image_ids[i] + '.png')
                cropped_image.save(dirs)
        except:
            invalid_image_paths.append(image_files)
    
    return print('We have {} Invalid Images found from the dataset.'.format(len(invalid_image_paths)))