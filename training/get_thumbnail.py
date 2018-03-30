import utils
import openslide
import cv2
import os
import numpy as np

def generate_files():
    map_level = 4
    utils.ensure_dir('png')
    for root, dirs, files in os.walk('tif'):
        for file in files:
            if file.endswith('.tif') and not file.startswith('._'):
                print(file)
                slide_path = os.path.join(root, file)
                prefix = file.split('.tif')[0]
                thumbnail_path = os.path.join('png', prefix + '.png')
                slide = openslide.OpenSlide(slide_path)
                slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[map_level]))
                cv2.imwrite(thumbnail_path, slide_map)

if __name__ == '__main__':
    generate_files()

