import utils
import openslide
import cv2
import numpy as np
import os
import random
import csv

from xml.etree.ElementTree import parse
from PIL import Image




def _read_xml(xml_path, mask_level):
    ''' read xml files which has tumor coordinates list
        return coordinates of tumor areas

    Args:
        slide_num (int): number of slide used
        maks_level (int): level of mask
    '''

    xml = parse(xml_path).getroot()
    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X'))/(2**mask_level)),
                            round(float(area.get('Y'))/(2**mask_level))])
        coors_list.append(coors)
        coors=[]
    return np.array(coors_list)


def _make_masks(slide_path, xml_path, mask_level, map_level, **args):    
    #make tumor, normal, tissue mask using xml files and otsu threshold
    print('_make_masks(%s)' % slide_path)

    #slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[map_level]))

    # xml loading
    coors_list = _read_xml(xml_path, mask_level)
    
    # draw boundary of tumor in map
    for coors in coors_list:
        cv2.drawContours(slide_map, np.array([coors]), -1, 255, 1)

    # draw tumor mask
    tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1])
    for coors in coors_list:
        cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)

    # draw tissue mask
    slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
    slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
    slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
    slide_lv = slide_lv[:, :, 1]
    _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    # check normal mask / draw normal mask
    # tumor_mask = cv2.imread(tumor_mask_path, 0) 
    height, width = np.array(tumor_mask).shape
    normal_mask = np.array(tissue_mask)
    for i in range(width):
        for j in range(height):
            if tumor_mask[j][i] > 127:
                normal_mask[j][i] = 0

    return slide, slide_map, tumor_mask, tissue_mask, normal_mask


def _write_masks(mask_folder_path, slide_map, tumor_mask, tissue_mask, normal_mask, **args):
    print('_write_masks')
    utils.ensure_dir(mask_folder_path)
    map_path = os.path.join(mask_folder_path, 'map.png')
    cv2.imwrite(map_path, slide_map)
    tumor_mask_path = os.path.join(mask_folder_path, 'tumor_mask.png')
    cv2.imwrite(tumor_mask_path, tumor_mask) # CHANGED
    tissue_mask_path = os.path.join(mask_folder_path, 'tissue_mask.png')
    cv2.imwrite(tissue_mask_path, np.array(tissue_mask))
    normal_mask_path = os.path.join(mask_folder_path, 'normal_mask.png')
    cv2.imwrite(normal_mask_path, normal_mask)



def _write_patches(patch_folder_path, mask_level, patch_size,
                    slide, slide_map, tumor_mask, tissue_mask, normal_mask,
                    tumor_threshold, 
                    tumor_sel_ratio, 
                    tumor_sel_max, 
                    normal_threshold, 
                    normal_sel_ratio, 
                    normal_sel_max, 
                    **args):
    # Extract normal, tumor patches using normal, tumor mask

    width, height = np.array(slide.level_dimensions[0])//patch_size
    total = width * height
    all_cnt = 0
    t_cnt = 0
    n_cnt = 0
    t_over = False
    n_over = False
    step = int(patch_size/(2**mask_level))
    utils.ensure_dir(patch_folder_path)

    print('_write_patches(w=%d,h=%d)' % (width,height))
    margin = 3

    for i in range(width):
        for j in range(height):
            if i < margin or j < margin or i >= width - margin or j >= height - margin: continue

            ran = random.random()
            tumor_mask_sum = tumor_mask[step * j : step * (j+1),
                                            step * i : step * (i+1)].sum()
            normal_mask_sum = normal_mask[step * j : step * (j+1),
                                            step * i : step * (i+1)].sum()
            mask_max = step * step * 255
            tumor_area_ratio = tumor_mask_sum / mask_max
            normal_area_ratio = normal_mask_sum / mask_max

            # extract tumor patch
            if (tumor_area_ratio > tumor_threshold) and (ran <= tumor_sel_ratio) and not t_over:
                patch = slide.read_region((patch_size*i,patch_size*j), 0, (patch_size,patch_size))
                patch_name = os.path.join(patch_folder_path, 't_' + str(i) + '_' + str(j) + '.png')
                patch.save(patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,0,255), 1)
                t_cnt += 1
                t_over = (t_cnt > tumor_sel_max)
            
            # extract normal patch
            elif (normal_area_ratio > normal_threshold) and (ran <= normal_sel_ratio) and (tumor_area_ratio == 0) and not n_over:
                patch = slide.read_region((patch_size*i,patch_size*j), 0, (patch_size,patch_size))
                patch_name = os.path.join(patch_folder_path, 'n_' + str(i) + '_' + str(j) + '.png')
                patch.save(patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (255,255,0), 1)
                n_cnt += 1
                n_over = (n_cnt > normal_sel_max)
            
            # nothing
            else:
                pass

            # check max boundary of patch
            if n_over and t_over:
                print('\nPatch Selection Boundary OVER')
                return
            
            all_cnt += 1
            print('\rProcess: %.3f%%,  All: %d, Normal: %d, Tumor: %d'
                %((100.*all_cnt/total), all_cnt, n_cnt, t_cnt), end="")
    print('')



def generate_file(xml_path, slide_path, folder_path):
    args = {
        'slide_path' : slide_path,
        'xml_path': xml_path,
        'mask_level' : 4,
        'map_level' : 4,
        'patch_size' : 304,
        'tumor_threshold' : 0.8,
        'tumor_sel_ratio' : 1,
        'tumor_sel_max' : 100000,
        'normal_threshold' : 0.1,
        'normal_sel_ratio' : 1,
        'normal_sel_max' : 100000,
        'mask_folder_path' : folder_path,
        'patch_folder_path' : folder_path + '/patches',
    }
    args['slide'], args['slide_map'], args['tumor_mask'], args['tissue_mask'], args['normal_mask'] = _make_masks(**args)
    _write_patches(**args)
    _write_masks(**args)


def generate_files():
    for root, dirs, files in os.walk('lesion_annotations'):
        for file in files:
            if file.endswith('.xml') and not file.startswith('._'):
                prefix = file.split('.xml')[0]
                generate_file(os.path.join(root, file), 'tif/' + prefix + '.tif', 'patches/' + prefix)


if __name__ == '__main__':
    generate_files()

