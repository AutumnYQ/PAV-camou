import os
import re
import cv2
import sys
import mmcv
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = '/data/liuyuqiu/code/dataset/xfl_data_lt45/val'
save_new_npz = True # whether to save coco label and mask into npz as a new file (path: {datapath}/com_npz)
save_image = False # whether to export image from old npz file (path: {datapath}/png)
save_yolo_label = False # whether to save label as yolo format (path: {datapath}/label)
save_coco_label = True
cls_id = 2 # label id 


npz_path  = data_path + '/npz'
img_path  = data_path + '/png'
lbl_path  = data_path + '/label'
msk_path  = data_path + '/mask_car'
tex_path  = data_path + '/mask_tex'

# including all message
compile_path = data_path + '/com_npz'
if not os.path.exists(compile_path):
    os.mkdir(compile_path)
# export images in old npz
if not os.path.exists(img_path):
    os.mkdir(img_path)
if not os.path.exists(lbl_path):
    os.mkdir(lbl_path)

npz_names = sorted(os.listdir(npz_path), key=lambda x: int(re.findall("\d+", x)[0]))


# prepare for coco label
if save_coco_label:    
    dataset = {'categories': [], 'annotations': [], 'images': []}
    dataset['categories'].append({'id': 2, 'name': 'car', 'supercategory': 'vehicle'})
    json_name = data_path + '/' + data_path.split('/')[-1] + '.txt'
    print(json_name)

for npz_name in tqdm(npz_names):
    npz_file = os.path.join(npz_path, npz_name)
    lbl_file = os.path.join(lbl_path, npz_name.split('.')[0]+'.txt')
    msk_file = os.path.join(msk_path, npz_name.split('.')[0]+'.png')
    tex_file = os.path.join(tex_path, npz_name.split('.')[0]+'.png')
    img_file = os.path.join(img_path, npz_name.split('.')[0]+'.png')
    mask_car = cv2.imread(msk_file)  # H*W*3
    mask_tex = cv2.imread(tex_file)  # H*W*3
    old_npz = np.load(npz_file, allow_pickle=True)

    # use mask to generate labels
    w_test = mask_car[:,:,0].sum(axis=0) # add h, len = w
    h_test = mask_car[:,:,0].sum(axis=1) # add w, len = h
    w_test = np.where(w_test, 1, 0)
    h_test = np.where(h_test, 1, 0)
    cen = int(len(h_test)/2)  # image center index
    x1 = cen - w_test[:cen].sum()
    y1 = cen - h_test[:cen].sum()
    x2 = cen + w_test[cen:].sum()
    y2 = cen + h_test[cen:].sum()
    bbox = np.array([cls_id, x1, x2, y1, y2]) 

    # save the label with yolo format
    if save_yolo_label: 
        height = mask_car.shape[0]
        width = mask_car.shape[1]
        x_center = (x1+x2)/2/width
        y_center = (y1+y2)/2/height
        w = (x2-x1)/width
        h = (y2-y1)/height
        bbox_yolo = [cls_id, x_center, y_center, w, h]
        with open(lbl_file, 'w') as f:
            f.write("%d %s %s %s %s" % (cls_id, x_center, y_center, w, h))

    # save all info into a new npz file (including the label and mask)
    if save_new_npz:
        npz_com_file = os.path.join(compile_path, npz_name)
        image, veh_trans, cam_trans = old_npz['data'],old_npz['veh_trans'],old_npz['cam_trans']
        # new_npz
        np.savez(npz_com_file, data=image, mask_car=mask_car, mask_tex=mask_tex, veh_trans=veh_trans, cam_trans=cam_trans, label=bbox)

    # save image from npz file
    if save_image:
        cv2.imwrite(img_file, old_npz['data'])

    if save_coco_label:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)        # np.int64 to int for json.dump\
        box_w = x2 - x1
        box_h = y2 - y1
        index = int(re.findall("\d+", npz_name)[0])
        dataset['images'].append({'file_name': img_file,
                                  'id': index, 
                                  'width': box_w,
                                  'height': box_h})
        dataset['annotations'].append({
                    'area': box_h * box_w,
                    'bbox': [x1, y1, box_w, box_h],
                    'category_id': int(cls_id),
                    'id': index,
                    'image_id': index,
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

if save_coco_label:
    with open(json_name, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)


    


    

    