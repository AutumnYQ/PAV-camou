import os
import re
import cv2
import sys
import mmcv
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import Dataset,DataLoader

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# reproduce testpipeline with the format torch tensor 
from tools.pre_processor import LoadImageFromTensor, Tensor_Resize,LoadAnnotations_Tensor,PackDet_Tensor
from tools.dataset import MMdetAdv
from tools.evaluate import evaluate

import torchvision.transforms as transforms

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import sys
from math import sqrt
sys.path.insert(0,'/home/liuyuqiu/ssd3/mmdetection')
from mmdet.apis import init_detector

device_id = 9
    

test_data_dir = '../dataset/xfl_data_lt45/val'
# obj_file = '../dataset/xfl_5000/xfl_yxz.obj'
obj_file = '../dataset/xfl_5000/xfl_yxz_full.obj'
log_dir = '../log'
tex_mask = False
batch_size = 1
img_size = 800
save_tex_img = True
cfg_mdl_dir = '/ssd3/liuyuqiu/mmdetection/cfg_mdl'
# model = 'yolov3_d53_mstrain-608_273e_coco'
test_texs = [
            # '../log/full_20e_faster_20_tex_False_07_30_01_26/faster__16_0039.npy', # our full
            # '../log/lcls_20e_faster_20_tex_True_07_30_01_18/faster__6_0937.npy',  # lcls
            # '../log/lobj_20e_faster_20_tex_True_07_30_01_22/faster__7_1040.npy', # lobj
            # '../log/lclsobj_20e_faster_20_tex_True_07_30_01_21/faster__16_0827.npy', # lcls + lobj
            # '../log/XFL_20e_ssd512_20_tex_True_07_30_01_08/ssd512__17_3643.npy', # ssd
            # '../log/XFL_20e_dino-4scale_20_tex_True_07_30_14_04/dino-4scale__19_0688.npy', # dino
            # '../log/XFL_20e_ddq-detr-4scale_20_tex_True_07_30_21_28/ddq-detr-4scale__19_1897.npy' # ddq
            '../log/full_20e_faster_20_tex_False_07_30_01_26/faster__19_0039.npy'
            ]

test_models = ['faster_rcnn_r50_fpn_1x_coco', 'yolov3_d53_mstrain-608_273e_coco', 'ssd512_coco', 'detr_r50_8xb2-150e_coco', 'dino-4scale_r50_8xb2-12e_coco', 'ddq-detr-4scale_r50_8xb2-12e_coco']
save_path = log_dir + '/textures'
# GPU
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:"+str(device_id))
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


dataset = MMdetAdv(obj_file, test_data_dir, device, tex_mask=tex_mask)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) 
if save_tex_img:
    for tex_path in test_texs:
        tex_map = torch.tensor(np.load(tex_path)).to(device)
        save = save_path + '/' + tex_path.split('/')[-2] + '.png'
        dataset.set_maps(tex_map)
        map_ = dataset.mesh.textures._maps_padded
        Image.fromarray((map_[0]*255).data.cpu().numpy().astype('uint8')).save(save) 
exit() # only save images  

# model setting
nets = {}
chk_list = os.listdir(cfg_mdl_dir)
for model in test_models:
    for cfg in chk_list:
        mdl_name = model.split('.')[0]
        # find config and pretrained model file
        if cfg.split('.')[0].startswith(mdl_name) and cfg.split('.')[-1]=='pth':
            chk_file = os.path.join(cfg_mdl_dir, cfg)
        if cfg.split('.')[0].startswith(mdl_name) and cfg.split('.')[-1]=='py':
            cfg_file = os.path.join(cfg_mdl_dir, cfg)
    config = cfg_file
    checkpoint = chk_file
    nets[model] = init_detector(config, checkpoint, device=device) 


    # evaluate textures with all models

test_pipeline = transforms.Compose([LoadImageFromTensor(), Tensor_Resize((img_size,
                img_size,)), PackDet_Tensor()])

date = time.strftime("%m_%d_%H_%M", time.localtime())

loggerf = open(log_dir +'/log_'+ date + "_test.txt","w")
for tex_path in test_texs:
    tex_map = torch.tensor(np.load(tex_path)).to(device)
    log_dir_model = tex_path[-4:]
    thrs = np.array([0.5])
    loggerf.write('texture:'+ tex_path + ':\n')
    for test_model in test_models:
        test_model_name = test_model.split('_')[0]
        test_n = nets[test_model]
        # test_clean
        p_res, ASR = evaluate(test_data_dir, test_pipeline, test_n, obj_file, tex_map, device, clean=True, thrs=thrs)
        PR= p_res.coco_eval['bbox'].stats
        print('clean:' + test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
        loggerf.write('clean:'+ test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
        thrs = np.array([0.5])
        p_res, ASR = evaluate(test_data_dir, test_pipeline, test_n, obj_file, tex_map.detach(), device,clean=False, thrs=thrs)
        PR= p_res.coco_eval['bbox'].stats
        print('texture:'+ tex_path + ':\n')
        print('test: '+ test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
        loggerf.write('test: '+ test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
loggerf.close()