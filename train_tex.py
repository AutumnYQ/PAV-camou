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
sys.path.insert(0,'../mmdetection') # dir to mmdet
# before use: 
# modify two_stage.py and standard_roi_head.py (in mmdet/models/detectors, predict()) to get cls_scores
# modify single_stage.py, yolo_head.py, base_dense_head.py to get cls_scores
from mmdet.apis import init_detector

device_id = 9

# train
train_data_dir = '../xfl_data_lt45/train' # train_dir
test_data_dir = '../xfl_data_lt45/val'
obj_file = '../dataset/xfl_5000/xfl_yxz.obj'
log_dir = '///log'
tex_mask = True
test_all_model = True
batch_size = 1
img_size = 800

cfg_mdl_dir = '../mmdetection/cfg_mdl' # downloaded config & checkpoint
# model = 'yolov3_d53_mstrain-608_273e_coco'
train_models = ['ssd512_coco']
test_models = ['faster_rcnn_r50_fpn_1x_coco', 'yolov3_d53_mstrain-608_273e_coco', 'ssd512_coco', 'detr_r50_8xb2-150e_coco', 'dino-4scale_r50_8xb2-12e_coco', 'ddq-detr-4scale_r50_8xb2-12e_coco']
epochs = 20
target_class = 2
lr = 0.001

# GPU
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:"+str(device_id))
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def loss_smooth(img):
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return torch.sum((s1 + s2)) / (img.shape[-1]**2)

# model setting
nets = {}
chk_list = os.listdir(cfg_mdl_dir)
load_models = list(set(train_models) | set(test_models)) if test_all_model else train_models
print(load_models)
for model in load_models:
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

for model in test_models:
    if model not in train_models:
        continue
    net = nets[model]
    test_pipeline = transforms.Compose([LoadImageFromTensor(), Tensor_Resize((img_size,
                    img_size,)), PackDet_Tensor()])

    # data setting
    train_dataset = MMdetAdv(obj_file, train_data_dir, device, tex_mask=tex_mask)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # dataloader

    tex_map = torch.zeros(train_dataset.map_png.shape).to(device) + 0.5
    tex_map = torch.autograd.Variable(tex_map, requires_grad=True) 
    optim = torch.optim.Adam([tex_map], lr=lr)  

    model_name = model.split('_')[0]
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    name = f'XFL_{epochs}e_{model_name}_{epochs}_tex_{tex_mask}_'+date 
    log_dir_model = os.path.join(log_dir, name)
    if not os.path.exists(log_dir_model):
        os.mkdir(log_dir_model)

    net.eval()
    loggerf = open(log_dir_model + "_log.txt","w")

    thrs = np.array([0.5])
    p_res, ASR = evaluate(test_data_dir, test_pipeline, net, obj_file, tex_map, device, clean=True, thrs=thrs)
    print('clean:',p_res.coco_eval['bbox'].stats)
    PR= p_res.coco_eval['bbox'].stats
    loggerf.write('clean:'+ model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')

    for i_epoch in range(epochs):
        train_pbar = tqdm(train_dataloader)
        train_dataloader.dataset.set_maps(tex_map)
        for i, (total_img, imgs_pred, mask, img, npz_name) in enumerate(train_pbar): 
            data_test = train_dataloader.dataset.prepare_data((total_img*255).float(), test_pipeline)
            # modify two_stage.py and standard_roi_head.py (in mmdet/models/detectors, predict()) to get cls_scores
            # modify single_stage.py, yolo_head.py, base_dense_head.py to get cls_scores
            outputs, cls_scores = net.test_step(data_test) 
            predctions = outputs[0].pred_instances
            if len(cls_scores[0])>2000:
                cls_loss = cls_scores[0,:,2].max() * 0.1 # balance the losses
            else:        
                softmax = torch.softmax(cls_scores[0],dim=1)
                cls_loss = softmax[:,target_class].sum() * 0.01
            bboxes = predctions.bboxes[predctions.labels==target_class] # x1 y1 x2 y2
            if len(bboxes):
                box_loss = predctions.scores[predctions.labels==target_class].sum()
            else:
                box_score = 0
                continue
            # print()
            smt_loss = loss_smooth(tex_map.permute(0,3,1,2))
            if model == 'ssd512_coco':
                cls_loss = cls_scores[0][0,2].mean()+cls_scores[0][0,81+2].mean()+cls_scores[0][0,81*2+2].mean()+cls_scores[0][0,81*3+2].mean()
                loss = cls_loss + smt_loss
            else:
                loss = box_loss + cls_loss + smt_loss
            optim.zero_grad()
            loss.backward(retain_graph=True) # retain_graph=True
            optim.step()
            train_dataloader.dataset.set_maps(tex_map)
            train_pbar.set_description('box_loss %.4f cls_loss %.4f smt_loss %.4f' % (box_loss.data.cpu().numpy(), cls_loss.data.cpu().numpy(), smt_loss.data.cpu().numpy()))
        p_res, ASR = evaluate(test_data_dir, test_pipeline, net, obj_file, tex_map.detach(), device,clean=False, thrs=thrs)
        PR= p_res.coco_eval['bbox'].stats
        print('adv:'+ model_name + ' e:' + str(i_epoch)+' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
        # log and save
        loggerf.write('adv:'+ model_name + ' e:' + str(i_epoch)+' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
        np.save(os.path.join(log_dir_model, f'{model_name}__{i_epoch}_'+str(PR[0])[2:6]+'.npy'), tex_map.detach().cpu().numpy())

        map_ = train_dataloader.dataset.mesh.textures._maps_padded
        Image.fromarray((map_[0]*255).data.cpu().numpy().astype('uint8')).save(os.path.join(log_dir_model, f'{model_name}__{i_epoch}_'+str(PR[0])[2:6]+'.png'))

    # evaluate textures with all models
    if test_all_model:
        for test_model in test_models:
            test_model_name = test_model.split('_')[0]
            test_n = nets[test_model]
            thrs = np.array([0.5])
            p_res, ASR = evaluate(test_data_dir, test_pipeline, test_n, obj_file, tex_map.detach(), device,clean=False, thrs=thrs)
            PR= p_res.coco_eval['bbox'].stats
            print('adv:'+ model_name + ' test: '+ test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
            loggerf.write('adv:'+ model_name + ' test: '+ test_model_name +' ASR: ' +str(ASR)+ ' P: '+str(PR[0])+' R: '+str(PR[1])+'\n')
    loggerf.close()