import os
import re
import torch
import numpy as np

from math import sqrt
from torch.utils.data import Dataset
from torch.utils.data import Dataset,DataLoader

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# reproduce testpipeline with the format torch tensor 
import torchvision.transforms as transforms

# Data structures and functions for rendering
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

# dataset returning rendered images as items.import sys
import torch
from math import sqrt
from tools.pre_processor import LoadImageFromTensor, Tensor_Resize, LoadAnnotations_Tensor,PackDet_Tensor
import torchvision.transforms as transforms

# dataset returning rendered images as items.

class MMdetAdv(Dataset):
    '''
    dataset to get rendered images:
    --datadir: train data path, contain 'mask', 'npz', ('images' and 'labels').

    --device
    --ret_mask: whether to return mask
    '''
    def __init__(self, obj_file, data_dir, device, tex_mask=True, ret_mask=True):
        self.data_dir = data_dir
        self.npz_dir = os.path.join(data_dir, 'com_npz')
        self.device = device
        self.ret_mask = ret_mask
        self.tex_mask = tex_mask
        # rendered image mask
        self.npz_files = sorted(os.listdir(self.npz_dir), key=lambda x: int(re.findall("\d+", x)[0]))
        # maps = cv2.imread(map_png) / 255.0
        # self.map_png = torch.from_numpy(np.transpose(maps, (2, 0, 1))).unsqueeze(0).to(device)
        # Load obj file
        mesh = load_objs_as_meshes([obj_file], device=self.device)
        verts = mesh.verts_packed()
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)));
        self.mesh = mesh

        # texture map 
        ori_maps = mesh.textures._maps_padded 
        self.map_mask = torch.where(ori_maps == 1, 1, 0)                
        self.map_png = ori_maps * (1 - self.map_mask)

        # random disturb
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

    def set_maps(self, maps):
        self.mesh.textures._maps_padded = (self.map_mask * maps.clamp(0.0001,0.9999) + self.map_png)

    def __getitem__(self, index):
        # load npz (after processing old npz to new one with mask and label)
        npz_name = self.npz_files[index]
        file = os.path.join(self.npz_dir, npz_name)
        data = np.load(file, allow_pickle=True)
        img, mask_tex, mask_car, cam_trans, label = data['data'], data['mask_tex'], data['mask_car'], data['cam_trans'], data['label']

        # load mask(not necessary if 3D car needed)
        mask = mask_tex if self.tex_mask else mask_car
        mask = mask / 255.0   # 0~1 [h,w,c]->[c,w,h]
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(self.device).unsqueeze(0)  #[1,c,h,w]
        
        # process the original image with the background
        img = img[:, :, ::-1]  # to RGB
        img = np.transpose(img, (2, 0, 1)) / 255.0  # 0~1 [h,w,c]->[c,h,w]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)  # 0~1 [1,c,h,w]
        
        # modify the location of the camera & car(differet pre in pytorch3d & carla)
        scale = 0.345  # depends on the size of your car model(obj)
        dist = sqrt(sum(cam_trans[0, :]**2)) * scale
        elev = -cam_trans[1, 0]-0.001  # avoid the upside down at 90°
        azim = -cam_trans[1, 1]
        car_at = (0, -.235, 0.07)
        
        # render
        R, T = look_at_view_transform(
                                      dist = dist,
                                      elev = elev,  # 0~180
                                      azim = azim,  # -180~180
                                      at = (car_at,),  # (1, 3)
                                      up = ((0, 1, 0),),  # (1, 3)
                                      device = self.device) 
        camera = FoVPerspectiveCameras(device=self.device, fov=60, R=R, T=T)
        lights = DirectionalLights(device=self.device, 
                                   ambient_color=((0.7, 0.7, 0.7),),
                                   diffuse_color=((0.4, 0.4, 0.4),),
                                   specular_color=((0.2, 0.2, 0.2),),
                                   direction=((1, 1, 1),))
        raster_settings = RasterizationSettings(image_size=img.shape[-1], blur_radius=0.0, faces_per_pixel=1)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=camera, lights=lights)
        )
        imgs_pred = renderer(self.mesh)  # 0~1 [1,h,w,c]
        imgs_pred = self.eot(imgs_pred)
        total_img = (1 - mask) * img + mask * imgs_pred
       
        if self.ret_mask:
            return total_img.squeeze(0), imgs_pred.squeeze(0), mask.squeeze(0), img.squeeze(0), npz_name
        else:
            return total_img.squeeze(0), imgs_pred.squeeze(0), img.squeeze(0), npz_name

    def eot(self, imgs_pred):
        imgs_pred = torch.clamp(imgs_pred[0, ..., :3], 0, 1).permute((2,0,1)).unsqueeze(0)  #[1,c,h,w]
        # contrast
        contrast = imgs_pred.new(imgs_pred.shape).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.to(imgs_pred)
        # brightness
        brightness = imgs_pred.new(imgs_pred.shape).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.to(imgs_pred)
        # noise
        noise = imgs_pred.new(imgs_pred.shape).uniform_(-1, 1) * self.noise_factor
        imgs_pred = imgs_pred * contrast + brightness + noise
        imgs_pred = torch.clamp(imgs_pred, 0.000001, 0.99999)
        return imgs_pred
        
    def prepare_data(self, img, tran):
        # img = im_bgr[:, :, ::-1]  # to RGB
        # image = torch.tensor(img.copy()).permute(2,0,1).unsqueeze(0).to(device)/255.0
        res={}
        res['img']=img
        data_new = tran(res)
        data_new['data_samples'] = [data_new['data_samples']]
        return data_new
    
    def __len__(self):
        # return 600
        return len(self.npz_files)
    
