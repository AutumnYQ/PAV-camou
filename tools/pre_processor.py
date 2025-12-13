import random
import os
import sys
import torch
import numpy as np 
sys.path.append('/home/liuyuqiu/ssd3/mmdetection')

import torchvision.transforms as transforms
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
# from mmdet.structures.bbox.box_type import autocast_box_type
from mmcv.transforms import to_tensor
from mmcv.image.geometric import _scale_size

from mmdet.structures.bbox import BaseBoxes
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmengine.structures import InstanceData, PixelData

from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from typing import List, Optional, Sequence, Tuple, Union
from mmcv.transforms import Resize as MMCV_Resize


@TRANSFORMS.register_module()
class LoadImageFromTensor(LoadImageFromFile):
    def transform(self, results: dict) -> dict:
        img = results['img']
        # if self.to_float32:
        #     img = img.astype(np.float32)
        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[-2:]
        results['ori_shape'] = img.shape[-2:]
        return results


@TRANSFORMS.register_module()
class Tensor_Resize(MMCV_Resize):

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            
            results['gt_bboxes'] = results['gt_bboxes'][-2]/results['scale_factor'][0]
            results['gt_bboxes'] = results['gt_bboxes'][-1]/results['scale_factor'][1]
            # results['gt_bboxes'].rescale_(results['scale_factor'])
            # if self.clip_object_border:
            #     results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']
    
    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get('img', None) is not None:           
            img_shape = results['img'].shape[-2:]
            if self.keep_ratio:
                # calculate new h w
                new_h = results['scale']
                new_w = int(results['scale']/img_shape[-2]*img_shape[-1])
                h, w = results['img'].shape[-2:]
                w_scale = new_w / w
                h_scale = new_h / h                
                resize_ = transforms.Resize((new_h,new_w))
            else:
                resize_ = transforms.Resize(results['scale'])
                new_s = torch.tensor(results['scale'])/torch.tensor(img_shape)
                h_scale, w_scale = new_s[0], new_s[1]
                img = resize_(results['img'])

            results['img'] = img
            results['img_shape'] = img_shape
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        # if self.scale:
        #     img_shape = results['img'].shape[-2:]
        #     # results['scale'] = self.scale
        #     # scale_fac = torch.tensor(self.scale)/img_shape
        #     # results['scale_factor'] = 
        # else:
        #     img_shape = results['img'].shape[-2:]
        #     results['scale'] = _scale_size(img_shape, self.scale_factor)
        results['scale'] = self.scale[1], self.scale[0]
        self._resize_img(results)
        self._resize_bboxes(results)
        # self._record_homography_matrix(results)
        return results

@TRANSFORMS.register_module()
class LoadAnnotations_Tensor(MMCV_LoadAnnotations):
    def __init__(
            self,
            box_type: str = 'hbox',
            # use for semseg
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            **kwargs) -> None:
        super(LoadAnnotations_Tensor, self).__init__(**kwargs)
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)
        
    def transform(self, results: dict) -> dict:
        # load annotation
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        return results


@TRANSFORMS.register_module()
class PackDet_Tensor(BaseTransform):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
       
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            img = img.contiguous()
            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = torch.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = torch.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] =results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results


