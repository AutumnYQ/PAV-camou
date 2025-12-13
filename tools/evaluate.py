import re
import torch
import torchvision
from pycocotools.coco import COCO
from tools.coco_eval.coco_eval import CocoEvaluator

from torch.utils.data import Dataset
from torch.utils.data import Dataset,DataLoader

from .dataset import MMdetAdv
import numpy as np
from tqdm import tqdm

def evaluate(test_data_dir, test_pipeline, net, obj_file, tex_map, device, clean=False, thrs=None, conf_thr=0.6):
    thrs = np.array([0.5]) if thrs is None else thrs

    COCO_label = test_data_dir+'/val.json'
    batch_size = 1
    COCO_gt = COCO(annotation_file=COCO_label)
    coco_evaluator = CocoEvaluator(COCO_gt, ["bbox"], thrs)
    # print(coco_evaluator.coco_eval['bbox'].params.iouThrs)
    test_dataset = MMdetAdv(obj_file, test_data_dir, device, tex_mask=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)  # dataloader
    test_dataset.set_maps(tex_map)
    test_pbar = tqdm(test_dataloader)
    # adv
    num_success = 0
    for i, (total_img, imgs_pred, mask, img, npz_name) in enumerate(test_pbar): 
        if not clean:
            data = total_img
        else:
            data = img
        data_test = test_dataset.prepare_data(data*255, test_pipeline)
        outputs, _ = net.test_step(data_test) 
        predctions = outputs[0].pred_instances

        image_id = int(re.findall("\d+", npz_name[0])[0])
        bboxes = predctions.bboxes[predctions.labels==2] # x1 y1 x2 y2
        scores = predctions.scores[predctions.labels==2]
        if len(bboxes):
            area = predctions.bboxes[predctions.labels==2][:,-1]*predctions.bboxes[predctions.labels==2][:,-2]
            labels = predctions.labels[predctions.labels==2]
            # if conf_thr is not None:
            #     bboxes = bboxes[scores>conf_thr]
            #     labels = labels[scores>conf_thr]
            #     scores = scores[scores>conf_thr]
            if (scores>conf_thr).sum()>0:
                num_success += 1
        else: 
            area = 0
            labels = torch.tensor([])
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)
        output = dict(boxes=bboxes, labels=labels, area=area, scores=scores, iscrowd=iscrowd)
        dt = {image_id: output}
        coco_evaluator.update(dt)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    ASR = 1 - num_success/len(test_dataset)
    return coco_evaluator, ASR