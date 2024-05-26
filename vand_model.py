import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)
from build_sam_hq import sam_hq_model_registry
# Grounding DINO
from grounded_sam import (
    load_image,
    load_model,
    get_grounding_output
)

import tqdm
import os
import glob
import random
from component_feature_extractor import ComponentFeatureExtractor
import sklearn.cluster

import scipy.stats
import torch
from PIL import Image
import yaml
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
import timm
from model import Encoder, ResNetTeacher, Segmentor, get_ae
from pseudo_label import grounding_segmentation, segmentation
import itertools
from scipy.spatial.distance import mahalanobis

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

class HistLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,pred, trg):
        pred = torch.softmax(pred,dim=1)
        new_trg = torch.zeros_like(trg).repeat(1, pred.shape[1], 1, 1).long()
        new_trg = new_trg.scatter(1, trg, 1).float()
        diff = torch.abs(new_trg.mean((2, 3)) - pred.mean((2, 3)))
        if self.ignore_index is not None:
            diff = torch.concat([diff[:,:self.ignore_index],diff[:,self.ignore_index+1:]],dim=1)
        loss = diff.sum() / pred.shape[0]  # exclude BG
        return loss


class ClassBalancedDiceLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        probabilities = torch.softmax(prediction,dim=1)
        targets_one_hot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=prediction.shape[1])
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)


        class_weights = self._calculate_class_weights(targets_one_hot)
        dice_loss = self._dice_loss(probabilities, targets_one_hot)
        class_balanced_loss = class_weights * dice_loss
        return class_balanced_loss.mean()

    def _calculate_class_weights(self, target):
        """
        Calculates class weights based on their inverse frequency in the target.
        """
        weights = torch.zeros((target.shape[0],target.shape[1])).cuda()
        for c in range(target.shape[1]):
            weights[:,c] = 1 / (target[:,c].sum() + 1e-5)
        weights = weights / weights.sum(dim=1,keepdim=True)
        return weights.detach()

    def _dice_loss(self, prediction, target):
        """
        Calculates dice loss for each class and then averages across all classes.
        """
        intersection = 2 * (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5
        dice = (intersection + 1e-5) / (union + 1e-5)
        return 1 - dice
    
class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probabilities = logits

        probabilities = nn.Softmax(dim=1)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes=logits.shape[1])
        # print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).type(torch.float)

        if self.ignore_index is not None:
            targets_one_hot = torch.concat([targets_one_hot[:,:self.ignore_index],targets_one_hot[:,self.ignore_index+1:]],dim=1)
            probabilities = torch.concat([probabilities[:,:self.ignore_index],probabilities[:,self.ignore_index+1:]],dim=1)
        
        return self.ce_loss(probabilities,targets_one_hot)
    
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, ignore_index=None, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.ignore_index = ignore_index

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        logit = torch.softmax(logit, dim=1)
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.ignore_index is not None:
            one_hot_key = torch.concat([one_hot_key[:,:self.ignore_index],one_hot_key[:,self.ignore_index+1:]],dim=1)
            logit = torch.concat([logit[:,:self.ignore_index],logit[:,self.ignore_index+1:]],dim=1)


        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss

class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred):
        prob = nn.Softmax(dim=1)(pred)
        return (-1*prob*((prob+1e-5).log())).mean()
    
def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class CSAD(nn.Module):
    def __init__(self):
        super(CSAD, self).__init__()
        self.proj_root = os.path.dirname(os.path.abspath(__file__))

        # download pretrained model
        sam_save_path = os.path.join(self.proj_root,'sam_hq_vit_h.pth')
        if not os.path.exists(sam_save_path):
            os.system(f"wget --no-check-certificate 'https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing' -O sam_hq_vit_h.pth")
        dino_save_path = os.path.join(self.proj_root,'groundingdino_swint_ogc.pth')
        if not os.path.exists(dino_save_path):
            os.system(f"wget --no-check-certificate 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth' -O groundingdino_swint_ogc.pth")
        


        

    def setup(self,dataset_data):
        few_shot_images = dataset_data["few_shot_samples"]
        dataset_category = dataset_data["dataset_category"]
        self.images = few_shot_images
        self.dataset_category = dataset_category
        self.norm_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.hist_config = read_config(os.path.join(self.proj_root,f'configs/class_histogram/{self.dataset_category}.yaml'))
        self.seg_config = read_config(os.path.join(self.proj_root,f'configs/segmentor/{self.dataset_category}.yaml'))
        self.encoder = timm.create_model('wide_resnet50_2.tv2_in1k'
                                          ,pretrained=True,
                                          features_only=True,
                                          out_indices=[1,2,3]).cuda().eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # seg
        self.num_classes = self.generate_pseudo_labels(self.images)
        self.segmentor = Segmentor(self.num_classes).cuda().train()
        self.train_segmentation()
        self.cal_hists(self.images)

        # LGST
        self.teacher = ResNetTeacher().cuda().eval()
        self.st = Encoder().cuda().train()
        self.ae = get_ae().cuda().train()
        self.train_LGST()

        self.cal_val()


    def cal_val(self):
        hist_scores = []
        patch_hist_scores = []
        LGST_scores = []
        for i in range(len(self.images)):
            image = torch.unsqueeze(self.images[i],dim=0).cuda()
            image = self.norm_transform(image)
            hist_score,patch_hist_score,LGST_score,_ = self.predict(image)
            hist_scores.append(hist_score)
            patch_hist_scores.append(patch_hist_score)
            LGST_scores.append(LGST_score)
        self.hist_score_mean = np.mean(hist_scores)
        self.hist_score_std = np.std(hist_scores)
        self.patch_hist_score_mean = np.mean(patch_hist_scores)
        self.patch_hist_score_std = np.std(patch_hist_scores)
        self.LGST_score_mean = np.mean(LGST_scores)
        self.LGST_score_std = np.std(LGST_scores)

    def histogram(self,label_map,num_classes):
        hist = np.zeros(num_classes)
        for i in range(1,num_classes+1): # not include background
            hist[i-1] = (label_map == i).sum()
        hist = hist / label_map.size
        return hist 
    
    def patch_histogram(self,label_map, num_classes):
        h,w = label_map.shape
        p1 = label_map[0:h//2,0:w//2]
        p2 = label_map[0:h//2,w//2:w]
        p3 = label_map[h//2:h,0:w//2]
        p4 = label_map[h//2:h,w//2:w]
        hists = [self.histogram(p,num_classes) for p in [p1,p2,p3,p4]]
        hists = np.hstack(hists)
        return hists
    
    @torch.no_grad()
    def cal_hists(self,images):
        self.segmentor.eval()
        hists = []
        patch_hists = []
        for i in range(len(images)):
            image = torch.unsqueeze(images[i],dim=0).cuda()
            image = self.norm_transform(image)
            image = self.encoder(image)
            pred = self.segmentor(image)
            pred = torch.argmax(pred,dim=1).cpu().numpy()[0]
            hist = self.histogram(pred,self.num_classes)
            patch_hist = self.patch_histogram(pred,self.num_classes)
            hists.append(hist)
            patch_hists.append(patch_hist)
        hists = np.vstack(hists)
        self.hist_mean = np.mean(hists,axis=0)
        self.hist_cov = np.linalg.pinv(np.cov(hists.T))
        patch_hists = np.vstack(patch_hists)
        self.patch_hist_mean = np.mean(patch_hists,axis=0)
        self.patch_hist_cov = np.linalg.pinv(np.cov(patch_hists.T))

    @torch.no_grad()
    def predict(self,image):
        self.segmentor.eval()
        self.st.eval()
        self.ae.eval()
        self.teacher.eval()
        self.encoder.eval()
        feat = self.encoder(image)
        pred = self.segmentor(feat)
        pred = torch.argmax(pred,dim=1).cpu().numpy()[0]
        hist = self.histogram(pred,self.num_classes)
        hist_score = mahalanobis(hist,self.hist_mean,self.hist_cov)
        patch_hist = self.patch_histogram(pred,self.num_classes)
        patch_hist_score = mahalanobis(patch_hist,self.patch_hist_mean,self.patch_hist_cov)
        teacher_out = self.teacher(feat)
        st_out,stae_out = self.st(image)
        ae_out = self.ae(image)
        map_st = torch.mean((teacher_out - st_out)**2,dim=1)
        map_stae = torch.mean((ae_out - stae_out)**2,dim=1)
        map_combine = map_st + map_stae
        LGST_score = torch.max(map_combine).cpu().detach().numpy()
        return hist_score,patch_hist_score,LGST_score,map_combine.squeeze().cpu().detach().numpy()


    def train_LGST(self):
        self.optimizer = torch.optim.Adam(itertools.chain(self.st.parameters(),
                                                    self.ae.parameters()),
                                    lr=1e-4, weight_decay=1e-5)
        self.teacher = self.teacher.cuda()
        self.st = self.st.cuda()
        self.ae = self.ae.cuda()
        self.st.train()
        self.ae.train()
        self.teacher.eval()
        self.encoder = self.encoder.cuda()
        self.encoder.eval()
        for i in range(10000):
            image = self.images[i%len(self.images)]
            image = torch.unsqueeze(image,dim=0).cuda()
            image = self.norm_transform(image)
            loss = self.LGST_train_one_step(image)
            if i%1000==0:
                print(f"step:{i},loss:{loss}")
        pass

    def LGST_train_one_step(self,image):
        teacher_out = self.teacher(self.encoder(image))
        st_out,stae_out = self.st(image)
        ae_out = self.ae(image)

        st_diff = (st_out - teacher_out)**2
        q = torch.quantile(st_diff,0.999)
        st_loss = torch.mean(st_diff[st_diff>q])
        stae_loss = F.mse_loss(stae_out,ae_out)
        ae_loss = F.mse_loss(ae_out,teacher_out)

        loss = st_loss + stae_loss + ae_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train_segmentation(self):

        self.optimizer = torch.optim.Adam(self.segmentor.parameters(),lr=self.seg_config["lr"])
        self.ce_loss = MulticlassCrossEntropyLoss(ignore_index=None)
        self.focal_loss = FocalLoss(ignore_index=None)
        self.dice_loss = ClassBalancedDiceLoss(ignore_index=None)
        self.hist_loss = HistLoss(ignore_index=0)
        self.entropy_loss = EntropyLoss()

        self.loss_dict = {
            "ce":self.ce_loss,
            "focal":self.focal_loss,
            "dice":self.dice_loss,
            "hist":self.hist_loss,
            "entropy":self.entropy_loss
        }
        self.loss_weight = self.seg_config["loss_weight"]
        self.segmentor = self.segmentor.cuda()
        self.segmentor.train()
        
        for i in range(10000):
            image = self.images[i%len(self.images)]
            image = torch.unsqueeze(image,dim=0).cuda()
            image = self.norm_transform(image)
            gt = self.pseudo_labels[i%len(self.images)]
            gt = torch.tensor(gt).long()
            gt = torch.unsqueeze(gt,dim=0).cuda()
            loss = self.seg_train_one_step(image,gt)
            if i%1000==0:
                print(f"step:{i},loss:{loss}")
            # print(f"step:{i},loss:{loss}")

        ## vis
        self.segmentor.eval()
        for i in range(len(self.images)):
            image = self.images[i]
            gt = self.pseudo_labels[i]
            with torch.no_grad():
                image = torch.unsqueeze(image,dim=0).cuda()
                image = self.norm_transform(image)
                image = self.encoder(image)
                pred = self.segmentor(image)
                pred = torch.argmax(pred,dim=1).squeeze().cpu().numpy()
                color_pred = np.zeros((pred.shape[0],pred.shape[1],3))
                color_gt = np.zeros((gt.shape[0],gt.shape[1],3))
                for j in range(self.num_classes):
                    color = np.random.rand(3)
                    color_pred[pred==j] = color
                    color_gt[gt==j] = color

                color_pred = (color_pred*255).astype(np.uint8)
                color_gt = (color_gt*255).astype(np.uint8)
                cv2.imshow('gt',color_gt)
                cv2.imshow('pred',color_pred)
                cv2.waitKey(0)


    
    def seg_train_one_step(self,image,gt):
    
        with torch.no_grad():
            image = self.encoder(torch.concat([image,image],dim=0))

        sup_out = self.segmentor(image)[0,:,:,:]
        sup_out = torch.unsqueeze(sup_out,dim=0)
        sup_ce = self.loss_dict["ce"](sup_out,gt) * self.loss_weight['ce']
        sup_focal = self.loss_dict["focal"](sup_out,gt) * self.loss_weight['focal']*0
        sup_dice = self.loss_dict["dice"](sup_out,gt) * self.loss_weight['dice']*0
        sup_entro = self.loss_dict["entropy"](sup_out) * self.loss_weight['entropy']*0
        total_loss = sup_ce + sup_focal + sup_dice + sup_entro

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
    



    def generate_pseudo_labels(self,images):
        # load grounded sam model
        sam = sam_hq_model_registry['vit_h'](checkpoint=os.path.join(self.proj_root,'sam_hq_vit_h.pth')).cuda()
        grounding_dino =  load_model(os.path.join(self.proj_root,'groundingdino/config/GroundingDINO_SwinT_OGC.py'),
                       os.path.join(self.proj_root,'groundingdino_swint_ogc.pth'),"cuda")
        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam,
                                                points_per_side=32,
                                                points_per_batch=16,
                                                pred_iou_thresh=0.88,
                                                stability_score_thresh=0.9,# 0.97
                                                stability_score_offset=1.0,#1.0,
                                                box_nms_thresh=0.7,
                                                crop_n_layers=1,
                                                crop_nms_thresh=0.7,
                                                crop_overlap_ratio=512 / 1500,
                                                crop_n_points_downscale_factor=1,
                                                point_grids=None,
                                                min_mask_region_area=500,
                                                output_mode="binary_mask",
                                                )
        refined_masks = []
        for image in images:
            # grounded sam
            grounding_result = grounding_segmentation(image,grounding_dino,predictor,self.hist_config['grounding_config'])
            # sam
            image = (image*255).permute(1,2,0).cpu().detach().numpy().astype(np.uint8)
            refined_mask = segmentation(image,mask_generator,grounding_result,self.hist_config)
            refined_masks.append(refined_mask)

        # component feature extraction
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.norm_transform
        ])
        numpy_images = [(image*255).permute(1,2,0).cpu().detach().numpy().astype(np.uint8) for image in images]
        component_feature_extractor = ComponentFeatureExtractor(transform,model=self.encoder)
        component_features = []
        for image,refined_mask in zip(numpy_images,refined_masks):
            component_feature = component_feature_extractor.compute_component_feature(image,refined_mask)
            print(f"number of components:{len(component_feature)} component_num:{np.max(refined_mask)}")
            component_features.append(component_feature)
        component_features = torch.concat(component_features)
        print()

        self.projector = torch.nn.Linear(1024,
                                   512)
        self.projector.bias.data.zero_()
        self.projector.weight.data.normal_(mean=0.0, std=0.01)
        self.projector = self.projector.cuda()
        component_features = self.projector(component_features)
        component_features = component_features.detach().cpu().numpy()

        # clustering
        hyper = self.hist_config['max_hyper']
        self.cluster_model = sklearn.cluster.MeanShift(
                                                        bandwidth=hyper,
                                                        cluster_all=True,
                                                    ).fit(component_features)
        
        self.component_labels = self.cluster_model.labels_
        # remove small clusters
        # for i in range(np.max(self.component_labels)+1):
        #     class_num = np.sum(self.component_labels==i)
        #     if class_num < int(len(self.image_paths)*0.5):
        #         self.component_labels[self.component_labels==i] = -1
        print(f"number of clusters:{np.max(self.component_labels)+1} ,hyper={hyper}")
        print()

        self.pseudo_labels = []
        com_num = 0
        for refined_mask in refined_masks:
            pseudo_label = np.zeros_like(refined_mask)
            cur_num = np.max(refined_mask)
            labels = self.component_labels[com_num:com_num+cur_num]
            com_num += cur_num
            for i in range(cur_num):
                if labels[i] != -1:
                    pseudo_label[refined_mask==(i+1)] = labels[i]+1
            self.pseudo_labels.append(pseudo_label)
        return np.max(self.component_labels)+2

    def forward(self, images):
        hist_scores = []
        patch_hist_scores = []
        LGST_scores = []
        combined_maps = []
        for i in range(len(images)):
            image = torch.unsqueeze(images[i],dim=0).cuda()
            image = self.norm_transform(image)
            hist_score,patch_hist_score,LGST_score,combined_map = self.predict(image)
            hist_scores.append(hist_score)
            patch_hist_scores.append(patch_hist_score)
            LGST_scores.append(LGST_score)
            combined_maps.append(combined_map)

        hist_scores = (hist_scores - self.hist_score_mean) / self.hist_score_std
        patch_hist_scores = (patch_hist_scores - self.patch_hist_score_mean) / self.patch_hist_score_std
        LGST_scores = (LGST_scores - self.LGST_score_mean) / self.LGST_score_std

        hist_scores = np.array(hist_scores)
        patch_hist_scores = np.array(patch_hist_scores)
        LGST_scores = np.array(LGST_scores)
        combined_maps = np.array(combined_maps)


        hist_scores = torch.from_numpy(hist_scores)
        patch_hist_scores = torch.from_numpy(patch_hist_scores)
        LGST_scores = torch.from_numpy(LGST_scores)

        pred_score = hist_scores + patch_hist_scores + LGST_scores*2
        anomaly_maps = torch.from_numpy(combined_maps)
        anomaly_maps = F.interpolate(anomaly_maps.unsqueeze(1),size=(256,256),mode='bilinear',align_corners=False).squeeze(1)

        return {"pred_score": pred_score, "anomaly_map": anomaly_maps}

if __name__ == '__main__':
    csad = CSAD()
    dataset_category = "pushpins"
    image_paths = glob.glob(f'C:/Users/kev30/Desktop/anomaly/EfficientAD-res/datasets/mvtec_loco_anomaly_detection/{dataset_category}/train/good/*.png')[1:5]
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    images = torch.stack([transform(Image.open(path).convert('RGB')) for path in image_paths])
    csad.setup({"few_shot_samples":images,"dataset_category":dataset_category})
    result = csad(images)
    print(result)
    
