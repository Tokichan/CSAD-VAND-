import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from einops import rearrange
from scipy.ndimage import binary_fill_holes
def split_masks_from_one_mask(masks):
    result_masks = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.001:
            
            result_masks.append(mask)
    return result_masks
class ComponentFeatureExtractor():
    """
    "com_config":{
            'reduce_feature': 'mean',
            'in_dim': 1024,
            'proj_dim': 8,
            'bins': 20,
            'dim_reduction': False,
            'normalize': True,
            'image_size': 256,
            'transform': transform,
        }
    """
    def __init__(self,transform,model=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transform
        if self.model is not None:
            self.model.to(self.device)

    def crop_by_mask(self,image,mask):
        if image.shape[-1] == 3:
            mask = cv2.merge([mask,mask,mask])
        return np.where(mask!=0,image,0)
    

    def align(self,masks,target_size,image):
        """
            resize object and record scale position
            target_size: Int
            output: List[cropped_masks,cropped_images,center_positions,scales]
        """
        cropped_masks = list()
        cropped_images = list()
        center_positions = list()
        scales = list()

        i=0
        for mask in masks:
            
            # crop image by cropped_mask
            croped_image = self.crop_by_mask(image, mask)

            cnt,_ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key = cv2.contourArea)

            # fill holes in the mask
            mask = cv2.drawContours(np.zeros_like(mask), [c], -1, (255,255,255), -1)

            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            p1,p2 = box[0], box[2]

            diagonal = np.linalg.norm((p1-p2),ord=2)
            scale = target_size/(diagonal+1)
            cropped_mask = mask[y:y+h, x:x+w]
            croped_image = croped_image[y:y+h, x:x+w]
            temp_size = (np.array(cropped_mask.shape) * scale).astype(np.int32)
            temp_size = np.where(temp_size>target_size,target_size,temp_size)
            temp_size = np.where(temp_size<=0,1,temp_size)

            cropped_mask = cv2.resize(cropped_mask,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            croped_image = cv2.resize(croped_image,(temp_size[1],temp_size[0]),cv2.INTER_LINEAR)
            cropped_mask[cropped_mask>128] = 255
            cropped_mask[cropped_mask<=128] = 0

            # padding
            padw = int((target_size - cropped_mask.shape[1])//2)
            padh = int((target_size - cropped_mask.shape[0])//2)
            cropped_mask = cv2.copyMakeBorder(cropped_mask,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))
            croped_image = cv2.copyMakeBorder(croped_image,padh,padh,padw,padw,cv2.BORDER_CONSTANT,value=(0))

            if cropped_mask.shape[0] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,1,0,0,0,cv2.BORDER_CONSTANT,value=(0))
            if cropped_mask.shape[1] != target_size:
                cropped_mask = cv2.copyMakeBorder(cropped_mask,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))
                croped_image = cv2.copyMakeBorder(croped_image,0,0,1,0,cv2.BORDER_CONSTANT,value=(0))

            #print(cropped_mask.shape)
            # normalized center position of the mask object
            # range:[0,1]
            center_position = [(x+w/2)/mask.shape[1], (y+h/2)/mask.shape[0]]
            
            cropped_masks.append(cropped_mask)
            cropped_images.append(croped_image)
            center_positions.append(center_position)
            scales.append(scale)

            # cv2.imwrite(f"/home/anomaly/DRAEM/pack/capsule/mask2/{i}.jpg",cropped_mask)
            i+=1
        return [cropped_masks,cropped_images,center_positions,scales]
    
    
    def rot_images(self,images,num_angles=20):
        images = [Image.fromarray(image) for image in images]
        result = list()
        for image in images:
            rotated_image = torch.stack([self.transform(image.rotate(angle,Image.Resampling.BILINEAR)) for angle in np.linspace(0,360,num_angles)])
            result.append(rotated_image)
        result = torch.cat(result,dim=0)
        return result
    
    def fill_holes(self,masks):
        result = list()
        for mask in masks:
            mask = binary_fill_holes(np.where(mask!=0,1,0))
            mask = np.where(mask!=0,255,0)
            mask = mask.astype(np.uint8)
            result.append(mask)
        return result


        
    def compute_component_feature(self,image,masks):
        """
        image: [256,256,3] contains a component, value range [0,255]
        mask: [N,256,256] contains a component mask, value range [0,255]
        output: List[feature_dict]
        """
        # area feature (from ComAD)
        component_features = list()
        masks = split_masks_from_one_mask(masks)
        aligned_info = self.align(masks=masks,target_size=65,image=image)
        # scaled_aligned_info = self.align_with_relative_scale(masks=masks,target_size=256,image=image)
        # scaled_aligned_images = scaled_aligned_info[1]
        # aligned_masks = self.fill_holes(aligned_info[0])
        aligned_masks = aligned_info[0]
        aligned_images = aligned_info[1]



        # CNN feature
        # crop image and resize to 64x64
        num_rotation = 60
        
        if self.model is not None:
            rotated_images = self.rot_images(aligned_images,num_angles=num_rotation)
            with torch.no_grad():
                image_outputs = self.model(rotated_images.to(self.device))[2]

            del rotated_images

            image_outputs = rearrange(image_outputs,'(N R) C H W -> N R C H W',R=num_rotation)
            image_outputs = torch.mean(image_outputs,dim=[3,4],keepdim=False) # average over spatial dimension
            image_outputs = torch.mean(image_outputs,dim=1,keepdim=False) # average over rotation
        return image_outputs
    
    def concat_component_feature(self,component_features):
        """
        result: List[component1_feature,component2_feature,...]
        """
        result = dict()
        for feature in ['area','color','position','cnn_image']: #,'cnn_shape'
            result[feature] = list()
            for i in range(len(component_features)):
                result[feature].append(component_features[i][feature])
            result[feature] = np.array(result[feature])


        # for i in range(len(component_features)):
        #     result.append(np.concatenate([component_features[i][f] for f in feature_list],axis=0))
        return result
    
    def extract(self,image,masks):
        """
        image: [256,256,3] contains a component, value range [0,255]
        mask: [N,256,256] contains a component mask, value range [0,255]
        output: List[feature_dict]
        """
        component_features = self.compute_component_feature(image,masks)
        component_features = self.concat_component_feature(component_features)
        return component_features

