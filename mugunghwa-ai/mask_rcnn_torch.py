from PIL import Image
from torchvision.io import write_png
from torchvision.models.detection import maskrcnn_resnet50_fpn as mrcnn
from typing import Tuple
import cv2
import numpy as np
import time
import torch
from convert import *

class Segmenter(object):
    def __init__(self):
        self.net = mrcnn(pretrained=True)
        self.net.eval()

        self.is_gpu = torch.cuda.device_count() > 0
        if self.is_gpu:
            self.net.cuda()
    
    def set_img(self, src, img_size: Tuple[int, int]=(400, 400)):
        self.img = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), img_size)
        self.img = torch.from_numpy(self.img.transpose(2, 0, 1)) / 255
        if self.is_gpu:
            self.img = self.img.cuda()

    def read_img(self, img_fp:str, img_size: Tuple[int, int]=(400, 400)):
        self.img = cv2.resize(cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB), img_size)
        self.img = torch.from_numpy(self.img.transpose(2, 0, 1)) / 255
        if self.is_gpu:
            self.img = self.img.cuda()
        
    def get_instance_mask_list(self):
        # instance_mask_list is a list of binary mask for each instance
        # Each instance binary mask has value of 0 or 1
        x = torch.stack([self.img])
        out = self.net(x)[0]
        person_mask = out['labels'] == 1
        score_mask = out['scores'] > 0.90
        comb_mask = person_mask & score_mask

        labels = out['labels'][comb_mask].cpu().detach().numpy()
        masks = out['masks'][comb_mask].cpu().detach().numpy()
        
        self.labels = np.array(list(range(len(labels)))) # [0, 1, 2, ...]
        self.instance_mask_list = [CHW2HWC(np.array(masks[i] > 0.5, dtype=np.int32)) for i in self.labels]
        
        return self.instance_mask_list
    
    def get_instance_mask_combined(self):
        # instance_mask_combined is a single instance mask
        # Each pixel posseses one of the following values:
        # 0(background), 1(instance 1), 2(instance 2), etc.
        self.instance_mask_combined = np.zeros([*self.img.shape[1:], 1], dtype=np.int32)
        instance_mask_list = self.get_instance_mask_list()
        for instance_num, instance_mask in enumerate(instance_mask_list):
            instance_num += 1
            instance_mask = instance_mask * instance_num
            self.instance_mask_combined = np.clip(self.instance_mask_combined + instance_mask, 0, instance_num, dtype=np.int32)
        return self.instance_mask_combined, self.instance_mask_list
    
    def save_mask(self):
        write_png((self.img.cpu().detach() * 255).type(torch.uint8), "original.png")
        for i, mask in enumerate(self.instance_mask_combined):
            write_png(torch.from_numpy((self.img.cpu().detach().numpy() * mask * 255).astype(np.int32)), f"{i}.png")

    def find_containing_mask(self, pt_coord):
        mask_idx = [idx for idx in range(len(self.instance_mask_combined)) if self.instance_mask_combined[idx][0][pt_coord]]
        if len(mask_idx) == 0:
            return -1
        else:
            return mask_idx[0]

    def find_instance_faces(self, instance_idx_list):
        faces = []
        src_img = self.img.detach().cpu().numpy().transpose(1, 2, 0)

        for instance_idx in instance_idx_list:
            if instance_idx == 0:
                continue

            # Mask current instance
            instance_mask = (self.instance_mask_combined == instance_idx).transpose(1, 2, 0)
            masked_img = src_img * instance_mask * 255

            # Extract face from the masked image
            self.face.img = Image.fromarray(masked_img.astype(np.uint8))
            # box, _ = self.face.detection()
            face_list = self.face.update_face_log(thres=1.2)
            faces = [np.array(f) for f in face_list]

        return faces

    # def find_containing_face(self, pt_coord):
    #     mask_idx = self.find_containing_mask(pt_coord[::-1])

    #     if mask_idx < 0:
    #         print("Could not find face")
    #         return None
    #     mask = self.masks[mask_idx]
    #     img = self.img.cpu().numpy().transpose(1, 2, 0)
    #     masked_img = img * mask.transpose(1, 2, 0) * 255
    #     masked_img = Image.fromarray(masked_img.astype(np.uint8))
        
    #     from face_detection import FaceNet
    #     face = FaceNet()
    #     face.img = masked_img
    #     box, _ = face.detection()
    #     if box is None:
    #         return None
    #     box_coord = tuple(box[0].tolist())
    #     crop_face = Image.fromaray((img * 255).astype(np.uint8)).crop(box_coord)
    #     return crop_face

if __name__ == "__main__":
    seg = Segmenter()
    seg.read_img('img1.jpg')
    instance_mask_combined, instance_mask_list = seg.get_instance_mask_combined()
    Image.fromarray((instance_mask_combined[...,0]*255/len(instance_mask_list)).astype(np.uint8)).show()
