from PIL import Image
from torchvision.io import write_png
from torchvision.models.detection import maskrcnn_resnet50_fpn as mrcnn
from typing import Tuple
import cv2
import numpy as np
import time
import torch

class Segmenter(object):
    def __init__(self):
        self.is_gpu = torch.cuda.device_count() > 0
        self.net = mrcnn(pretrained=True)
        self.net.eval()
        if self.is_gpu:
            self.net.cuda()
    
    def read_img(self, img_fp:str, img_size: Tuple[int, int]=(400, 400)):
        self.img = cv2.resize(cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB), img_size)
        self.img = torch.from_numpy(self.img.transpose(2, 0, 1)) / 255
        if self.is_gpu:
            self.img = self.img.cuda()
    
    def set_image_size(self, img_size):
        self.img_size = img_size
        self.width, self.height = img_size
        
    def get_mask(self):
        x = torch.stack([self.img])
        out = self.net(x)[0]
        person_mask = out['labels'] == 1
        self.scores = out['scores'][person_mask].cpu().detach().numpy()
        self.masks = out['masks'][person_mask].cpu().detach().numpy()[self.scores > 0.90]
        return self.masks
    
    def save_mask(self):
        write_png((self.img.cpu().detach() * 255).type(torch.uint8), "original.png")
        for i, mask in enumerate(self.masks):
            write_png(torch.from_numpy((self.img.cpu().detach().numpy() * mask * 255).astype(np.uint8)), f"{i}.png")

    def find_containing_mask(self, pt_coord):
        mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][0][pt_coord]]
        
        if len(mask_idx) == 0:
            return Exception('Containing mask not found')
        else:
            mask_idx = mask_idx[0]
        
        return mask_idx

    def find_containing_face(self, pt_coord):
        mask_idx = self.find_containing_mask(pt_coord)
        mask = self.masks[mask_idx]
        img = self.img.numpy().transpose(1, 2, 0)
        masked_img = img * mask.transpose(1, 2, 0) * 255
        masked_img = Image.fromarray(masked_img.astype(np.uint8))
        
        from face_detection import FACE
        face = FACE()
        face.img = masked_img
        box, _ = face.detect()
        box_coord = tuple(box[0].tolist())
        crop_face = Image.fromarray((img * 255).astype(np.uint8)).crop(box_coord)
        return crop_face



if __name__ == "__main__":
    model = Segmenter()
    model.read_img("assets/img1.jpg")
    start = time.time()
    model.get_mask()
    model.save_mask()
    print(f"Took {time.time() - start}")

    pt_coord = (300, 30)
    face = model.find_containing_face(pt_coord)
    face.show()
