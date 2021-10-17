from torchvision.models.detection import maskrcnn_resnet50_fpn as mrcnn
import torch
from torchvision.io import read_image, write_png
import time
import numpy as np
import cv2
from typing import Tuple

class Segmenter(object):
    def __init__(self):
        self.is_gpu = torch.cuda.device_count() > 0
        self.net = mrcnn(pretrained=True)
        self.net.eval()
        if self.is_gpu:
            self.net.cuda()
    
    def read_img(self, img_fp:str, img_size: Tuple[int, int]=(400, 400)):
        self.img = read_image(img_fp) / 255.0
        self.img = torch.from_numpy(cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) / 255
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
        print(self.masks.shape)
        return self.masks
    
    def save_mask(self):
        write_png((self.img.cpu().detach() * 255).type(torch.uint8), "original.png")
        print(self.img.shape)
        for i, mask in enumerate(self.masks):
            print(mask.shape)
            write_png(torch.from_numpy((self.img.cpu().detach().numpy() * mask * 255).astype(np.uint8)), f"{i}.png")

    def find_containing_mask(self, pt_coord):
        mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][pt_coord]]
        
        if len(mask_idx) == 0:
            return Exception('Containing mask not found')
        else:
            mask_idx = mask_idx[0]
        
        return mask_idx

    # def find_containing_face(self, pt_coord):
    #     mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][pt_coord]]
    #     mask = self.masks[mask_idx]
    #     masked_img = np.array(self.img) & mask.transpose(1, 2, 0) * 255
    #     masked_img = Image.fromarray(masked_img)
        
    #     from face_detection import FACE
    #     face = FACE()
    #     face.img = masked_img
    #     box, _ = face.detect()
    #     box_coord = tuple(box[0].tolist())
    #     crop_face = self.img.crop(box_coord)
    #     return crop_face



if __name__ == "__main__":
    model = Segmenter()
    model.read_img("assets/img1.jpg")
    start = time.time()
    model.get_mask()
    model.save_mask()
    print(f"Took {time.time() - start}")

