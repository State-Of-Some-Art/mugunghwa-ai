from PIL import Image
from face_detection import FACE
from torchvision.io import write_png
from torchvision.models.detection import maskrcnn_resnet50_fpn as mrcnn
from typing import Tuple
import cv2
import numpy as np
import time
import torch

class Segmenter(object):
    def __init__(self):
        self.net = mrcnn(pretrained=True)
        self.face = FACE()

        self.net.eval()
        self.face.mtcnn.eval()
        self.face.resnet.eval()

        self.is_gpu = torch.cuda.device_count() > 0
        if self.is_gpu:
            self.net.cuda()
        
    def reset(self):
        self.face.instance_name_dict = None
        self.face.instance_embeddings_log = []
        self.face.face_log = []
    
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
    
    def set_image_size(self, img_size):
        self.img_size = img_size
        self.width, self.height = img_size
        
    def get_mask(self):
        x = torch.stack([self.img])
        out = self.net(x)[0]
        person_mask = out['labels'] == 1
        score_mask = out['scores'] > 0.90
        comb_mask = person_mask & score_mask

        labels = out['labels'][comb_mask].cpu().detach().numpy()
        masks = out['masks'][comb_mask].cpu().detach().numpy()
        
        self.labels = np.array(list(range(len(labels))))
        instance_masks = np.array([(self.labels[i] + 1) * np.array(masks[i] > 0.5, dtype=np.int) for i in self.labels], dtype=np.int)
        
        self.masks = np.zeros([1, *self.img.shape[1:]], dtype=np.int)
        for instance, instance_mask in enumerate(instance_masks):
            instance += 1
            self.masks = np.clip(self.masks + instance_mask, 0, instance, dtype=np.int)
        return self.masks
    
    def save_mask(self):
        write_png((self.img.cpu().detach() * 255).type(torch.uint8), "original.png")
        for i, mask in enumerate(self.masks):
            write_png(torch.from_numpy((self.img.cpu().detach().numpy() * mask * 255).astype(np.uint8)), f"{i}.png")

    def find_containing_mask(self, pt_coord):
        mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][0][pt_coord]]
        if len(mask_idx) == 0:
            return -1
        else:
            return mask_idx[0]

    def find_instance_faces(self, instance_mask_list):
        faces = []
        src_img = self.img.detach().cpu().numpy().transpose(1, 2, 0)

        for idx in instance_mask_list:
            if idx == 0:
                continue

            # Mask current instance
            instance_mask = (self.masks == idx).transpose(1, 2, 0)
            masked_img = src_img * instance_mask * 255

            # Extract face from the masked image
            self.face.img = Image.fromarray(masked_img.astype(np.uint8))
            # box, _ = self.face.detection()
            face_list = self.face.recognition(thres=1.2)
            faces = [np.array(f) for f in face_list]

        return faces

    def find_containing_face(self, pt_coord):
        mask_idx = self.find_containing_mask(pt_coord[::-1])

        if mask_idx < 0:
            print("Could not find face")
            return None
        mask = self.masks[mask_idx]
        img = self.img.cpu().numpy().transpose(1, 2, 0)
        masked_img = img * mask.transpose(1, 2, 0) * 255
        masked_img = Image.fromarray(masked_img.astype(np.uint8))
        
        from face_detection import FACE
        face = FACE()
        face.img = masked_img
        box, _ = face.detection()
        if box is None:
            return None
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
