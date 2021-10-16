from typing import List, Tuple
from gluoncv import model_zoo, data, utils
import mxnet as mx
import numpy as np
from PIL import Image, ImageDraw


class MASK_RCNN(object):
    """
    https://cv.gluon.ai/build/examples_instance/demo_mask_rcnn.html
    """
    def __init__(self, model='mask_rcnn_resnet50_v1b_coco'):
        self.net = model_zoo.get_model(model, pretrained=True)
    
    def read_img(self, img_fp:str, img_size:Tuple[int, int]=(400, 400)):
        self.img = Image.open(img_fp).resize(img_size)
        self.img_size = img_size
        self.width, self.height = img_size
        
    def get_mask(self):
        x = np.asarray(self.img)
        x, _ = data.transforms.presets.rcnn.transform_test(mx.nd.array(x), short=self.width)
        out = self.net(x) # out = [class, confidence, bboxes, seg masks]
        classes, scores, bboxes, masks = [item[0].asnumpy() for item in out]
        
        person_mask = (classes.flatten() == 0).tolist()
        scores = scores[person_mask]
        bboxes = bboxes[person_mask]
        masks = masks[person_mask]

        self.masks, _ = utils.viz.expand_mask(masks, bboxes, self.img_size, scores)
        # >>> masks.shape
        # >>> (n_person, width, height)
        return self.masks
    
    def save_mask(self, out_fp):
        mask_img = None
        for mask in self.masks:
            if mask_img is None: mask_img = mask
            else: mask_img = mask_img | mask
        mask_img = Image.fromarray(mask_img*255)
        mask_img.save(out_fp)
        print(f'mask image saved at {out_fp}')

    def find_containing_mask(self, pt_coord):
        mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][pt_coord]]
        
        if len(mask_idx) == 0:
            return Exception('Containing mask not found')
        else:
            mask_idx = mask_idx[0]
        
        return mask_idx

    def save_containing_face(self, pt_coord, out_fp):
        mask_idx = [idx for idx in range(len(self.masks)) if self.masks[idx][pt_coord]]
        mask = self.masks[mask_idx]
        masked_img = np.array(self.img) & mask.transpose(1, 2, 0) * 255
        masked_img = Image.fromarray(masked_img)
        
        from face_detection import FACE
        face = FACE()
        face.img = masked_img
        box, _ = face.detect()
        box_coord = tuple(box[0].tolist())
        crop_face = self.img.crop(box_coord)
        crop_face.save(out_fp)
        print(f'crop face image saved at {out_fp}')
