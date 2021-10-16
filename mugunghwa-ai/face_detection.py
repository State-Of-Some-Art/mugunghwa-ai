from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw

class FACE(object):
    """
    https://github.com/timesler/facenet-pytorch
    """
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
    
    def read_img(self, img_fp:str):
        self.img = Image.open(img_fp)

    def detect(self):
        self.boxes, self.probs = self.mtcnn.detect(self.img)
        return self.boxes, self.probs
    
    def draw_bbox(self, out_fp='bbox_out.png'):
        if not self.boxes:
            raise Exception('self.detect not called')

        img_draw = self.img.copy()
        draw = ImageDraw.Draw(img_draw)
        for box in self.boxes:
            draw.rectangle(box.tolist(), width=5)
        img_draw.save(out_fp)
