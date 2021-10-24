from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torch
import facenet_pytorch
import numpy as np
from pdb import set_trace as bp

class FACE(object):
    """
    https://github.com/timesler/facenet-pytorch
    """
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.instance_name_dict = None
        self.instance_embeddings_log = []
        self.face_log = []
    
    def read_img(self, img_fp:str):
        self.img = Image.open(img_fp)

    def detection(self):
        self.boxes, self.probs = self.mtcnn.detect(self.img)
        return self.boxes, self.probs
    
    def recognition(self, thres=1.0):
        # recog_faces is a list of cropped faces
        recog_faces = self.mtcnn(self.img)
        if recog_faces is not None:
            recog_faces = [r for r in recog_faces]

            # instance_embedding_list is a list of 512-dimensional feature vector of each instance
            aligned = torch.stack(recog_faces)
            instance_embeddings = self.resnet(aligned).detach().cpu().numpy()
            instance_embeddings = [e for e in instance_embeddings]

            # compare with self.embeddings to see if the instance is already present
            if len(self.instance_embeddings_log) == 0:
                # self.instance_embeddings_log = instance_embeddings
                
                # for 
                # recog_face = recog_faces[idx].detach().cpu().numpy().transpose(1, 2, 0)
                # recog_face = ((recog_face + 1) / 2 * 255).astype(np.uint8)
                # recog_face = Image.fromarray(recog_face)
                # self.face_log.append(recog_face)
                for idx, instance_embedding in enumerate(instance_embeddings):
                    self.instance_embeddings_log.append(instance_embedding)
                    recog_face = recog_faces[idx].detach().cpu().numpy().transpose(1, 2, 0)
                    recog_face = ((recog_face + 1) / 2 * 255).astype(np.uint8)
                    recog_face = Image.fromarray(recog_face)
                    self.face_log.append(recog_face)


            else:
                for idx, instance_embedding in enumerate(instance_embeddings):
                    dist = [np.linalg.norm(self.instance_embeddings_log[idx] - instance_embedding) for idx in range(len(self.instance_embeddings_log))]
                    if np.min(dist) > thres:
                        print('new guy')
                        self.instance_embeddings_log.append(instance_embedding)
                        recog_face = recog_faces[idx].detach().cpu().numpy().transpose(1, 2, 0)
                        recog_face = ((recog_face + 1) / 2 * 255).astype(np.uint8)
                        recog_face = Image.fromarray(recog_face)
                        self.face_log.append(recog_face)
        return self.face_log
    
    def draw_bbox(self, out_fp='bbox_out.png'):
        if not self.boxes:
            raise Exception('self.detect not called')

        img_draw = self.img.copy()
        draw = ImageDraw.Draw(img_draw)
        for box in self.boxes:
            draw.rectangle(box.tolist(), width=5)
        img_draw.save(out_fp)
    

if __name__=="__main__":
    face = FACE()
    face.read_img('img1.jpg')
    face.detection()
    embeddings = face.recognition()

    face.read_img('img2.jpg')
    face.recognition()
