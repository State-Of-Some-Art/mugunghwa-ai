from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from convert import *
from pdb import set_trace as bp

class FaceNet(object):
    """
    https://github.com/timesler/facenet-pytorch
    Provides face recognition feature
    """
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2')

        self.mtcnn.eval()
        self.resnet.eval()

        self.inst_embs_log = []
        self.face_log = []
    
    def read_img(self, img_fp:str):
        self.img = Image.open(img_fp)
    
    def set_img(self, src):
        # src is a (W, H, C) np.ndarray
        self.img = Image.fromarray(src)
    
    def reset_log(self):
        self.inst_embs_log = []
        self.face_log = []    
    
    def update_face_log(self, thres=1.0):
        # recog_faces is a (N, C, H, W) tensor of cropped faces
        # C = 3, H = W = 160 by default
        recog_faces = self.mtcnn(self.img)
        if recog_faces is not None:
            # curr_inst_embs is a list of 512-dimensional feature vector of each instance
            curr_inst_embs = self.resnet(recog_faces).detach().cpu().numpy()

            # If instance embeddings log is empty, append instance embedding and face images to
            # corresponding lists
            if len(self.inst_embs_log) == 0:
                for idx, curr_inst_emb in enumerate(curr_inst_embs):
                    self.inst_embs_log.append(curr_inst_emb)
                    
                    recog_face = recog_faces[idx].detach().cpu().numpy()
                    recog_face = CHW2HWC(recog_face)
                    recog_face = to_8bit(recog_face, min=-1, max=1)
                    # recog_face = Image.fromarray(recog_face)
                    self.face_log.append(recog_face)
            # Else, compare with instance embeddings log to see if the instance is already present
            else:
                for idx, curr_inst_emb in enumerate(curr_inst_embs):
                    dist = [np.linalg.norm(self.inst_embs_log[idx] - curr_inst_emb) for idx in range(len(self.inst_embs_log))]
                    if np.min(dist) > thres:
                        print('new guy')
                        self.inst_embs_log.append(curr_inst_emb)

                        recog_face = recog_faces[idx].detach().cpu().numpy()
                        recog_face = CHW2HWC(recog_face)
                        recog_face = to_8bit(recog_face, min=-1, max=1)
                        # recog_face = Image.fromarray(recog_face)
                        self.face_log.append(recog_face)
        return self.face_log    

if __name__=="__main__":
    face = FaceNet()
    face.read_img('img3.jpg')
    face.update_face_log(thres=2.0)

    face.read_img('img2.jpg')
    face.update_face_log()

    face_img = np.hstack(face.face_log)
    Image.fromarray(face_img).show()