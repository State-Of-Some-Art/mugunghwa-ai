from face_detection import FACE
from mask_rcnn_torch import Segmenter

# Create face detection instance
face = FACE()
face.read_img('assets/img1.jpg')
boxes, _ = face.detection()

# Create mask-rcnn instance
m = Segmenter()

# Read image and extract instance segmentation mask
m.read_img('assets/img1.jpg')
masks = m.get_mask()

# Mask can be saved as an image
m.save_mask('assets/mask.png')

# Particular mask that contains given point coordinate can be extracted
pt = (300, 30)
m.find_containing_mask(pt)
m.find_containing_face(pt)