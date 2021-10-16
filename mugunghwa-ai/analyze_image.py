from face_detection import FACE
from mask_rcnn import MASK_RCNN

# Create face detection instance
face = FACE()
face.read_img('assets/img1.jpg')
boxes, _ = face.detect()

# Create mask-rcnn instance
m = MASK_RCNN()

# Read image and extract instance segmentation mask
m.read_img('assets/img1.jpg')
masks = m.get_mask()

# Mask can be saved as an image
m.save_mask('assets/mask.png')

# Particular mask that contains given point coordinate can be extracted
pt = (300, 30)
m.find_containing_mask(pt)
m.save_containing_face(pt, 'assets/containing_face.png')