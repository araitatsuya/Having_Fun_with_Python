#
#   Face Detection using Open CV and haarcascade
#
#                                   Tatsuya Arai 
#
#   Python 3.6.1
#   cv2.__version__ '3.3.0'

# References
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
# https://stackoverflow.com/questions/30506126/open-cv-error-215-scn-3-scn-4-in-function-cvtcolor
# http://qiita.com/mo4_9/items/fda37ac4f9ddab2d7f45

#
# ----faceRecog.py
#  |
#  ---data---faceRecog---eight_faces.jpg
#                     |
#                     ---haarcascade
#                     |
#                     ---output_faces
#

# Modules
import os
import cv2

# Classifier
# You need to download .xml files from the link. 
# https://github.com/opencv/opencv/tree/master/data/haarcascades
classifier = cv2.CascadeClassifier('./data/faceRecog/haarcascade/haarcascade_frontalface_default.xml')

### Face Recognition
# Image
image = cv2.imread('./data/faceRecog/eight_faces.jpg')
# Gray Scale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(gray_image)

print('There are {:d} faces'.format(len(faces)))

# Output
# Make Directry
output_dir = './data/faceRecog/output_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, (x,y,w,h) in enumerate(faces):
    # crop faces
    face_image = image[y:y+h, x:x+w]
    output_path = os.path.join(output_dir,'{0}.jpg'.format(i))
    cv2.imwrite(output_path,face_image)

cv2.imwrite('./data/faceRecog/output_faces/face.jpg',image)

for x,y,w,h in faces:
    # Boundary Box
    cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,0,255), thickness=3)

cv2.imwrite('./data/faceRecog/output_faces/faces.jpg',image)
