import cv2
import os
import dlib
import numpy as np
import matplotlib.pyplot as plt
from openface import AlignDlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from .align_dlib import AlignDlib

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

facePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
print("facePredictor--->",facePredictor)
AD = AlignDlib(facePredictor)

def featureDetection():
    cv2.namedWindow("camera")
    camera = cv2.VideoCapture(0)

    # initialise features to track
    while camera.isOpened():
        ret,im_cv= camera.read()
        if ret:
            #new_img = AD.getAllFaceBoundingBoxes(rgbImg=im_cv)        
             #print(new_img)                                 
             #break    
            #im_cv = cv2.imread('adams.jpg')
            bbs = AD.getAllFaceBoundingBoxes(rgbImg=im_cv)
            print("dlib.rectangle-->",dlib.rectangle)
            color = (0, 0, 255)
            window_name = 'Image'  
            for bb in bbs:
                print("bbs--->",bb.left())
                landmarks = AD.findLandmarks(im_cv, bb)
                print("landmarks--->",landmarks)
                cv2.rectangle(im_cv, (bb.left(), bb.top()), (bb.right(), bb.bottom()), color, thickness = 4)
            
            for landmark in landmarks:
                cv2.circle(im_cv, landmark, 1, color, thickness = 4)
           
            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                break

            cv2.imshow(window_name, im_cv) 
            #cv2.waitKey(0)

    camera.release()
    cv2.destroyWindow("camera")  

def simple():
    cv2.namedWindow("camera")
    camera = cv2.VideoCapture(0)

    # initialise features to track
    while camera.isOpened():
        ret,im_cv= camera.read()
        window_name = 'Image' 
        if ret:

            k = cv2.waitKey(1)
            if k%256 == 27:
                print("Escape hit, closing...")
                break

            cv2.imshow(window_name, im_cv) 
            #cv2.waitKey(0)

    camera.release()
    cv2.destroyWindow("camera")  

featureDetection()
#simple()