import cv2
import numpy as np

def FaceDetection(image):
    # Get user supplied values
    NewImage=np.copy(image) 
    cascPath = "Cascade/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(NewImage, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
   
    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for (a, b, c, d) in faces:
        cv2.rectangle(NewImage, (a, b), (a+c, b+d), (255, 0, 0), 2)
    if(len(faces)>0):    
        cv2.imwrite("Images/FacesDetected.png", NewImage)
    return faces