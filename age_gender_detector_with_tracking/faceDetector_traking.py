# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
import cv2

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import dlib

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']
 
# allow the camera to warmup
time.sleep(0.1)


def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_model/deploy_age.prototxt", 
                        "age_gender_model/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_model/deploy_gender.prototxt", 
                        "age_gender_model/gender_net.caffemodel")
 
    return (age_net, gender_net)


def capture_loop(age_net, gender_net): 
    font = cv2.FONT_HERSHEY_SIMPLEX
    # capture frames from the camera
    totalFrames = 0
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
     
    W = None
    H = None
    tracker = dlib.correlation_tracker()
    ages = []
    genders = []
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        #/usr/local/share/OpenCV/haarcascades/
        face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   
       
        if W is None or H is None:
            (H,W) = image.shape[:2]
        
        rects = []
        
        if totalFrames % 5 == 0:
            trackers = []
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print("Found " + str(len(faces)) + " face(s)")
            #Draw a rectangle around every found face
            ages = []
            genders = []
            for (x,y,w,h) in faces:
                
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
                face_img = image[y:y+h, x:x+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                # Predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                # Predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                overlay_text = "%s, %s" % (gender, age)
                cv2.putText(image, overlay_text ,(x,y), font,1,(255,255,255),2,cv2.LINE_AA)

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x+w, y+h)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                ages.append(age)
                genders.append(gender)

        else :
            # loop over the trackers
            for i in range (len(trackers)):
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                trackers[i].update(rgb)
                pos = trackers[i].get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 0), 2)
                overlay_text = "%s, %s" % (genders[i], ages[i])
                cv2.putText(image, overlay_text ,(startX,startY), font,1,(255,255,255),2,cv2.LINE_AA)
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
            
            
                
                
 # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
        objects = ct.update(rects,ages,genders, image)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            #cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.imshow("Image", image)
 
        key = cv2.waitKey(1) & 0xFF
      
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        totalFrames += 1
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
 
if __name__ == '__main__':
    age_net, gender_net = initialize_caffe_model()
    capture_loop(age_net, gender_net)
        