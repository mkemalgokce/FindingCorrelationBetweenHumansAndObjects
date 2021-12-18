
import copy
import math
import cv2
import mediapipe as mp
from main import getObjectsFromImages


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
def getLandmarksFromVideo(video):

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        
        
        if not success:
          print("Ignoring empty camera frame.")
          break
          # If loading a video, use 'break' instead of 'continue'.
          continue
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

            
            #cv2.imshow("Cropped", croppedImage)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results.pose_landmarks:
            imageWidth = image.shape[1]
            imageHeight = image.shape[0]
            upperY = int(results.pose_landmarks.landmark[0].y * imageHeight ) -200
            lowerY = int(results.pose_landmarks.landmark[32].y * imageHeight ) + 200
            lowerX = int(results.pose_landmarks.landmark[18].x * imageWidth ) + 200
            upperX = int(results.pose_landmarks.landmark[17].x * imageWidth ) - 200
            
            croppedImage = image[min(lowerY, upperY): max(lowerY, upperY)+1, min(lowerX, upperX): max(lowerX, upperX)]   
            cv2.imshow("Image", croppedImage)
            
        if cv2.waitKey(1) & 0xFF == 27:
            success = False
            break

        # cv2.putText(image,f"X : {a}", (400,400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)
        
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

def getStepCountsFromVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    stepCounts = 0
    firstDistance = 0
    distanceList = []
    p1,p2,p3,p4 = [0,0], [0,0], [0,0], [0,0]
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

                
                #cv2.imshow("Cropped", croppedImage)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image2 = image.copy()
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if not ret:
                break 
            if results.pose_landmarks:
                imageWidth = image.shape[1]
                imageHeight = image.shape[0]
                
                oncekiDistance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                p1 = int(results.pose_landmarks.landmark[30].x * imageWidth) , int(results.pose_landmarks.landmark[30].y * imageHeight)
                p2 = int(results.pose_landmarks.landmark[31].x * imageWidth) , int(results.pose_landmarks.landmark[31].y * imageHeight)
                
                
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distanceList.append(distance)
                
                if len(distanceList) == 14:
                    minIndex = distanceList.index(min(distanceList))
                    
                    if minIndex != 0 and minIndex != len(distanceList)-1:
                        print(distanceList[minIndex] , distanceList[minIndex + 1])
                        if distanceList[minIndex] < distanceList[minIndex + 1]:
                            stepCounts += 1
                    distanceList = []
                
                # if len(distanceList) == 15:
                #     print(min(distanceList), max(distanceList))
                #     if (min(distanceList) < 40 and max(distanceList) - min(distanceList) > 20):
                #         stepCounts +=1
                #     distanceList = []

                                
                

            if cv2.waitKey(1) & 0xFF == 27:
                ret = False 
                break
            
            if cv2.waitKey(1) == ord('s'):
                getObjectsFromImages(image2)
            cv2.putText(image,f"Step Count: {stepCounts}", [0,50], cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255],5)
            cv2.imshow("Image",image)
if __name__ == "__main__":
 getStepCountsFromVideo("Examples/lastvideo.mp4")