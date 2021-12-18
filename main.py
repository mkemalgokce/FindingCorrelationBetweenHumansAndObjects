import cv2
import numpy as np
import glob
import random
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import math

def getObjectsFromImages(img):
# Load Yolo
    net = cv2.dnn.readNet("Utilities/yolov3_training_last.weights", "Utilities/yolov3_testing.cfg")

    # Name custom object
    classes = ["bidon"]

    # Images path
    #images_path = ["bid.png","example.png"]


    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images

    # loop through all the images
        # Loading image
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


    plt.imshow(img)
    plt.show()
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

def getObjectsFromVideos(videoList, coolDown):
    """
    [summary]
    Bu fonksiyon videolardaki benzin bidonu objelerini ve bu objelerle insan vucudu arasindaki
    iliskiyi gosterir. 
    Args:
        videoList ([[String]]): [Videolarin pathlerinden olusan array.] 
        coolDown ([Int]): [Objelerin tekrar hesaplanmasi icin gecmesi gereken sure.] 
    """    
    whichHand = "" # Bidonun hangi elde oldugunu gosteren string
    humanPos = "" # Insanin kameraya gore hangi konumda oldugunu gosteren string 
    totalCal = 0 # Insanin yaktigi toplam kaloriyi gosteren Int 
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    # Load Yolo
    net = cv2.dnn.readNet("Utilities/yolov3_training_last.weights", "Utilities/yolov3_testing.cfg")
    classes = ["bidon"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    distanceList = []
    p1,p2 = [0,0], [0,0]
    stepCounts = 0 #Toplam adim sayisi
    tic = time.time()
    previousStepCounts = 0 # 2 saniye onceki adim sayisi
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        for video in videoList:
            cap = cv2.VideoCapture(video)
            while cap.isOpened():
                currentSecond = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                print(currentSecond)
                timer = cv2.getTickCount()
                ret, frame = cap.read()
                img = frame
                img = cv2.resize(img,[960, 540], cv2.INTER_AREA)
                height, width, channels = img.shape
                if not ret:
                    break
                if cv2.waitKey(1) & 0xFF == 27:
                    ret = False
                    break

                image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = pose.process(image)
                if results.pose_landmarks:
                    # Insanin kameraya gore yonunun tespiti
                    if not abs(results.pose_landmarks.landmark[12].x - results.pose_landmarks.landmark[11].x) > 0.009: # 5.6 pixel in normal scene
                        if (results.pose_landmarks.landmark[12].z > results.pose_landmarks.landmark[11].z):
                            humanPos = "Sol"
                        else:
                            humanPos = "Sag"
                        
                    else:
                        if results.pose_landmarks.landmark[0].z > results.pose_landmarks.landmark[12].z or results.pose_landmarks.landmark[0].z > results.pose_landmarks.landmark[11].z :
                            humanPos = "Arka"
                        else:
                            humanPos = "On"
                    
                    ## Adim Hesaplama
                    imageWidth = image.shape[1]
                    imageHeight = image.shape[0]
                    
                    oncekiDistance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    p1 = int(results.pose_landmarks.landmark[30].x * imageWidth) , int(results.pose_landmarks.landmark[30].y * imageHeight)
                    p2 = int(results.pose_landmarks.landmark[31].x * imageWidth) , int(results.pose_landmarks.landmark[31].y * imageHeight)
                    
                    
                    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) # 2landmark arasindaki mesafe
                    distanceList.append(distance)
                    
                    if len(distanceList) == 14: #14 Frame olduktan sonra 
                        minIndex = distanceList.index(min(distanceList))
                        
                        if minIndex != 0 and minIndex != len(distanceList)-1:
                            #minimum elemandan sonra eleman varsa kisi adim atmis..
                            print(distanceList[minIndex] , distanceList[minIndex + 1])
                            if distanceList[minIndex] < distanceList[minIndex + 1]:
                                stepCounts += 1
                        distanceList = []
                    
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                toc = time.time()
                if toc - tic > coolDown :
                    # Detecting Objects
                    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)
                    # Showing informations on the screen
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.7:
                                # Object detected
                                print(class_id)
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    font = cv2.FONT_HERSHEY_PLAIN
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            color = colors[class_ids[i]]
                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
                            cv2.putText(image, label, (x, y + 30), font, 3, color, 4)
                            imageWidth = image.shape[1]
                            imageHeight = image.shape[0]
                            ## Obje hangi elde tutuluyor?
                            if int(results.pose_landmarks.landmark[17].x * imageWidth) > x and int(results.pose_landmarks.landmark[17].x * imageWidth) < x+h:
                                if int(results.pose_landmarks.landmark[17].y * imageHeight) < y+h and int(results.pose_landmarks.landmark[17].y * imageHeight) - y < 10:
                                    whichHand = 'Sol el'
                            elif int(results.pose_landmarks.landmark[18].x * imageWidth) > x and int(results.pose_landmarks.landmark[18].x * imageWidth) < x+h:
                                if int(results.pose_landmarks.landmark[18].y * imageHeight) < y+h and int(results.pose_landmarks.landmark[18].y * imageHeight) - y < 10:   
                                    whichHand = 'Sag el'  
                            elif int(results.pose_landmarks.landmark[18].x * imageWidth) > x and int(results.pose_landmarks.landmark[18].x * imageWidth) < x+h and int(results.pose_landmarks.landmark[17].x * imageWidth) > x and int(results.pose_landmarks.landmark[17].x * imageWidth) < x+h:
                                if results.pose_landmarks.landmark[18].z > results.pose_landmarks.landmark[17].z:
                                    whichHand = 'Sag el'
                                else:
                                    whichHand = 'Sol el'
                            else:
                                whichHand = 'Tutmuyor.'
                            
                    tic = time.time()
                    if len(boxes) == 0:
                        whichHand = 'Tutmuyor.'
                    
                mp_drawing.draw_landmarks(
                   image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # yakılan kalori = [ zaman(S)/ 60 *(MET * 3.5 * ağırlık(80 kg)] /200 
                if currentSecond % 2 < 0.05:
                    if stepCounts - previousStepCounts < 100/30:
                        totalCal += 2/60 * (80 * 2 * 3 ) / 200
                    elif stepCounts - previousStepCounts > 100/30 and stepCounts - previousStepCounts < 4:
                        totalCal += 2/60 * (80 * 3.5 * 3 ) / 200
                    else:
                        totalCal += 2/60 * (80 * 6 * 3 ) / 200
                    previousStepCounts = stepCounts
                
                # Textlerin ekrana yerlestirilmesi
                cv2.putText(image, "Nesne Durumu: "+whichHand, (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0,255,255], 2) 
                cv2.putText(image, "Pozisyon: "+humanPos,(0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0,255,255], 2)
                cv2.putText(image,f"Adim Sayisi: {stepCounts}", [0,90], cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0,255,255],2)
                cv2.putText(image,f"Kalori: {totalCal}", [0,120], cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0,255,255],2)
                # Resmin gosterilmesi
                cv2.imshow("Image", image)

                    
if __name__ == "__main__":
    getObjectsFromVideos(["Examples/bidonludaire.mp4"],0.5)
