import cv2 
import os
import time 
import sys
cap = cv2.VideoCapture(sys.argv[1])
tic = time.time()
imageCount = 0

while cap.isOpened():
    ret, frame = cap.read()
    toc = time.time()
    if not ret:
        print("Finishing !")
        break
    if toc - tic > 0.1:
        image = cv2.resize(frame, [640,360])
        print(frame.shape)
        cv2.imwrite("image"+str(imageCount)+".jpeg", image)
        tic = time.time()
        imageCount += 1

    if cv2.waitKey(1) & 0xFF == 27:
        ret = False

cap.release()
cv2.destroyAllWindows()

