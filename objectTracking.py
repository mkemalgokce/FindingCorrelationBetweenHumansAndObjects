import cv2


def trackObjects(image, bbox):
    cap = cv2.VideoCapture(0)

    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker.init(image, bbox)
    while cap.isOpened():
        timer = cv2.getTickCount()
        ret,frame = cap.read()
        fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
        if fps > 60:
            fps = 60
        img = cv2.putText(frame, f"FPS: {fps}", [0,50], cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)

        if cv2.waitKey(1) & 0xFF == 27:
            ret = False
            break
        if not ret:
            break    
        cv2.imshow("Tracking", img)
    cap.release()
    cv2.destroyAllWindows()

    def drawRect(img, bbox):
        cv2.rectangle(img, [bbox[0], bbox[1]] , [bbox[0]+ bbox[2], bbox[1]+ bbox[3]], [255, 0, 0], 1)