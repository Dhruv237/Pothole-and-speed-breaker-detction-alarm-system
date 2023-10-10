from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from pygame import mixer

cap = cv2.VideoCapture('C:/Users/Dell/Downloads/mixkit-potholes-in-a-rural-road-25208-medium (1).mp4')  # For Webcam
#cap = cv2.VideoCapture(0) 
#cap = cv2.VideoCapture("C:/Users/Dell/Downloads/speed_breaker.mp4")
#cap = cv2.VideoCapture('C:/Users/Dell/Downloads/stock-footage-cars-drive-in-close-up-on-a-bad-bumpy-road-with-holes-large-potholes-after-rain-damaged-road.webm')
#cap = cv2.VideoCapture('C:/Users/Dell/Downloads/stock-footage-aerial-top-down-car-driving-over-big-road-pits-on-countryside-road-in-tropics-careful-drive-while.webm')

#cap=cv2.VideoCapture('C:/Users/Dell/Downloads/video_of_travel (1080p).mp4')
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("C:/Users/Dell/Downloads/best_model.pt")

classNames = ["speed breaker","potholes"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "speed breaker" and conf > 0.5:
                print(f"speed breaker or broken road detected at {x1, y1}!")
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1,
                                   offset=3)
                mixer.init()
                sound = mixer.Sound("C:/Users/Dell/Downloads/beep-01a.mp3")
                sound.play()
            if currentClass == "potholes" and conf>0.5:
                print(f"pothole detected at {x1, y1}!")
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1,
                                     offset=3)
                mixer.init()
                sound = mixer.Sound("C:/Users/Dell/Downloads/beep-01a.mp3")
                sound.play()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

