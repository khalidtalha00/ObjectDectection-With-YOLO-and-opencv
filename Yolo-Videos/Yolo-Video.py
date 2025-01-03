from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import  math

# capture = cv2.VideoCapture("../RunningYolo/Videos/person-bicycle-car-detection.mp4")
# capture = cv2.VideoCapture("../RunningYolo/Videos/Cars Moving On Road Stock Footage - Free Download.mp4")
# capture = cv2.VideoCapture("../RunningYolo/Videos/worker-zone-detection.mp4")
capture = cv2.VideoCapture("../RunningYolo/Videos/classroom.mp4")
capture.set(3,640)
capture.set(4,480)

model = YOLO("../yolo-weights/yolo11n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike","water purifier", "aeroplane","train","truck", "boat","bus",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "skis", "snowboard", "sports ball",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "potted plant", "bed", "dining table", "toilet", "laptop", "mouse", "remote",
    "keyboard", "Mobile Phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "Bench","sign"
]




while True:
    success,img = capture.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
           x1, y1, x2, y2 =box.xyxy[0]
           x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
           w, h =x2-x1, y2-y1
           cvzone.cornerRect(img,(x1,y1,w,h))

            # confidence
           conf = math.ceil((box.conf[0]*100))
           cls = int(box.cls[0])

            #clASS NAME
           cvzone.putTextRect(img,f'{classNames[cls]} {conf}%',(max(0,x1),max(35,y1)),scale=2,thickness=2)

    # cv2.namedWindow("Detecting Video...", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Detecting Video...",854,480)
    cv2.imshow("Detecting Video...",img)
    key =cv2.waitKey(1)
