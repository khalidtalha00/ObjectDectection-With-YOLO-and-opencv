from ultralytics import YOLO
import cv2
import cvzone
import  math

capture = cv2.VideoCapture("../RunningYolo/Videos/person-bicycle-car-detection.mp4")
# capture = cv2.VideoCapture("Videos/bottle-detection.gif")
capture.set(3,640)
capture.set(4,480)

model = YOLO("../yolo-weights/yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "Mobile Phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "Bench"
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
           conf = math.ceil((box.conf[0]*100))/100
           cls = int(box.cls[0])

            #clASS NAME
           cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=2,thickness=3)




    cv2.imshow("Image",img)
    cv2.waitKey(1)