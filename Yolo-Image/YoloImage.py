from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8n.pt')
results = model("../RunningYolo/Images/image3.jpg",show=True)
cv2.waitKey(0)
