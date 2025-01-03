from ultralytics import YOLO
import cv2
import cvzone
model = YOLO('../yolo-weights/yolo11n.pt')
results = model("../RunningYolo/Images/image1.jpg",show=True)


# cv2.namedWindow("ImageDetection",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("ImageDetection",854,480)
# cv2.imshow("ImageDetection",results)

cv2.waitKey(0)




