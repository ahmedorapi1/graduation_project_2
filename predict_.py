from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
pred = model.predict(source='3.jpg', conf=0.65, imgsz=320, save=True)
result = pred[0]
img = cv2.imread('2.jpg')
crops = []
for i, box in enumerate(result.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    crops.append(crop)

