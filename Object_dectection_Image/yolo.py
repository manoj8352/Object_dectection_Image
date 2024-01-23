from ultralytics import YOLO
import cv2

# Load the image
img = cv2.imread("images1.jpg")

# Specify the desired width and height
width = 1000
height = 1000

# Resize the image
img_resized = cv2.resize(img, (width, height))

# Load the model
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Pass the resized image to the model
results = model(img_resized, show=True)

cv2.waitKey(0)
