import inline as inline
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import time


transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), ])
dataset = ['Closed', 'Opened']

model = torch.jit.load(r"C:\Pycharm\SleepyDriver\model_scripted.pt", map_location=torch.device('cpu'))
model.eval()
wait_time = 5
last_closed_eyes_time = time.time()


def predict_image(img, model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)

    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label

    return dataset[preds[0].item()]


def predict_external_image(img):
    image = Image.fromarray(img)
    example_image = transformations(image)
    # plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, model))


cap = cv2.VideoCapture(r'C:\Pycharm\SleepyDriver\test2.mp4')

if not cap.isOpened():
    print("Error opening video stream or file")

i = 0
# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    predict_external_image(frame)
    image = Image.fromarray(frame)
    example_image = transformations(image)
    cls = predict_image(example_image, model)

    if cls == 'Closed':
        if i == 0:
            last_closed_eyes_time = time.time()
        i += 1
        if time.time() - last_closed_eyes_time > wait_time:
            # Отправка сообщения о том, что человек спит
            print("Человек спит!")
    else:
        i = 0

    if cv2.waitKey(1) == ord('q'):
        break

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
