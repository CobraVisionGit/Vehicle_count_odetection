import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('highway.mp4')

count = 0
object_count = 0

# Define two points for the blue rectangle- To change width, change difference in y values (510 and 560 here)
rectangle_point1 = (50, 480) #(x,y) of top left corner of rectangle
rectangle_point2 = (750, 560)#(x,y) of bottom right corner of rectangle

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Resize frame first
    frame = cv2.resize(frame, (800, 800))

    # Draw the blue rectangle
    cv2.rectangle(frame, rectangle_point1, rectangle_point2, (255, 0, 0), 2)

    # Detect objects
    results = model(frame)

    # Extract bounding boxes
    boxes = results.xyxy[0].cpu().numpy()

    # Draw bounding boxes and count objects inside the blue rectangle
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        # Preprocessing
        # Scale bounding box coordinates based on the resized frame
        x1 = int(x1 * frame.shape[1] / 800)
        y1 = int(y1 * frame.shape[0] / 800)
        x2 = int(x2 * frame.shape[1] / 800)
        y2 = int(y2 * frame.shape[0] / 800)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Check if the bounding box is inside the blue rectangle
        if x1 > rectangle_point1[0] and y1 > rectangle_point1[1] and x2 < rectangle_point2[0] and y2 < rectangle_point2[1]:
            object_count += 1

    # Display object count on the frame
    cv2.putText(frame, f'Object Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("FRAME", frame)

    # Display object count in the terminal
    print(f'Object Count: {object_count}')

    if cv2.waitKey(1) == 27 or cv2.getWindowProperty('FRAME', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
