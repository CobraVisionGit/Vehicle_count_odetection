# Vehicle_count_odetection
Vehicle counting through object detection


* The code utilizes OpenCV for video capture and processing along with Pytorch and the pre-trained YOLOv5s model for object detection.
It first imports the necessary libraries - OpenCV for image and video processing, torch for loading and using the YOLOv5s model, numpy for numerical operations on arrays.
A video stream from 'highway.mp4' is captured and opened using OpenCV's VideoCapture.
Two counters are defined - count to keep track of frames and object_count to count the detected objects.
Next, two points are defined to specify the blue rectangular region of interest on the frame where objects need to be counted. This region can be changed by modifying the (x,y) coordinates.


* In the main loop, video frames are read one by one. Only every 3rd frame is processed to reduce latency and redundant counting.
The frame is resized to a fixed size for faster processing. A blue rectangle is drawn using the predefined points.
The YOLOv5s model is used to detect objects in this frame and bounding boxes are extracted.


* These bounding boxes are scaled to adjust for the frame resizing. Then each box is checked - if it lies inside the blue ROI, the object count is incremented.
Finally, the object count is displayed on the frame itself. The video visual is shown via OpenCV along with terminal prints of object count.
So in summary, this code demonstrates how to utilize deep learning based object detection to count objects in a specific region of interest in real-time in a video stream. The comments guide modifying the ROI and other parameters.
We can modify and fine-tune two parameters in the code: the difference in y values (height) of the blue rectangle which is the ROI, and the frame processing rate.
Having a wider region of interest can increase redundant counts, and processing more frames in less time can also increase redundant count, since things don’t change much between subsequent frames. 
