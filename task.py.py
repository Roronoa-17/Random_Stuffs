## This if only for local machine
import cv2
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow
path_weight = "D:/DL/SpittingDetectionMajorProject/yolov8m_200e.pt" # set the path of yolo weight for face detection . download from here - https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing

path_video = "D:/DL/SpittingDetectionMajorProject/test_video.mp4" # set the path of video . download from here - https://drive.google.com/file/d/1nyeeqBJyDr2zphBDQ9ruh99JBdYm4nPH/view?usp=sharing

# Load the model
yolo = YOLO(path_weight)

# Load the video capture
videoCap = cv2.VideoCapture(path_video)


no_boxes = 0

while True:
    ret, frame = videoCap.read() # Reading frames from videos
    if not ret:
        continue
    results = yolo.track(frame, stream=True) # using yolo for detecting faces
    for result in results:
        # get the classes names
        classes_names = result.names 
        boxes = result.boxes 
        no_boxes = no_boxes + len(boxes) # counter for detected faces in frame


        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), 2)
                
                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(frame, f'No of faces:{no_boxes}', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 3) # displaying count of faces detected in a frame
                
    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()