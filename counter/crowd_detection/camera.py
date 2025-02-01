import cv2
import numpy as np
from ultralytics import YOLO

class VideoCamera:
    def __init__(self, width=1280, height=720):
        self.video = cv2.VideoCapture(0)  # Use webcam
        
        # Set custom resolution (width and height)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set the frame width
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set the frame height
        
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 pre-trained model
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, 0

        # Run YOLOv8 on the frame
        results = self.model(frame)

        # Count number of people detected
        people_count = 0

        # Process results and draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Filter for 'person' class (COCO class 0)
                if int(box.cls[0]) == 0 and confidence > 0.4:
                    people_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display people count on the frame
        cv2.putText(frame, f'People Count: {people_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), people_count

def generate_frames(camera):
    while True:
        frame, _ = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
