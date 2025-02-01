import os
import cv2
import numpy as np
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from ultralytics import YOLO
from .camera import VideoCamera, generate_frames
from django.views.decorators.csrf import csrf_exempt






# Global variable to store the people count
people_count = 0

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Ensure the uploads directory exists
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    global people_count
    camera = VideoCamera()

    def generate():
        global people_count
        while True:
            frame, count = camera.get_frame()
            people_count = count
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(generate(), content_type="multipart/x-mixed-replace; boundary=frame")

def get_people_count(request):
    return JsonResponse({'count': people_count})

@csrf_exempt  # Disable CSRF for this API
def process_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        file_path = os.path.join(UPLOADS_DIR, image_file.name)  # Corrected path

        # Save the uploaded image
        with open(file_path, "wb") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Load image with OpenCV
        img = cv2.imread(file_path)

        # Check if the image was loaded correctly
        if img is None:
            return JsonResponse({"error": "Failed to load the image"}, status=400)

        # Run YOLOv8 detection
        results = model(img)

        # Count people
        people_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Filter for 'person' class (COCO class 0)
                if int(box.cls[0]) == 0 and confidence > 0.4:
                    people_count += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'Person {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image
        processed_image_path = os.path.join(UPLOADS_DIR, f"processed_{image_file.name}")
        cv2.imwrite(processed_image_path, img)

        return JsonResponse({
            "count": people_count,
           
        })

    
    return JsonResponse({"error": "Invalid request"}, status=400)
