import cv2
import numpy as np
from tempfile import NamedTemporaryFile

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

def blur_faces(image, faces, style="blur", intensity=50, show_box=False):
    result = image.copy()

    for (x, y, w, h) in faces:
        face_region = result[y:y+h, x:x+w]

        if style == "blur":
            blurred = cv2.GaussianBlur(face_region, (intensity | 1, intensity | 1), 30)
        elif style == "pixelate":
            small = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif style == "blackbar":
            blurred = np.zeros_like(face_region)
        else:
            blurred = face_region

        result[y:y+h, x:x+w] = blurred

        if show_box:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return result

def process_video(input_path, style, intensity, show_box):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        blurred_frame = blur_faces(frame, faces, style=style, intensity=intensity, show_box=show_box)
        out.write(blurred_frame)

    cap.release()
    out.release()
    return output_temp.name
