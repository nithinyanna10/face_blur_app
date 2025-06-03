import cv2
import numpy as np
import os
from PIL import Image

# Load DNN model
MODEL_DIR = "models"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# Load emoji if needed
EMOJI_PATH = os.path.join("assets", "emoji.png")
if os.path.exists(EMOJI_PATH):
    emoji_img = cv2.imread(EMOJI_PATH, cv2.IMREAD_UNCHANGED)  # Emoji with alpha
else:
    emoji_img = None

def detect_faces_dnn(image, conf_threshold=0.6):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1 - x, y1 - y))  # x, y, w, h
    return faces

def apply_blur(image, x, y, w, h, intensity):
    roi = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (intensity, intensity), 0)
    image[y:y+h, x:x+w] = blurred
    return image

def apply_pixelate(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated
    return image

def apply_blackbar(image, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
    return image

def apply_emoji(image, x, y, w, h):
    if emoji_img is None:
        return apply_pixelate(image, x, y, w, h)
    emoji_resized = cv2.resize(emoji_img, (w, h))
    for c in range(3):
        alpha = emoji_resized[:, :, 3] / 255.0
        image[y:y+h, x:x+w, c] = image[y:y+h, x:x+w, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
    return image

def blur_faces(image, faces, style="blur", intensity=51, show_box=False):
    for (x, y, w, h) in faces:
        if style == "blur":
            image = apply_blur(image, x, y, w, h, intensity)
        elif style == "pixelate":
            image = apply_pixelate(image, x, y, w, h)
        elif style == "blackbar":
            image = apply_blackbar(image, x, y, w, h)
        elif style == "emoji":
            image = apply_emoji(image, x, y, w, h)
        if show_box:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def process_video(input_path, style="blur", intensity=51, show_box=False):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = input_path.replace(".mp4", "_blurred.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces_dnn(frame)
        processed = blur_faces(frame, faces, style=style, intensity=intensity, show_box=show_box)
        out.write(processed)

    cap.release()
    out.release()
    return output_path
