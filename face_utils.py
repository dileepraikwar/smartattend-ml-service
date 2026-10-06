import base64
import io
import os
import numpy as np
import cv2
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_data")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_recognizer = cv2.face.LBPHFaceRecognizer_create()
_model_trained = os.path.exists(MODEL_PATH)
if _model_trained:
    _recognizer.read(MODEL_PATH)


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Base64 string ko grayscale numpy image mein convert karta hai."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    return np.array(image)


def detect_face(gray_image: np.ndarray):
    """Image mein face dhundta hai, cropped+resized grayscale face lauta hai. Na mile to None."""
    faces = _face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray_image[y:y + h, x:x + w]
    face = cv2.resize(face, (200, 200))
    return face


def enroll_face(user_id: int, image_base64: str) -> bool:
    """User ka face model mein add/update karta hai. Face na mile to False."""
    global _model_trained
    gray = decode_base64_image(image_base64)
    face = detect_face(gray)
    if face is None:
        return False

    label = int(user_id)
    if _model_trained:
        _recognizer.update([face], np.array([label]))
    else:
        _recognizer.train([face], np.array([label]))
        _model_trained = True

    _recognizer.save(MODEL_PATH)
    return True


def verify_face(claimed_user_id: int, image_base64: str, confidence_threshold: float = 100.0):
    """Image claimed user se match karti hai ya nahi. LBPH confidence: kam = zyada accha match."""
    if not _model_trained:
        return False, 0.0

    gray = decode_base64_image(image_base64)
    face = detect_face(gray)
    if face is None:
        return False, 0.0

    predicted_label, confidence = _recognizer.predict(face)

    match = (predicted_label == int(claimed_user_id)) and (confidence <= confidence_threshold)
    normalized_confidence = max(0.0, 1 - (confidence / 100))
    return match, round(normalized_confidence, 4)
