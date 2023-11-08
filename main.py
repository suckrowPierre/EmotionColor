import cv2
from fer import FER
import numpy as np
import emotion_to_color as e2c

EMOTIONS_COLOR_MAP = {
    "Angry": (0, 0, 255),  # Red
    "Disgust": (0, 128, 0),  # Dark Green
    "Fear": (255, 0, 255),  # Magenta
    "Happy": (0, 255, 255),  # Yellow
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (255, 255, 0),  # Cyan
    "Neutral": (128, 128, 128)  # Gray
}

BASE_RGB_COLOR = (255, 255, 255)  # White
DESIRED_WIDTH = 1000
SHOW_WEBCAM = True

def calculate_window_size(capture, width):
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height
    height = int(width / aspect_ratio)
    return width, height

def display_emotions(frame, face, emotions, width, height, show_webcam):
    if show_webcam:
        if face:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for idx, (emotion, score) in enumerate(emotions.items()):
                text_position = (x, y + h + 20 + idx * 20)
                cv2.putText(frame, f"{emotion}: {score:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Real-time Emotion Recognition', resized_frame)
    else:
        blank_image = 255 * np.ones((500, 500, 3), dtype=np.uint8)
        if emotions:
            for idx, (emotion, score) in enumerate(emotions.items()):
                cv2.putText(blank_image, f"{emotion}: {score:.2f}", (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Real-time Emotion Recognition', blank_image)


def main():
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(0)
    width, height = calculate_window_size(cap, DESIRED_WIDTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face, emotions, emotion_color = e2c.detect_emotions_and_convert_to_color(frame, detector, EMOTIONS_COLOR_MAP)
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:] = emotion_color if face else BASE_RGB_COLOR

        cv2.imshow('Emotion Color', color_image)
        display_emotions(frame, face, emotions, width, height, SHOW_WEBCAM)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
