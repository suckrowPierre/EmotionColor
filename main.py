import cv2
from fer import FER
import pandas as pd
import numpy as np
import e2c

emotions_color_map = {
    "Angry": (0, 0, 255),  # Red
    "Disgust": (0, 128, 0),  # Dark Green
    "Fear": (255, 0, 255),  # Magenta
    "Happy": (0, 255, 255),  # Yellow
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (255, 255, 0),  # Cyan
    "Neutral": (128, 128, 128)  # Gray
}

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)
new_width = 1000


def calculate_window_size(capture, desired_width):
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height
    calculated_height = int(desired_width / aspect_ratio)
    return desired_width, calculated_height


width, height = calculate_window_size(cap, new_width)

emotion_data = pd.DataFrame(columns=["Time", list(emotions_color_map.keys())])

baseRGB_Color = (255, 255, 255)  # White


def displayWebcamWithEmotions(frame, face, emotions, display_width, display_height):
    if face:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the emotions
        for idx, (emotion, score) in enumerate(emotions.items()):
            cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y + h + 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Recognition',
               cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_CUBIC))


def displayEmotionsWithoutWebcam(emotions, display_width, display_height):
    # Initialize a blank image to display emotions
    blank_image = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)

    # Display the emotions of the first detected face
    if face:
        for idx, (emotion, score) in enumerate(emotions.items()):
            cv2.putText(blank_image, f"{emotion}: {score:.2f}", (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

    cv2.imshow('Real-time Emotion Recognition', blank_image)
def getRGB_Color(color_map, emotion):
    return color_map[emotion.capitalize()]


# Set this flag to True if you want to display the webcam feed
show_webcam = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # e2c: emotion to color
    face, emotions, emotion_color = e2c.emotion_detection_and_to_color(frame, detector, emotions_color_map,
                                                                         e2c.color_summation)
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    if face:
        emotion_color = emotion_color.astype(np.uint8).tolist()
        color_image[:] = emotion_color
    else:
        # display base color
        color_image[:] = baseRGB_Color

    # Display the average color
    cv2.imshow('Emotion Color', color_image)


    if show_webcam:
        displayWebcamWithEmotions(frame, face, emotions, width, height)
    else:
        displayEmotionsWithoutWebcam(emotions, width, height)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
