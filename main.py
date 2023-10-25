import cv2
from fer import FER
import pandas as pd
import datetime
import numpy as np

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = 1000
aspect_ratio = original_width / original_height
new_height = int(new_width / aspect_ratio)

emotion_data = pd.DataFrame(columns=["Time", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])

baseRGB_Color = (255, 255, 255) # White

# Function to convert emotions to RGB colors
emotion_colors = {
    "Angry": (0, 0, 255),  # Red
    "Disgust": (0, 128, 0),  # Dark Green
    "Fear": (255, 0, 255),  # Magenta
    "Happy": (0, 255, 255),  # Yellow
    "Sad": (255, 0, 0),  # Blue
    "Surprise": (255, 255, 0),  # Cyan
    "Neutral": (128, 128, 128)  # Gray
}







# Set this flag to True if you want to display the webcam feed
show_webcam = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    result = detector.detect_emotions(frame)

    avg_color = np.array([0, 0, 0], dtype=np.float32)

    if result:
        emotions = result[0]["emotions"]
        for emotion, score in emotions.items():
            emotion_title = emotion.capitalize()
            avg_color += np.array(emotion_colors[emotion_title]) * score
        # Convert the average color to BGR format (OpenCV format)
        avg_color = avg_color.astype(np.uint8).tolist()

        # Create an image to display the average color
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:] = avg_color

        # Display the average color
        cv2.imshow('Emotion Color', color_image)

    if show_webcam:
        # Draw bounding boxes around faces and display the emotions
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the emotions
            for idx, (emotion, score) in enumerate(emotions.items()):
                cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y + h + 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Recognition', cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC))
    else:
        # Initialize a blank image to display emotions
        blank_image = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)

        # Display the emotions of the first detected face
        if result:
            emotions = result[0]["emotions"]
            for idx, (emotion, score) in enumerate(emotions.items()):
                cv2.putText(blank_image, f"{emotion}: {score:.2f}", (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2)

            # Append emotion data to the DataFrame
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # emotion_data = emotion_data.append({"Time": current_time, **emotions}, ignore_index=True)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Recognition', blank_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
