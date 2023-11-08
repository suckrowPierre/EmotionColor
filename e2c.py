# e2c: emotion to color
import numpy as np


def color_summation(emotions, color_map, base_color):
    avg_color = np.array([0, 0, 0], dtype=np.float32)

    for emotion, score in emotions.items():
        emotion_title = emotion.capitalize()
        avg_color += np.array(color_map[emotion_title]) * score

    return avg_color


def emotion_detection_and_to_color(frame, detector, emotions_color_map, color_merge_function,
                                   base_color=(255, 255, 255)):
    result = detector.detect_emotions(frame)

    if result:
        face = result[0]["box"]
        emotions = result[0]["emotions"]
        emotion_color = color_merge_function(emotions, emotions_color_map, base_color)
        return face, emotions, emotion_color
    return None, None, None