import numpy as np

def color_summation(emotions, color_map):
    avg_color = np.array([0, 0, 0], dtype=np.float32)
    for emotion, score in emotions.items():
        avg_color += np.array(color_map[emotion.capitalize()]) * score
    return avg_color

def detect_emotions_and_convert_to_color(frame, detector, emotions_color_map, base_color=(255, 255, 255)):
    results = detector.detect_emotions(frame)
    if results:
        face = results[0]["box"]
        emotions = results[0]["emotions"]
        emotion_color = color_summation(emotions, emotions_color_map)
        return face, emotions, emotion_color.astype(np.uint8).tolist()
    return None, None, base_color