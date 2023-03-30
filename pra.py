import cv2
import math
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

model = load_model('./emotion_detection_model.h5')
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

cap = cv2.VideoCapture(0)

degree_array = []

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    center_x = int(width / 2)
    center_y = int(height / 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
    faces_sorted_by_size = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    for idx, (x, y, w, h) in enumerate(faces_sorted_by_size, 1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_normalized = roi_resized / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)
        roi_input = np.expand_dims(roi_input, axis=-1)
        emotion_probabilities = model.predict(roi_input)[0]
        dominant_emotion_label = emotion_labels[np.argmax(emotion_probabilities)]

        cv2_rectangle_center = [int(x + w / 2), int(y + h / 2)]
        proportion_x = 40 / 700
        proportion_y = 50 / 150000
        real_x = (center_x - cv2_rectangle_center[0]) * proportion_x
        real_y = w * h * proportion_y
        real_r = (real_x ** 2 + real_y ** 2) ** 0.5
        sin_x = real_x / real_r
        if not -1 <= sin_x <= 1:
            continue
        rad = math.asin(sin_x)
        degree = round(math.degrees(rad), -1)

        text = str(1) if idx == 1 else str(2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + w + 10
        text_y = y + int(h / 2) + int(text_size[1] / 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        degree_average = 0
        count = 0

        if idx == 1:
            degree_array.append(degree)

        if idx == 1 and len(degree_array) > 5:
            degree_average = sum(degree_array) / len(degree_array)
            print(round(degree_average, -1))
            degree_array.clear()
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()