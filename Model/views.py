from django.shortcuts import render
from django.http import HttpResponse
import joblib
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow import keras

def index(request):
    return render(request, 'index.html')

def btn1(request):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def extract_keypoints(results):
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([rh])

    Alphabet = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                         'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Dot', 'Back'])

    model = keras.models.load_model('action.h8')

    sequence = []
    sentence = []
    threshold = 0.90

    RTSP_URL = 'http://192.168.87.67:8080//video'

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
  
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(Alphabet[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if Alphabet[np.argmax(res)] != sentence[-1]:
                            sentence.append(Alphabet[np.argmax(res)])
                    else:
                        sentence.append(Alphabet[np.argmax(res)])

            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('SigPlanet_Screen', image)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')