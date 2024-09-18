import cv2
import dlib
from imutils import face_utils
from PIL import Image
import time
import json
from facepose import Facepose
from utils import crop_img, draw_axis, eye_aspect_ratio, mouth_aspect_ratio, rec_to_roi_box
from fer import FER  # Importer la bibliothèque de reconnaissance des émotions faciales

# Initialize face detection and landmarking
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def main():
    cap = cv2.VideoCapture(0)

    blinkCount = 0
    yawnCount = 0
    lostFocusCount = 0
    lostFocusDuration = 0
    focusTimer = None
    faceNotPresentDuration = 0
    faceTimer = None
    yawning = False
    eyeClosed = False
    lostFocus = False
    microsleep_start_time = None
    microsleep_detected = False

    frame_rate_use = 5
    prev = 0

    facepose = Facepose()
    emotion_detector = FER(mtcnn=True)  # Initialiser le détecteur d'émotions

    shape = None
    yaw_predicted, pitch_predicted, roll_predicted = None, None, None

    start_time = time.time()
    fatigue_warning = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate_use:
            prev = time.time()

            rects = detector(gray, 0)

            if len(rects) == 0:
                if faceTimer is None:
                    faceTimer = time.time()
                faceNotPresentDuration += time.time() - faceTimer
                faceTimer = time.time()

            for (i, rect) in enumerate(rects):
                faceTimer = None

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[36:42]
                rightEye = shape[42:48]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                mar = mouth_aspect_ratio(shape[60:69])

                if ear < 0.24:
                    eyeClosed = True
                    if microsleep_start_time is None:
                        microsleep_start_time = time.time()
                    elif time.time() - microsleep_start_time > 2:
                        microsleep_detected = True
                if ear > 0.24 and eyeClosed:
                    blinkCount += 1
                    eyeClosed = False
                    microsleep_start_time = None
                    microsleep_detected = False

                if mar > 0.31:
                    yawning = True
                if mar < 0.28 and yawning:
                    yawnCount += 1
                    yawning = False

                roi_box, center_x, center_y = rec_to_roi_box(rect)
                roi_img = crop_img(frame, roi_box)
                img = Image.fromarray(roi_img)

                (yaw_predicted, pitch_predicted, roll_predicted) = facepose.predict(img)

                if yaw_predicted < -30 or yaw_predicted > 30:
                    lostFocus = True
                    if focusTimer is None:
                        focusTimer = time.time()

                    lostFocusDuration += time.time() - focusTimer
                    focusTimer = time.time()
                if -30 < yaw_predicted < 30 and lostFocus:
                    lostFocusCount += 1
                    lostFocus = False
                    focusTimer = None

                # Détecter l'émotion
                emotion, emotion_score = emotion_detector.top_emotion(frame)

                # Afficher les émotions détectées
                if emotion:
                    cv2.putText(frame_display, f"Emotion: {emotion} ({emotion_score*100:.2f}%)", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Visualization (optional)
                for idx, (x, y) in enumerate(shape):
                    cv2.circle(frame_display, (x, y), 2, (255, 255, 0), -1)

            # Calculate blink rate
            elapsed_minutes = (time.time() - start_time) / 60
            blink_rate = blinkCount / elapsed_minutes if elapsed_minutes > 0 else 0

            # Detect fatigue based on blink rate, yawn count, and lost focus duration
            fatigue_warning =  yawnCount > 5 or lostFocusDuration > 300 or microsleep_detected

            # Display metrics on the frame
            cv2.putText(frame_display, f"Blink Count: {blinkCount}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Yawn Count: {yawnCount}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Lost Focus Count: {lostFocusCount}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Lost Focus Duration: {lostFocusDuration}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Face Not Present Duration: {faceNotPresentDuration}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Microsleep Detected: {'Yes' if microsleep_detected else 'No'}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame_display, f"Fatigue Warning: {'Yes' if fatigue_warning else 'No'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if fatigue_warning else (0, 255, 0), 1)

            if shape is not None and len(list(rects)) != 0:
                rect = list(rects)[0]
                draw_border(frame_display, (rect.left(), rect.top()), (rect.left() + rect.width(), rect.top() + rect.height()), (255, 255, 255), 1, 10, 20)
                draw_axis(frame_display, yaw_predicted, pitch_predicted, roll_predicted, tdx=int(center_x), tdy=int(center_y), size=100)
                for idx, (x, y) in enumerate(shape):
                    cv2.circle(frame_display, (x, y), 2, (255, 255, 0), -1)
                    if 36 <= idx < 48:
                        cv2.circle(frame_display, (x, y), 2, (255, 0, 255), -1)
                    elif 60 <= idx < 68:
                        cv2.circle(frame_display, (x, y), 2, (0, 255, 255), -1)

            cv2.imshow('frame', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


if __name__ == '__main__':
    main()
