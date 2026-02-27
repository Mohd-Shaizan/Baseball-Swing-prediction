import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pose import extract_pose_features

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_img = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)

pose_vid = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= IMAGE PROCESSING =================

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, None, None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose_img.process(rgb)

    if not res.pose_landmarks:
        return img, None, None

    lm = res.pose_landmarks.landmark
    feats = extract_pose_features(lm)

    # Sanity check
    if feats["arm_extension"] < 0.10:
        return None, None, None

    skeleton = img.copy()
    mp_draw.draw_landmarks(
        skeleton,
        res.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    return img, skeleton, feats


# ================= LIVE WEBCAM =================

PRED_HISTORY = deque(maxlen=15)

def webcam_generator(model, scaler, le):
    cap = cv2.VideoCapture(0)

    FEATURE_ORDER = [
        "elbow_angle",
        "arm_extension",
        "hip_rotation",
        "torso_rotation"
    ]

    current_label = "Analyzing..."
    current_conf = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_vid.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            feats = extract_pose_features(lm)

            if feats["arm_extension"] > 0.18:
                X = scaler.transform([[feats[f] for f in FEATURE_ORDER]])
                probs = model.predict_proba(X)[0]
                PRED_HISTORY.append(probs)
                frame_count += 1

                if frame_count % 5 == 0 and len(PRED_HISTORY) >= 5:
                    avg = np.mean(PRED_HISTORY, axis=0)
                    idx = np.argmax(avg)
                    conf = avg[idx]

                    if conf > 0.6:
                        current_label = le.inverse_transform([idx])[0]
                        current_conf = int(conf * 100)
                    else:
                        current_label = "Uncertain"
                        current_conf = int(conf * 100)

        cv2.putText(
            frame,
            f"Shot: {current_label} ({current_conf}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        _, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

    cap.release()

def process_frame_raw(frame):
    """Processes a raw CV2 frame for video analysis."""
    if frame is None:
        return None, None
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_img.process(rgb)

    if not res.pose_landmarks:
        return None, None

    lm = res.pose_landmarks.landmark
    feats = extract_pose_features(lm)
    
    # Draw skeleton for the UI
    skeleton = frame.copy()
    mp_draw.draw_landmarks(
        skeleton,
        res.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    return skeleton, feats