import numpy as np
import math

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def extract_pose_features(lm):
    rs = (lm[12].x, lm[12].y)
    re = (lm[14].x, lm[14].y)
    rw = (lm[16].x, lm[16].y)

    lh = (lm[23].x, lm[23].y)
    rh = (lm[24].x, lm[24].y)

    ls = (lm[11].x, lm[11].y)

    elbow_angle = calculate_angle(rs, re, rw)
    arm_extension = np.linalg.norm(np.array(rs) - np.array(rw))
    hip_rotation = abs(lh[0] - rh[0])
    torso_rotation = abs(ls[0] - rs[0])

    return {
        "elbow_angle": elbow_angle,
        "arm_extension": arm_extension,
        "hip_rotation": hip_rotation,
        "torso_rotation": torso_rotation
    }