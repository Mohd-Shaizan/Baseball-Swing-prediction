def auto_label(features):
    if features["arm_extension"] > 0.55 and features["hip_rotation"] > 0.04:
        return "Swing"
    if features["elbow_angle"] < 120:
        return "Hit"
    return "Strike"