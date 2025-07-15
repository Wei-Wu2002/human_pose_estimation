BODY_PARTS_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

BODY_CONNECTIONS_DRAW = {
    "legs": (
        [("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle")],
        (255, 0, 0)  # Red
    ),
    "arms": (
        [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")],
        (0, 255, 0)  # Green
    ),
    "head": (
        [("nose", "left_eye"), ("nose", "right_eye"),
         ("left_eye", "left_ear"), ("right_eye", "right_ear")],
        (0, 0, 255)  # Blue
    ),
    "torso": (
        [("left_shoulder", "right_shoulder"),
         ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"),
         ("left_hip", "right_hip")],
        (255, 255, 0)  # Cyan-Yellow
    )
}