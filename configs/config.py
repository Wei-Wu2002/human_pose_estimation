from pathlib import Path

# ==================== ROOT path configuration ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

# ==================== MODEL path configuration ====================
YOLO_MODEL_DIR = MODEL_DIR / 'Yolo_Pose' / 'yolo11n-pose.pt'

# ==================== Dataset path configuration ====================
DATASET_DIR = PROJECT_ROOT / 'dataset'
URFD_DIR = DATASET_DIR / 'UR'
YouTube_DIR = DATASET_DIR / 'YouTube'


if __name__ == '__main__':
    print("PROJECT_ROOT: {}".format(PROJECT_ROOT))
    print("MODEL_DIR: {}".format(MODEL_DIR))
    print("YOLO_MODEL_DIR: {}".format(YOLO_MODEL_DIR))
    print("DATASET_DIR: {}".format(DATASET_DIR))
    print("URFD_DIR: {}".format(URFD_DIR))
    print("YouTube_DIR: {}".format(YouTube_DIR))

