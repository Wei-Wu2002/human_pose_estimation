"""
姿态检测配置文件
集中管理所有配置参数
"""

# ==================== 基础配置 ====================

MODE = "COCO"  # 可选: "COCO", "MPI"
OUTPUT_PATH = "./output/fall_output_test"

FPS = 30

# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    "COCO": {
        "proto_file": "./models/pose/coco/pose_deploy_linevec.prototxt",
        "weights_file": "./models/pose/coco/pose_iter_440000.caffemodel"
    },
    "MPI": {
        "proto_file": "./models/pose/mpi/pose_deploy_linevec.prototxt",
        "weights_file": "./models/pose/mpi/pose_iter_160000.caffemodel"
    }
}

# ==================== 人体关键点配置 ====================
BODY_PARTS_CONFIG = {
    "COCO": {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17
    },
    "MPI": {
        "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14
    }
}

# ==================== 姿态连接配置 ====================
POSE_PAIRS_CONFIG = {
    "COCO": [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
    ],
    "MPI": [
        ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
        ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
        ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]
    ]
}

# ==================== 状态配置 ====================
STATUS_CONFIG = {
    "STANDING": 0,
    "FALLING": 5
}

# ==================== 几何参数配置 ====================
GEOMETRY_CONFIG = {
    "COS_THRESHOLD": 0.5 ** 0.5,  # 0.7071067811865476
    "NORM_LENGTH": 0,
    "NEW_POSITION": 0,
    "OLD_POSITION": -1
}

# ==================== 数据集配置 ====================
DATASET_CONFIG = {
    "UR_FALL_PATH": "./dataset/UR",
    "IMAGE_SUBDIR": "img",
    "DEPTH_SUBDIR": "depth",
    "LABEL_FILE": "urfall-cam0-falls.csv"
}