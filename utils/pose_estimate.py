from configs.logger import logger
import cv2

new_position = 0
old_position = -1
norm_length = 0
cos = 0.5 ** 0.5

# 0 for standing, 5 for falling
status = 0

MODE = "COCO"

if MODE == "COCO":
    proto_file = "./models/OpenPose/pose/coco/pose_deploy_linevec.prototxt"
    weights_file = "./models/OpenPose/pose/coco/pose_iter_440000.caffemodel"

    # Body Parts attr（omit background: 18)
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

elif MODE == "MPI":
    proto_file = "./models/pose/mpi/pose_deploy_linevec.prototxt"
    weights_file = "./models/pose/coco/pose_iter_160000.caffemodel"

    # Body Parts attr （omit background: 15)
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
else:
    raise ValueError(f"Unknown MODE: {MODE}")

output_path = "./output/fall_output_test"
data_set = []
fps = 30

def load_network(name):
    if name == "openpose":
        return load_openpose()
    else:
        raise ValueError(f"Unsupported network: {name}")

def load_openpose():
    """加载OpenPose网络"""
    try:
        network = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        logger.info(f"Successfully loaded network from {proto_file} and {weights_file}")
        return network
    except cv2.error as e:
        logger.error(f"Error loading network: {e}")
        logger.error(f"Proto file: {proto_file}")
        logger.error(f"Weights file: {weights_file}")
        logger.error("Please check if the model files exist and are valid.")
        return None