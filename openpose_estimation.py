import argparse
import os
import traceback

from configs.logger import logger
from utils.pose_estimate import load_network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenPose Pose Estimation')
    parser.add_argument('--bg', default="./dataset/UR/img/bg",
                        help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--net', default="openpose", choices=["openpose"],
                        help='Select the backbone network to use')
    parser.add_argument('--test', default="./dataset/UR/img/fall-01-cam0-rgb")
    parser.add_argument('--thr', default=0.11, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--mode', default="fall", help='Choose the modes.')
    args = parser.parse_args()

    # 加载网络
    net = load_network(args.net)
    if net is None:
        logger.error("Failed to load network. Exiting.")
        exit(1)

    # 创建输出目录
    output = "./output"
    if not os.path.exists(output):
        os.mkdir(output)

    try:
        logger.info("Trying to load network...")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        traceback.print_exc()

