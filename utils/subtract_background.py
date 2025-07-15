import os

import cv2
import numpy as np

from configs.logger import logger


def subtract_background(args):
    # 检查并创建背景模型
    if not os.path.exists("maxFrame.npy") or not os.path.exists("minFrame.npy"):
        logger.info("Creating background model...")
        medianFrame, frames = get_median_frame(args)
        maxFrame, minFrame = get_range(frames, medianFrame)
        np.save("maxFrame", maxFrame)
        np.save("minFrame", minFrame)
    else:
        logger.info("Loading existing background model...")
        maxFrame = np.load("maxFrame.npy")
        minFrame = np.load("minFrame.npy")

    # 创建背景减法器
    sub = cv2.createBackgroundSubtractorMOG2(history=2 * fps, varThreshold=2, detectShadows=True)

    # 初始化背景减法器
    initFrame = cv2.imread("./output/medianFrame.png")
    if initFrame is not None:
        for i in range(10):
            sub.apply(initFrame)
    else:
        logger.warning("Could not load median frame for background subtraction initialization")


def get_median_frame(args):
    """计算背景中值帧"""
    filenames = os.listdir(args.bg)
    filenames.sort()
    frames = []
    for file in filenames:
        imgPath = os.path.join(args.bg, file)
        frame = cv2.imread(imgPath)
        if frame is not None:
            logger.info(f"Loaded background image: {imgPath}")
            frames.append(frame)
        else:
            logger.warning(f"Failed to load image: {imgPath}")

    if not frames:
        raise ValueError("No valid background images found!")

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.imwrite("./output/medianFrame.png", medianFrame)
    return medianFrame, frames


def get_range(frames, medianFrame):
    """计算背景范围"""
    old_frame = frames[0]
    divs = []
    for i in range(1, len(frames)):
        dframe = cv2.absdiff(frames[i], old_frame)
        old_frame = frames[i]
        divs.append(dframe)
    divMedianFrame = np.median(divs, axis=0).astype(dtype=np.uint8)
    maxFrame = medianFrame + divMedianFrame * 9
    minFrame = medianFrame - divMedianFrame * 9
    return maxFrame, minFrame
