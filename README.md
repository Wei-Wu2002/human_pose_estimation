# README

## pose estimation pipeline
1. 输入图像序列
3. 背景建模 (get_median_frame + get_range)
5. 背景减法器初始化 (MOG2)
7. 循环处理每张图像:
9. 人体检测 (get_mask_dots_sub)
11. 区域裁剪 (get_crop_frame)
13. OpenPose姿态估计 (apply_openpose)
15. 关键点提取和坐标转换
17. 几何特征计算 (get_square + get_width_height_ratio)
19. 跌倒检测 (fall_detect)
21. 结果绘制 (draw_position)
23. 保存结果图像和姿态数据
