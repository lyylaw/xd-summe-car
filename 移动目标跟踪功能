#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.backends.cudnn as cudnn

class TargetTracker:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('target_tracker', anonymous=True)
        
        # 加载YOLOv5模型（需要提前下载模型权重）
        self.model = self.load_yolov5_model()
        self.class_names = self.model.names
        self.target_class = 'person'  # 要跟踪的目标类别
        
        # 颜色检测参数（跟踪红色目标）
        self.red_lower = np.array([0, 100, 100])
        self.red_upper = np.array([10, 255, 255])
        
        # 图像处理工具
        self.bridge = CvBridge()
        
        # 控制参数
        self.Kp = 0.01  # 比例控制系数
        self.max_linear_speed = 0.3  # 最大线速度 m/s
        self.max_angular_speed = 0.5  # 最大角速度 rad/s
        self.target_size_threshold = 0.2  # 目标在图像中的比例阈值
        
        # ROS订阅者和发布者
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 目标状态
        self.target_detected = False
        self.target_center = (0, 0)
        self.target_size = 0
        self.image_center = (0, 0)
        
        rospy.loginfo("Target Tracker initialized")

    def load_yolov5_model(self):
        """加载YOLOv5目标检测模型"""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        cudnn.benchmark = True  # 加速推理
        return model

    def detect_target(self, cv_image):
        """使用YOLOv5检测目标并筛选红色目标"""
        # 使用YOLOv5进行目标检测
        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]
        
        # 筛选目标类别（人）和颜色（红色）
        targets = []
        for _, det in detections.iterrows():
            if det['name'] == self.target_class and det['confidence'] > 0.5:
                # 提取目标区域
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                target_roi = cv_image[y1:y2, x1:x2]
                
                # 检查目标颜色（红色）
                if self.is_red_target(target_roi):
                    targets.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'size': (x2 - x1) * (y2 - y1)
                    })
        
        # 选择最大的目标进行跟踪
        if targets:
            largest_target = max(targets, key=lambda t: t['size'])
            return largest_target
        return None

    def is_red_target(self, roi):
        """检测区域是否为红色目标"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        
        # 计算红色像素比例
        red_ratio = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])
        
        return red_ratio > 0.3  # 如果红色像素超过30%，认为是红色目标

    def calculate_control_command(self):
        """根据目标位置计算控制命令"""
        cmd = Twist()
        
        if not self.target_detected:
            # 未检测到目标，原地旋转寻找
            cmd.angular.z = self.max_angular_speed * 0.5
            return cmd
        
        # 计算目标在图像中的水平偏移（像素）
        offset_x = self.target_center[0] - self.image_center[0]
        
        # 比例控制：角速度与水平偏移成正比
        cmd.angular.z = -self.Kp * offset_x
        
        # 根据目标大小调整线速度（目标越大表示越近，速度越慢）
        if self.target_size > self.target_size_threshold:
            cmd.linear.x = 0  # 目标足够近时停止
        else:
            cmd.linear.x = self.max_linear_speed * (1 - self.target_size / self.target_size_threshold)
        
        # 限制速度范围
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)
        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed), 0)
        
        return cmd

    def image_callback(self, msg):
        """处理摄像头图像的回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_center = (cv_image.shape[1] // 2, cv_image.shape[0] // 2)
            
            # 检测目标
            target = self.detect_target(cv_image)
            
            if target:
                self.target_detected = True
                self.target_center = target['center']
                self.target_size = target['size'] / (cv_image.shape[0] * cv_image.shape[1])
                
                # 在图像上标记目标
                x1, y1, x2, y2 = target['bbox']
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(cv_image, self.target_center, 5, (0, 0, 255), -1)
            else:
                self.target_detected = False
            
            # 显示处理结果
            cv2.imshow("Target Tracking", cv_image)
            cv2.waitKey(1)
            
            # 计算并发布控制命令
            cmd_vel = self.calculate_control_command()
            self.cmd_vel_pub.publish(cmd_vel)
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {str(e)}")

    def run(self):
        """主运行循环"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        tracker = TargetTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
