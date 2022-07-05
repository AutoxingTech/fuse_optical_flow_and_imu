#!/usr/bin/env python3
import math
from tkinter import SEL_FIRST
import rospy
import os
import roslaunch
import cv2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from copy import deepcopy
from geometry_msgs.msg import PointStamped, TwistWithCovarianceStamped
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_conjugate, quaternion_inverse, quaternion_multiply, random_quaternion

gravity = 9.8

class OdomRelay:
    """
    Relay Odom message. But reset initial pose to (0,0,0), for easier charting.
    """

    def __init__(
        self, input_topic: str, output_topic: str, speed_topic: str = None
    ) -> None:
        self.__sub = rospy.Subscriber(
            input_topic, Odometry, self.odom_callback, queue_size=100
        )
        self.__publisher = rospy.Publisher(output_topic, Odometry, queue_size=100)
        self.__first_position = None

        self.__speed_pub = None
        if speed_topic != None:
            self.__speed_pub = rospy.Publisher(
                speed_topic, TwistWithCovarianceStamped, queue_size=100
            )

        self.__last_cov = None
        self.__output_topic = output_topic
        self.__position = PointStamped()

    def odom_callback(self, data: Odometry):    
        if self.__first_position == None:
            self.__first_position = data.pose.pose.position

        new_odom = deepcopy(data)
        new_odom.pose.pose.position.x -= self.__first_position.x
        new_odom.pose.pose.position.y -= self.__first_position.y
        if self.__position.header.stamp != None:
            dt = (data.header.stamp - self.__position.header.stamp).to_sec()
            self.__position.point.x += dt * data.twist.twist.linear.x
            self.__position.point.y += dt * data.twist.twist.linear.y
        self.__position.header = data.header
        if self.__output_topic == "/wheel_odom":
            new_odom.pose.pose.position = self.__position.point
        self.__publisher.publish(new_odom)

        speed = TwistWithCovarianceStamped()
        speed.header = data.header

        if self.__last_cov == None or data.twist.covariance[0] - self.__last_cov <= 0:
            # TODO: optical flow has bias, we should find a way to estimate it
            bias = 1
            speed.twist.twist.linear.x = data.twist.twist.linear.x * bias
            speed.twist.twist.linear.y = data.twist.twist.linear.y * bias
        else:
            speed.twist.twist.linear.x = math.nan
            speed.twist.twist.linear.y = math.nan

        speed.twist.covariance[0] = data.twist.covariance[0]
        self.__last_cov = data.twist.covariance[0]

        if self.__speed_pub != None:
            self.__speed_pub.publish(speed)


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def correct_acc(quaternion, ori_acc):
    # rotate local frame to earth frame: q.v.q*
    q = quaternion
    qc = quaternion_conjugate(q)
    acc = [
            ori_acc[0],
            ori_acc[1],
            ori_acc[2],
            0
        ]
    t1 = quaternion_multiply(q, acc)
    rotated_acc = quaternion_multiply(t1, qc)
    # remove gravity
    rotated_acc[2] -= gravity
    # rotate earth frame to local frame: inv(q).v.(inv(q))*
    q_inv = quaternion_inverse(q)
    qc_inv = quaternion_conjugate(q_inv)
    t1 = quaternion_multiply(q_inv, rotated_acc)
    acc = quaternion_multiply(t1, qc_inv)
    return acc[0:3]

def correct_acc_with_matrix(quaternion, ori_acc):
    matrix = quaternion_matrix(quaternion)
    rotated_acc = matrix.dot(
        [
            ori_acc[0],
            ori_acc[1],
            ori_acc[2],
            1
        ]
    )
    # remove gravity
    rotated_acc[2] -= gravity
    matrix_inv = quaternion_matrix(quaternion_inverse(quaternion))
    new_acc = matrix_inv.dot(rotated_acc)
    return new_acc[0:3]


class Node:
    def __init__(self) -> None:
        self.__roll_publisher = rospy.Publisher("/rpy", PointStamped, queue_size=100)
        self.__predict_publisher = rospy.Publisher(
            "/predict_speed", TwistWithCovarianceStamped, queue_size=100
        )
        self._predict_pos = PointStamped()
        self.__predict_pos_publisher = rospy.Publisher(
            "/predict_pos", PointStamped, queue_size=100
        )
        self.__kalman_filter = kalman = cv2.KalmanFilter(4, 4, 0)

        kalman.processNoiseCov = 1e-3 * np.eye(4)
        kalman.errorCovPost = 1.0 * np.ones((4, 4))
        kalman.statePost = np.zeros((4, 1))

        self.__time: rospy.Time = None
        self.__publish_time : rospy.Time = None

    def __get_gravity_corrected_acc_and_rpy(self, imu: Imu):
        quaternion = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        acc = [imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z]
        acc = correct_acc(quaternion, acc)

        r, p, y = euler_from_quaternion(
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w
        )
        rpy = PointStamped()
        rpy.header = imu.header
        rpy.point.x = math.degrees(r)
        rpy.point.y = math.degrees(p)
        rpy.point.z = math.degrees(y)

        return acc, rpy

    def __imu_callback(self, imu: Imu):
        dt = self.__get_delta_time_secs()
        if dt == None:
            return

        # predict
        kalman = self.__kalman_filter
        kalman.transitionMatrix = np.array(
            [[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
        )
        prediction = kalman.predict()

        acc, rpy = self.__get_gravity_corrected_acc_and_rpy(imu)
        self.__roll_publisher.publish(rpy)

        # correct
        cov = 0.1
        kalman.measurementNoiseCov = np.array(
            [
                [cov * cov, 0],
                [0, cov * cov],
            ]
        )
        kalman.measurementMatrix = np.array(
            [
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )
        measurement = np.array(
            [
                [acc[0]],
                [acc[1]],
            ]
        )
        kalman.correct(measurement)

        self.__publish_result(imu.header, prediction, kalman.errorCovPost)

    def __optical_speed_callback(self, speed: TwistWithCovarianceStamped):
        if math.isnan(speed.twist.twist.linear.x) or speed.twist.covariance[0] > 0:
            pass
        else:
            dt = self.__get_delta_time_secs()
            if dt == None:
                return

            # predict
            kalman = self.__kalman_filter
            kalman.transitionMatrix = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            _ = kalman.predict()

            # correct
            cov = speed.twist.covariance[0]
            cov = 0.05 + (pow(2, cov) - 1)
            kalman.measurementNoiseCov = np.array(
                [
                    [cov * cov, 0],
                    [0, cov * cov],
                ]
            )
            kalman.measurementMatrix = np.array(
                [
                    [1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                ]
            )
            measurement = np.array(
                [
                    [speed.twist.twist.linear.x],
                    [speed.twist.twist.linear.y],
                ]
            )
            kalman.correct(measurement)

    def __get_delta_time_secs(self) -> float or None:
        now = rospy.Time.now()

        # first time call
        if self.__time == None:
            self.__time = now
            return None

        dt: rospy.Duration = now - self.__time
        self.__time = now
        return dt.to_sec()

    def __publish_result(self, header: Header, prediction, errorCovPost):
        # publish speed
        vx = prediction[0][0]
        vy = prediction[1][0]
        speed = TwistWithCovarianceStamped()
        speed.header = header
        speed.twist.twist.linear.x = vx
        speed.twist.twist.linear.y = vy
        # convert to standard covariance for charting
        speed.twist.covariance[0] = math.sqrt(errorCovPost[0][0])
        speed.twist.covariance[1 * 3 + 1] = math.sqrt(errorCovPost[1][1])
        self.__predict_publisher.publish(speed)

        now = rospy.Time.now()
        if self.__publish_time == None:
            self.__publish_time = now
            return
        dt = (now - self.__publish_time).to_sec()
        self.__publish_time = now

        # publish position
        self._predict_pos.header = header
        self._predict_pos.point.x += vx * dt
        self._predict_pos.point.y += vy * dt
        self.__predict_pos_publisher.publish(self._predict_pos)

    def run(self):
        rospy.init_node("optical_odom_test_node")

        r1 = OdomRelay("/odom", "/wheel_odom")
        r2 = OdomRelay("/optical_flow_odom", "/optical_odom", "/optical_speed")

        rospy.Subscriber("/imu", Imu, self.__imu_callback)
        rospy.Subscriber(
            "/optical_speed", TwistWithCovarianceStamped, self.__optical_speed_callback
        )

        rqt_plot_node = roslaunch.Node(
            "rqt_plot",
            "rqt_plot",
            args=" ".join(
                [
                    "/wheel_odom/pose/pose/position/y",
                    "/wheel_odom/twist/twist/linear/y",
                    "/optical_odom/pose/pose/position/y",
                    "/optical_odom/twist/covariance[0]",
                    "/optical_speed/twist/twist/linear/y",  # optical flow speed
                    "/imu/linear_acceleration/y",
                    "/rpy/point/x",
                    "/rpy/point/y",
                    # "/rpy/point/z",
                    "/predict_speed/twist/twist/linear/y",
                    # "/predict_speed/twist/covariance[4]",
                    "/predict_pos/point/y",
                    "/speed_cov/point/y",
                ]
            ),
            required=True,
        )

        rosbag_node = roslaunch.Node(
            "rosbag", "play", args=f"{os.getcwd()}/test_2022-06-22-18-07-35.bag"
            # "rosbag", "play", args=f"{os.getcwd()}/po2.bag"
        )

        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        launch.launch(rosbag_node)
        launch.launch(rqt_plot_node)
        launch.spin()


def main():
    node = Node()
    node.run()

def test():
    for i in range(1, 10):
        q = random_quaternion()
        v = np.random.random(3)
        v1 = correct_acc(q, v)
        v2 = correct_acc_with_matrix(q, v)
        if not np.allclose(v1, v):
            print(f"not close {v1}  {v}")
        if not np.allclose(v2, v):
            print(f"not close {v2}  {v}")    


if __name__ == "__main__":
    main()
