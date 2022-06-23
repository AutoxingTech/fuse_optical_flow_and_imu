#!/usr/bin/env python3
import math
import rospy
import os
import roslaunch
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from copy import deepcopy
from geometry_msgs.msg import PointStamped
import numpy as np
from tf.transformations import quaternion_matrix
import tf


class OdomRelay:
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
                speed_topic, PointStamped, queue_size=100
            )

        self.__last_cov = None

    def odom_callback(self, data: Odometry):
        if self.__first_position == None:
            self.__first_position = data.pose.pose.position

        new_odom = deepcopy(data)
        new_odom.pose.pose.position.x -= self.__first_position.x
        new_odom.pose.pose.position.y -= self.__first_position.y
        self.__publisher.publish(new_odom)

        speed = PointStamped()
        speed.header = data.header

        if self.__last_cov == None or data.twist.covariance[0] - self.__last_cov <= 0:
            speed.point.x = data.twist.twist.linear.x
            speed.point.y = data.twist.twist.linear.y
        else:
            speed.point.x = math.nan
            speed.point.y = math.nan
        speed.point.z = data.twist.covariance[0]
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


class Node:
    def __init__(self) -> None:
        self.__roll_publisher = rospy.Publisher("/rpy", PointStamped, queue_size=100)
        self.__predict_publisher = rospy.Publisher(
            "/predict", PointStamped, queue_size=100
        )
        self._predict_pos = PointStamped()
        self.__predict_pos_publisher = rospy.Publisher(
            "/predict_pos", PointStamped, queue_size=100
        )
        self.__kalman_filter = kalman = cv2.KalmanFilter(4, 4, 0)
        self.__cov_pub = rospy.Publisher("/speed_cov", PointStamped, queue_size=100)

        kalman.processNoiseCov = 1e-3 * np.eye(4)
        kalman.errorCovPost = 1.0 * np.ones((4, 4))
        kalman.statePost = np.zeros((4, 1))

        self.__time: rospy.Time = None

    def __get_gravity_corrected_acc_and_rpy(self, imu: Imu):
        matrix = quaternion_matrix(
            [
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w,
            ]
        )
        acc = matrix.dot(
            [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
                1,
            ]
        )

        r, p, y = euler_from_quaternion(
            imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w
        )
        rpy = PointStamped()
        rpy.header = imu.header
        rpy.point.x = r
        rpy.point.y = p
        rpy.point.z = y

        return acc, rpy

    def __imu_callback(self, imu: Imu):
        kalman = self.__kalman_filter

        now = rospy.Time.now()
        if self.__time == None:
            self.__time = now
            return

        dt = now - self.__time
        dt = dt.to_sec()
        self.__time = now
        kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        prediction = kalman.predict()

        acc, rpy = self.__get_gravity_corrected_acc_and_rpy(imu)
        self.__roll_publisher.publish(rpy)

        kalman.measurementNoiseCov = np.array(
            [
                [0.25 / 2, 0],
                [0, 0.25 / 2],
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

        speed = PointStamped()
        speed.header = imu.header
        speed.point.x = prediction[0][0]
        speed.point.y = prediction[1][0]

        self.__predict_publisher.publish(speed)

        self._predict_pos.header = imu.header
        self._predict_pos.point.x += speed.point.x * 0.01
        self._predict_pos.point.y += speed.point.y * 0.01
        self.__predict_pos_publisher.publish(self._predict_pos)

        cov = PointStamped()
        cov.header = imu.header
        cov.point.x = math.sqrt(kalman.errorCovPost[0][0])
        cov.point.y = math.sqrt(kalman.errorCovPost[1][1])
        self.__cov_pub.publish(cov)

    def __optical_speed_callback(self, speed: PointStamped):
        kalman = self.__kalman_filter

        if math.isnan(speed.point.x) or speed.point.z > 0:
            pass
        else:
            now = rospy.Time.now()
            if self.__time == None:
                self.__time = now
                return

            dt = now - self.__time
            dt = dt.to_sec()
            self.__time = now
            kalman.transitionMatrix = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

            prediction = kalman.predict()

            cov = speed.point.z
            cov = 0.05 + (pow(2, cov) - 1)
            kalman.measurementNoiseCov = np.array(
                [
                    [cov, 0],
                    [0, cov],
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
                    [speed.point.x],
                    [speed.point.y],
                ]
            )
            kalman.correct(measurement)

    def run(self):
        rospy.init_node("optical_odom_test_node")

        r1 = OdomRelay("/odom", "/wheel_odom")
        r2 = OdomRelay("/optical_flow_odom", "/optical_odom", "/optical_speed")

        rospy.Subscriber("/imu", Imu, self.__imu_callback)
        rospy.Subscriber("/optical_speed", PointStamped, self.__optical_speed_callback)

        rqt_plot_node = roslaunch.Node(
            "rqt_plot",
            "rqt_plot",
            args="/wheel_odom/pose/pose/position/y /optical_odom/pose/pose/position/y /optical_odom/twist/covariance[0] /optical_speed/point/y /imu/linear_acceleration/y /rpy/point/x /predict/point/y /predict_pos/point/y /speed_cov/point/y",
            required=True,
        )

        rosbag_node = roslaunch.Node(
            "rosbag", "play", args=f"{os.getcwd()}/test_2022-06-22-18-07-35.bag"
        )

        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        launch.launch(rosbag_node)
        launch.launch(rqt_plot_node)
        launch.spin()


def main():
    node = Node()
    node.run()


if __name__ == "__main__":
    main()
