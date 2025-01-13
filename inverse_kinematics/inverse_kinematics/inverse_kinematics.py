import time
from typing import Any, Optional, Union

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from inverse_kinematics_interfaces.action import Posegoal
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import numpy as np
import sympy as sp
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor
import math
from multiprocessing import Pool
# import matplotlib.pyplot as plt
import plotly.graph_objects as go


class UB100:
    LINK_LENGTHS = [0, 0.73731, 0.3878, 0, 0, 0]  # a_i
    LINK_TWISTS = [float(sp.pi / 2), 0, 0, float(sp.pi / 2), float(sp.pi / 2), float(sp.pi / 2)]  # alpha_i
    LINK_OFFSETS = [0.760, 0.0, 0.0, 0.850, 0.760, 0.15878]  # d_i

    JOINT_ANGLES_INITIAL = [
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
    ]
    LAST_ANGLES = [
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
    ]

    theta = sp.symbols("theta1 theta2 theta3 theta4 theta5 theta6")
    T_0_1 = sp.Matrix([])
    T_0_2 = sp.Matrix([])
    T_0_3 = sp.Matrix([])
    T_0_4 = sp.Matrix([])
    T_0_5 = sp.Matrix([])
    T_0_6 = sp.Matrix([])

    P_X_Y_Z_0 = sp.Matrix([])

    T_0_E = np.array(
        [
            [1.0, 0.0, 0.0, -0.1405],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.42391],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    dist = sp.symbols("dist")
    steps = 20

    # total_time = dist/0.15
    dt = dist / (0.15 * steps)
    dt_calculated = 0

    def get_dh_transformations(self, theta, i):
        return sp.Matrix(
            [
                [
                    sp.cos(theta),
                    -sp.sin(theta) * sp.cos(self.LINK_TWISTS[i]),
                    sp.sin(theta) * sp.sin(self.LINK_TWISTS[i]),
                    self.LINK_LENGTHS[i] * sp.cos(theta),
                ],
                [
                    sp.sin(theta),
                    sp.cos(theta) * sp.cos(self.LINK_TWISTS[i]),
                    -sp.cos(theta) * sp.sin(self.LINK_TWISTS[i]),
                    self.LINK_LENGTHS[i] * sp.sin(theta),
                ],
                [0, sp.sin(self.LINK_TWISTS[i]), sp.cos(self.LINK_TWISTS[i]), self.LINK_OFFSETS[i]],
                [0, 0, 0, 1],
            ]
        )

    def compute_derivative(self, args):
        jacobian_, theta_i = args
        return sp.diff(jacobian_, theta_i)

    def get_inverse_jacobian(self, joint_angles):
        # Define the Jacobian matrix symbolically (your actual Jacobian expressions)
        jacobian_ = sp.expand(self.T_0_6[0:3, 3])

        #  All independent steps can parallelize
        Z1 = self.T_0_1[0:3, 2]
        Z2 = self.T_0_2[0:3, 2]
        Z3 = self.T_0_3[0:3, 2]
        Z4 = self.T_0_4[0:3, 2]
        Z5 = self.T_0_5[0:3, 2]
        Z6 = self.T_0_6[0:3, 2]
        # p1 = sp.diff(jacobian_, self.theta[0])
        # p2 = sp.diff(jacobian_, self.theta[1])
        # p3 = sp.diff(jacobian_, self.theta[2])
        # p4 = sp.diff(jacobian_, self.theta[3])
        # p5 = sp.diff(jacobian_, self.theta[4])
        # p6 = sp.diff(jacobian_, self.theta[5])


        args_list = [(jacobian_, theta_i) for theta_i in self.theta]

        with Pool() as pool:
            results = pool.map(self.compute_derivative, args_list)

        # Unpack results into individual variables (if needed)
        p1, p2, p3, p4, p5, p6 = results

        ## Obtaining Jacobian Matrix J

        J1 = sp.Matrix.hstack(p1, p2, p3, p4, p5, p6)
        J2 = sp.Matrix.hstack(Z1, Z2, Z3, Z4, Z5, Z6)
        J = sp.Matrix.vstack(J1, J2)

        # sp.pprint(J)
        # Substitute joint angles
        jacobian_computed = J.subs(
            [(self.theta[0], joint_angles[0]), (self.theta[1], joint_angles[1]), (self.theta[2], joint_angles[2]),
             (self.theta[3], joint_angles[3]), (self.theta[4], joint_angles[4]), (self.theta[5], joint_angles[5])])

        # Return the pseudo-inverse
        return jacobian_computed.pinv()

    def get_joint_angular_velocities_ik(self, end_effector_velocities, joint_angles):
        # Get the inverse Jacobian and calculate joint angular velocities
        jacobian_inv = self.get_inverse_jacobian(joint_angles)
        joint_velocities = jacobian_inv * end_effector_velocities
        return joint_velocities

    def get_end_effector_positions_fk(self, joint_angles):
        T_0_6_computed = self.T_0_6.subs({self.theta[i]: joint_angles[i] for i in range(6)})
        P_X_Y_Z_0 = T_0_6_computed[:, 3]
        return P_X_Y_Z_0

    def __init__(self):
        #  No need to parallelize as its executed only once at startup
        T_0_1 = self.get_dh_transformations(self.theta[0] + sp.pi / 2, 0)
        T_1_2 = self.get_dh_transformations(self.theta[1] + sp.pi / 2, 1)
        T_2_3 = self.get_dh_transformations(self.theta[2], 2)
        T_3_4 = self.get_dh_transformations(self.theta[3] + sp.pi / 2, 3)
        T_4_5 = self.get_dh_transformations(self.theta[4] + sp.pi, 4)
        T_5_6 = self.get_dh_transformations(self.theta[5] + sp.pi, 5)
        #  No need to parallelize as its executed only once at startup
        self.T_0_1 = T_0_1
        self.T_0_2 = T_0_1 * T_1_2
        self.T_0_3 = T_0_1 * T_1_2 * T_2_3
        self.T_0_4 = T_0_1 * T_1_2 * T_2_3 * T_3_4
        self.T_0_5 = T_0_1 * T_1_2 * T_2_3 * T_3_4 * T_4_5
        self.T_0_6 = T_0_1 * T_1_2 * T_2_3 * T_3_4 * T_4_5 * T_5_6

        self.P_X_Y_Z_0 = self.T_0_6[:-1, -1]

    # This function generates straight line trajectory in 3D space. There could be some cases where trajectory passes through the robot which should be handled somehow
    def generate_trajectory(self, point_a: Pose, point_b: Posegoal.Goal):

        print(f"Start point: x={point_a.position.x}, y={point_a.position.y}, z={point_a.position.z}")
        print(f"Desired point: x={point_b.pose.position.x}, y={point_b.pose.position.y}, z={point_b.pose.position.z}")


        rot = self.quaternion_rotation_matrix(point_a.orientation)
        rot = np.vstack((rot, np.array([0, 0, 0])))
        T_0_E = np.hstack((rot, np.array([[point_a.position.x], [point_a.position.y], [point_a.position.z], [1]])))

        distance = math.sqrt(
            (point_a.position.x - point_b.pose.position.x) ** 2 +
            (point_a.position.y - point_b.pose.position.y) ** 2 +
            (point_a.position.z - point_b.pose.position.z) ** 2
        )

        self.dt_calculated = self.dt.subs({self.dist: distance})
        total_time = self.dt_calculated * 20
        time = np.arange(0, total_time, self.dt_calculated)

        points = []
        # TODO verify traj generation
        for t in time:
            x = point_a.position.x + (point_b.pose.position.x - point_a.position.x) * t / total_time
            y = point_a.position.y + (point_b.pose.position.y - point_a.position.y) * t / total_time
            z = point_a.position.z + (point_b.pose.position.z - point_a.position.z) * t / total_time

            local_xy = np.array([x, y, z, 1.0])

            # points.append(np.dot(T_0_E, local_xy))
            points.append(local_xy)

        xx = []
        yy = []
        zz = []
        for pen in points:
            xx.append(float(pen[0]))
            yy.append(float(pen[1]))
            zz.append(float(pen[2]))

        fig = go.Figure(data=[go.Scatter3d(x=np.array(xx), y=np.array(yy), z=np.array(zz), mode='markers')])

        fig.show()

        return np.vstack(points)

    # This functions generates joint angles to be published
    def get_trajectory_ik(self, current_EE_pose: Pose, desired_EE_pose: Posegoal.Goal):
        joint_angles = []
        traj = self.generate_trajectory(current_EE_pose, desired_EE_pose)

        #find current joint angles either by doing ik of current pose or save last joint angles.
        joint_angles_current = sp.Matrix(self.LAST_ANGLES)

        for i in range(traj.shape[0]):
            if i > 0:
                x_dot, y_dot, z_dot = (
                    (traj[i, 0] - traj[i - 1, 0]) / self.dt_calculated,
                    (traj[i, 1] - traj[i - 1, 1]) / self.dt_calculated,
                    (traj[i, 2] - traj[i - 1, 2]) / self.dt_calculated,
                )
            else:
                x_dot, y_dot, z_dot = 0.0, 0.0, 0.0
            new_end_effector_velocities = sp.Matrix([x_dot, y_dot, z_dot, 0, 0, 0])
            joint_angular_velocities = self.get_joint_angular_velocities_ik(
                new_end_effector_velocities, joint_angles_current
            )

            joint_angles.append(joint_angles_current)

            # normalize all angles from -3.14 to 3.14
            joint_angles_current = joint_angles_current + joint_angular_velocities * self.dt_calculated

        self.LAST_ANGLES = joint_angles[-1]

        # reset dt_calculated
        self.dt_calculated = 0.0
        return joint_angles

    def quaternion_rotation_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q.w
        q1 = Q.x
        q2 = Q.y
        q3 = Q.z

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix


class InverseKinematics(Node):

    def __init__(self):
        """Constructor"""
        super().__init__("inverse_kinematics")
        self.action_server = None
        self.publisher_findger_left = None
        self.publisher_joint_4 = None
        self.publisher_joint_6 = None
        self.publisher_joint_5 = None
        self.publisher_joint_3 = None
        self.publisher_joint_2 = None
        self.publisher_joint_1 = None
        self.publisher_finger_right = None
        self.subscriber = None
        self.publisher = None
        self.currentPose = None
        # Setup robot first
        self.robot = UB100()
        self.setup()
        self.server_busy = False
        self.cancel_request = False
        self.precision_tolerance = 0.02

    def setup(self):
        """Sets up subscribers, publishers, etc. to configure the node"""

        # subscriber for handling incoming messages
        self.subscriber = self.create_subscription(Pose,
                                                   "/end_effector_pose",
                                                   self.topicCallback,
                                                   qos_profile=10)
        self.get_logger().info(f"Subscribed to '{self.subscriber.topic_name}'")

        # publisher for publishing outgoing messages
        self.publisher_joint_1 = self.create_publisher(Float64,
                                               "/joint_angles/joint1",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_1.topic_name}'")

        # publisher for publishing outgoing messages
        self.publisher_joint_2 = self.create_publisher(Float64,
                                               "/joint_angles/joint2",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_2.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_joint_3 = self.create_publisher(Float64,
                                               "/joint_angles/joint3",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_3.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_joint_4 = self.create_publisher(Float64,
                                               "/joint_angles/joint4",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_4.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_joint_5 = self.create_publisher(Float64,
                                               "/joint_angles/joint5",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_5.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_joint_6 = self.create_publisher(Float64,
                                               "/joint_angles/joint6",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_joint_6.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_findger_left = self.create_publisher(Float64,
                                               "/joint_angles/gripper_left_joint",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_findger_left.topic_name}'")        # publisher for publishing outgoing messages
        self.publisher_finger_right = self.create_publisher(Float64,
                                               "/joint_angles/ripper_right_joint",
                                               qos_profile=10)
        self.get_logger().info(f"Publishing to '{self.publisher_finger_right.topic_name}'")
        # action server for handling action goal requests
        self.action_server = ActionServer(
            self,
            Posegoal,
            "/goal",
            callback_group=ReentrantCallbackGroup(),
            execute_callback=self.actionExecute,
            goal_callback=self.actionHandleGoal,
            cancel_callback=self.actionHandleCancel,
            handle_accepted_callback=self.actionHandleAccepted
        )

    def topicCallback(self, msg: Pose):
        """Processes messages received by a subscriber

        Args:
            msg (Pose): message
        """
        self.currentPose = msg
        self.get_logger().info(f"Message received: '{msg.position.x}' '{msg.position.y}' '{msg.position.z}'")

    def workspaceSanityCheck(self, goalPose: Posegoal.Goal):
        self.get_logger().info(
            f"Checking point: x={goalPose.pose.position.x}, y={goalPose.pose.position.y}, z={goalPose.pose.position.z}")

        max_reach = sum(self.robot.LINK_LENGTHS) + sum(self.robot.LINK_OFFSETS)

        # Calculate distance to the point
        point_position = np.array([goalPose.pose.position.x, goalPose.pose.position.y, goalPose.pose.position.z])
        distance_to_point = np.linalg.norm(point_position)

        self.get_logger().info(f"Distance to point: {distance_to_point}, Max reach: {max_reach}")

        # Check if the distance is within the manipulator's reach
        if distance_to_point >= max_reach or goalPose.pose.position.z < 0:
            self.get_logger().warn("Point is outside the reachable workspace.")
            return False

        self.get_logger().info("Point is within the reachable workspace.")
        return True

    def actionHandleGoal(self, goal: Posegoal.Goal) -> GoalResponse:
        """Processes action goal requests

        Args:
            goal (Posegoal.Goal): action goal

        Returns:
            GoalResponse: goal response
        """

        self.get_logger().info("Received action goal request")
        # Check if robot is in action or one action is already running and sanity check

        if self.server_busy:
            self.get_logger().error("Server busy")
            return GoalResponse.REJECT
        elif self.workspaceSanityCheck(goal):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def actionHandleCancel(self, goal: Posegoal.Goal) -> CancelResponse:
        """Processes action cancel requests

        Args:
            goal (Posegoal.Goal): action goal

        Returns:
            CancelResponse: cancel response
        """
        self.server_busy = False
        self.cancel_request = True
        self.get_logger().warn("Received request to cancel action goal")

        return CancelResponse.ACCEPT

    def actionHandleAccepted(self, goal: Posegoal.Goal):
        """Processes accepted action goal requests

        Args:
            goal (Posegoal.Goal): action goal
        """

        # execute action in a separate thread to avoid blocking
        self.server_busy = True
        goal.execute()

    async def actionExecute(self, goal: Posegoal.Goal) -> Posegoal.Result:
        """Executes an action

        Args:
            goal (Posegoal.Goal): action goal

        Returns:
            Posegoal.Result: action goal result
        """
        # create the action feedback and result
        feedback = Posegoal.Feedback()
        result = Posegoal.Result()
        joint_angle = Float64()

        self.get_logger().info("Executing action goal")
        angles = self.robot.get_trajectory_ik(self.currentPose, goal.request)

        xx = []
        yy = []
        zz = []


        # verification
        for angle in angles:
            pose  = self.robot.get_end_effector_positions_fk([float(angle[0]), float(angle[1]), float(angle[2]), float(angle[3]), float(angle[4]), float(angle[5])])
            xx.append(float(pose[0]))
            yy.append(float(pose[1]))
            zz.append(float(pose[2]))

        fig = go.Figure(data=[go.Scatter3d(x=np.array(xx), y=np.array(yy), z=np.array(zz), mode='markers')])

        fig.show()

        for a in angles:
            #  pick one angle , publish it and wait for robot EE to reach that position, publish another and repeat
            feedback.next_joint_angles = [float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5])]

            joint_angle.data = float(a[0])
            self.publisher_joint_1.publish(joint_angle)
            joint_angle.data = float(a[1])
            self.publisher_joint_2.publish(joint_angle)
            joint_angle.data = float(a[2])
            self.publisher_joint_3.publish(joint_angle)
            joint_angle.data = float(a[3])
            self.publisher_joint_4.publish(joint_angle)
            joint_angle.data = float(a[4])
            self.publisher_joint_5.publish(joint_angle)
            joint_angle.data = float(a[5])
            self.publisher_joint_6.publish(joint_angle)

            goal.publish_feedback(feedback)

            approx_EE_fk = self.robot.get_end_effector_positions_fk(
                [float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5])])

            # TODO improve this
            while not self.is_pose_near_point(self.currentPose, approx_EE_fk, self.precision_tolerance):
                self.get_logger().info(
                    "Waiting for robot to reach %s %s %s" % (approx_EE_fk[0], approx_EE_fk[1], approx_EE_fk[2]))
                # checking if cancel request is generated
                if self.cancel_request:
                    self.get_logger().warn("Cancel request , Stopping arm")
                    result.success = True
                    goal.succeed()
                    self.server_busy = False
                    self.cancel_request = False
                    self.get_logger().info("Goal Halted")
                    return result

                time.sleep(1)

            self.get_logger().info("Robot reached EE pose x=%s y=%s z=%s desired pose x=%s y=%s z=%s" % (
                self.currentPose.position.x, self.currentPose.position.y,
                self.currentPose.position.z, approx_EE_fk[0], approx_EE_fk[1], approx_EE_fk[2]))

        # finish by publishing the action result
        # finish when EE is in vicinity of past goal index
        if rclpy.ok():
            result.success = True
            goal.succeed()
            self.server_busy = False
            self.get_logger().info("Goal succeeded")

        return result

    def is_pose_near_point(self, pose: Pose, point: list, tolerance: float) -> bool:
        """
        Checks if a given pose is near a specified point within a given tolerance.

        Args:
            pose (Pose): The geometry_msgs/Pose to check.
            point (list): The point to check against, as a list [x, y, z].
            tolerance (float): The distance tolerance in meters.

        Returns:
            bool: True if the pose is near the point, False otherwise.
        """
        distance = math.sqrt(
            (pose.position.x - point[0]) ** 2 +
            (pose.position.y - point[1]) ** 2 +
            (pose.position.z - point[2]) ** 2
        )
        print("distance ", distance)
        return distance <= tolerance


def main():
    rclpy.init()
    node = InverseKinematics()
    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(node, executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
