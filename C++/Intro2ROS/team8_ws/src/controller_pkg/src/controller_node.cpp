#include <ros/ros.h>

#include <ros/console.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <mav_msgs/Actuators.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>
#include <math.h>
#include <std_msgs/Float64.h>

#include <geometry_msgs/Twist.h>  // for keyboard control
#include <tuple>                  // for returning multiple values from a function
#include <stdlib.h>               // abs

#define PI M_PI

float ControlLoopRate = 1.0;
// const float ControlLoopRate = 100.0;
// Constant control parameters
float K_P_yaw = 1;  // max value pi*1000=3145
const int SignCorrectionYaw = -1;
// Used for three-point controller
const float YawTol = 0.1745;  // 10 deg = 0.1745 rad
// Init Controll Tolorance
const float InitTol = 0.00001;
// Used for position controller
const float K_P_x = 20;
// Used for velocity controller
const float K_P_v = 1;
// Duration (in sec) for forward backward wiggle motion to turn on spot
// With this setting a turning rate of ca. 15 °/s can be achieved
const double wiggle_time_period = 0.05;
// State Models Switcher1 (0:Normal Drive; 1:Tuning)
int state_M = 0;
// State Models Switcher2 (0:Normal Drive; 1:vertical Oscillation)
int state_M2 = 0;
// Counter for Model 2, 30 Timesteps no change -> Model 2
int count_M2 = 0;
// Every 50 Times Pulishe one times ROS_INFO
int iter_info = 0;
// epsilon
const float epsilon = 0.01;
// Desired State (x, y) in last Timestep
// Tolerance for compare the two desired states
float xl = 0;
float yl = 0;
const float dsTol = 0.0005;
// Counter for Oscillation (0 or 1)
int count_M22 = 0;

#include <eigen3/Eigen/Dense>

// If you choose to use Eigen, tf provides useful functions to convert tf
// messages to eigen types and vice versa, have a look to the documentation:
// http://docs.ros.org/melodic/api/eigen_conversions/html/namespacetf.html
#include <eigen_conversions/eigen_msg.h>

class controllerNode{
  ros::NodeHandle nh;

  ros::Subscriber desired_state, current_state, command_speed;
  ros::Publisher prop_speeds;
  ros::Timer timer;

  // Controller internals (you will have to set them below)
  // Current state
  // current position of the robot's c.o.m. in the world frame
  Eigen::Vector3d x;
  // current velocity of the robot's c.o.m. in the world frame
  Eigen::Vector3d v;
  // current orientation of the robot
  Eigen::Matrix3d R;
  // current angular velocity of the robot's c.o.m. in the *body* frame
  Eigen::Vector3d omega;
  // current yaw angle
  double yaw;

  // Desired state
  // desired position of the robot's c.o.m. in the world frame
  Eigen::Vector3d xd;
  // desired velocity of the robot's c.o.m. in the world frame
  Eigen::Vector3d vd;
  // desired acceleration of the robot's c.o.m. in the world frame
  Eigen::Vector3d ad;
  // desired yaw angle
  double yawd;

  // Control variables
  // error of the position
  Eigen::Vector3d xe;
  Eigen::Vector3d x_Re;
  double x_control = 0.0;
  // error of the velocity
  Eigen::Vector3d ve;
  Eigen::Vector3d v_Re;
  double vel_control = 0.0;
  // error of the yaw angle
  double yaw_error = 0.0;
  double yaw_rate_control = 0.0;

  // keyboard control
  double vel_cmd = 0.0;
  double turn_cmd = 0.0;
  double amp_cmd = 0.0;
  double vel4_cmd = 0.0;
  // For on spot turning
  ros::Time time_a = ros::Time::now();
  // Wiggle movement direction
  int sign = 1;

  double hz;             // frequency of the main control loop

 public:
  controllerNode():hz(ControlLoopRate) {
    // Subscribe to desired state and curent state
    // desired_state = nh.subscribe(
    //   "desired_state",
    //   1,
    //   &controllerNode::onDesiredState,
    //   this);
    // Subscribe to trajectory from local planner
    // /stdr_move_base/TrajectoryPlannerROS/local_plan
    // -> use corresponding ros topic
    desired_state = nh.subscribe(
      "stdr_move_base/TrajectoryPlannerROS/global_plan",
      1,
      &controllerNode::onDesiredState,
      this);
    command_speed = nh.subscribe(
      "cmd_vel",
      1,
      &controllerNode::onCommandState,
      this);
    current_state = nh.subscribe(
      "current_state_est",
      1,
      &controllerNode::onCurrentState,
      this);
    // Publish control commands
    prop_speeds = nh.advertise<mav_msgs::Actuators>("rotor_speed_cmds", 1);
    timer = nh.createTimer(ros::Rate(hz), &controllerNode::controlLoop, this);
  }

  void onDesiredState(
    const nav_msgs::Path& des_state) {
      // Position
      xd << des_state.poses[30].pose.position.x,
         des_state.poses[30].pose.position.y,
         des_state.poses[30].pose.position.z;
      // ROS_INFO_NAMED("onDesiredState", "VEL: %f", xd(0));

      // Velocities
      // geometry_msgs::Vector3 v = des_state.velocities[0].linear;
      // vd << v.x, v.y, v.z;
      // ROS_INFO_NAMED("onDesiredState", "VEL: %f %f %f", v.x, v.y, v.z);

      // Accelerations
      // geometry_msgs::Vector3 a = des_state.accelerations[0].linear;
      // ad << a.x, a.y, a.z;

      tf::Quaternion q;
      tf::quaternionMsgToTF(des_state.poses[30].pose.orientation, q);
      yawd = tf::getYaw(q);
      
  }

  void onCurrentState(const nav_msgs::Odometry& cur_state) {
    x << cur_state.pose.pose.position.x,
      cur_state.pose.pose.position.y,
      cur_state.pose.pose.position.z;
    v << cur_state.twist.twist.linear.x,
      cur_state.twist.twist.linear.y,
      cur_state.twist.twist.linear.z;
    omega << cur_state.twist.twist.angular.x,
      cur_state.twist.twist.angular.y,
      cur_state.twist.twist.angular.z;
    // Eigen::Quaterniond q;
    // tf::quaternionMsgToEigen(cur_state.pose.pose.orientation, q);
    // R = q.toRotationMatrix();
    tf::Quaternion qd;
    tf::quaternionMsgToTF(cur_state.pose.pose.orientation, qd);
    yaw = tf::getYaw(qd);

    // Rotate omega
    omega = R.transpose()*omega;
  }
  
  // Not used in final version, for velocity control 
  void onCommandState(const geometry_msgs::Twist& command_speed) {
      vel_cmd = (command_speed.linear.x
        + command_speed.linear.y
        + command_speed.linear.z)*2;
      turn_cmd = command_speed.angular.z*-4;

      // For testing of on spot turning (keys j and l)
      if (vel_cmd == 0 && turn_cmd != 0) {
        std::tie(turn_cmd, vel_cmd) = OnSpotYawControl(turn_cmd, 0);
      }
  }

  // Yaw Control
  // Orientation difference bigger than 32 degree -> Tuning Model -> state_M := 1
  // Else adaptive tuning model
  double YawControl(double yaw_desired, double yaw_measured) {

    yaw_error = yaw_desired - yaw_measured;
    if (abs(yaw_error) <= YawTol) {
      yaw_rate_control = 0;
    } else if (yaw_error < -YawTol) {
      yaw_rate_control = 1.2;   // Rotate rights
    } else if (yaw_error > YawTol) {
      yaw_rate_control = -1.2;  // Rotate left
    } else {
      yaw_rate_control = 0;
    }
    
    if (abs(yaw_error) >= 3.2*YawTol){
      state_M = 1;
      K_P_yaw = 4;
    } else if (abs(yaw_error) >= 1.6*YawTol){
      state_M = 0;
      K_P_yaw = 1;
    } else if (abs(yaw_error) >= 1*YawTol){
      state_M = 0;
      K_P_yaw = 0.8;
    } else {
      state_M = 0;
      K_P_yaw = 0.5;
    }

    return yaw_rate_control;
  }

  // P-Control of position
  double PosControl(
    Eigen::Vector3d x_desired,
    Eigen::Vector3d x_measured,
    Eigen::Matrix3d R_measured) {
      xe = x_desired - x_measured;  // position error vector in world frame
      x_control = K_P_x*(abs(xe(0))+abs(xe(1)));
      return x_control;
  }

  // P-Control of velocity
  // Not used in final version, for velocity control 
  double VelControl(
    Eigen::Vector3d v_desired,
    Eigen::Vector3d v_measured,
    Eigen::Matrix3d R_measured) {
      ve = v_desired - v_measured;  // velocity error vector in world frame
      v_Re = R*ve;                  // velocity error vector in robot frame
      // x-component of the velocity error in robot frame used for P control
      vel_control = K_P_v*v_Re(0);
      return vel_control;
  }

  // Not used in final version, for velocity control 
  std::tuple<double, double> OnSpotYawControl(
    double yaw_desired,
    double yaw_measured) {
      // Three-Point Controller for yaw rate
      yaw_error = yaw_desired - yaw_measured;
      if (yaw_error <= YawTol && yaw_error >= -YawTol) {
        yaw_rate_control = 0;
      } else if (yaw_error < -YawTol) {
        yaw_rate_control = -4;  // Rotate left
      } else if (yaw_error > YawTol) {
        yaw_rate_control = 4;  // Rotate rights
      } else {
        yaw_rate_control = 0;
      }

    return std::make_tuple(yaw_rate_control, vel_control = sign);
  }

  // Main control part
  void controlLoop(const ros::TimerEvent& t) {
    mav_msgs::Actuators msg;

    // Calculate for desired orientation based on desired and current position
    // output interval for "atan" is [-pi/2, pi/2]
    // but in this project we need an interval for about [-0.4*pi/2, 1.6*pi/2]
    // so we made a mapping from [-pi/2, -0.4*pi/2] to [pi/2, 1.6*pi/2]
    yawd = -atan((xd(0)-x(0))/(xd(1)-x(1)+epsilon)); // epsilon for not too small denominator
    if (yawd > 0.4) {
      yawd = -1.57 - (1.57-yawd);
    }
    
    // Caulculate the basic control parameter
    turn_cmd = YawControl(yawd, yaw);
    vel_cmd = PosControl(xd, x, R);

    // Check for "cross the obstacle" Mode -> if true, state_M2 := 1
    // Basic idea here is that, position of robot doesn't change in 30 iterations (<dsTol)
    // Initial Fall of robot need to be excluded (>InitTol)
    state_M2 = 0;
    if ((abs(abs(xl)-abs(xd(0)))<dsTol) && (abs(abs(yl)-abs(xd(1)))<dsTol) && (abs(xd(0))>InitTol) && (abs(xd(1))>InitTol)){
      count_M2 += 1;
    } else {
      count_M2 = 0;
      xl = xd(0);
      yl = xd(1);
    }
    if (count_M2 > 30){
      state_M2 = 1;
    }
    
    // State Machine
    // Adjust control parameters based on different states
    if ((abs(xd(0))<InitTol) && (abs(xd(1))<InitTol) && (abs(xd(2))<InitTol)){ // Init Fall: For the situation without Information from Topic "/global_plan"
      amp_cmd = 1.0;
      turn_cmd = 0.0;
      vel_cmd = 0.0;
      vel4_cmd = 0.0;
    } else if ((state_M == 1) && (state_M2 == 1)) { // Tuning&Oscillation Fall: Small forward velocity, Big tuning velocity, Big vertical Amplitude
      amp_cmd = 1.0;
      turn_cmd *= K_P_yaw;
      vel_cmd = 0.1;
      vel4_cmd = 0.0;
    } else if (state_M == 1){ // Tuning Fall
      amp_cmd = 1.0;
      turn_cmd *= K_P_yaw;
      vel_cmd = 0.1;
      vel4_cmd = 0.0;
    } else if (state_M2 == 1){ // Oscillation Fall
      amp_cmd = 0.08;
      turn_cmd *= K_P_yaw;
      vel_cmd = 0.3;
      vel4_cmd = 0.5;
      if (count_M22 == 0){
        // in this fall, we start a climbing process
        // with a length of 100 loops/iterations
        // see Line 350-367
        count_M22 = 100;
      }
    } else { // Normal Fall
      amp_cmd = 1.0;
      turn_cmd *= K_P_yaw;
      vel_cmd = vel_cmd;
      vel4_cmd = 0.0;
      if (vel_cmd > 1.0){ // Maximal forward velocity is 1
      vel_cmd = 1.0;
      }
    }
    
    // Climbing process
    // Go back first before 1st stair
    // count_M22 in (80, 100): stop at the stair, do Leg raises in situ
    // count_M22 in (0, 80): go forward slowly
    if (count_M22 > 80){
      amp_cmd = 0.08;
      turn_cmd = 0.0;
      vel_cmd = 0.0;
      vel4_cmd = 0.0;
      count_M22 -= 1;
    } else if (count_M22 > 0){
      amp_cmd = 0.1;
      turn_cmd = 0.0;
      vel_cmd = 0.5;
      vel4_cmd = 0.0;
      count_M22 -= 1;
      count_M2 = 0;
    }
    // After entering state_M2, the counter that controls
    // whether to enter state_M2 is reset to zero (count_M2 := 0).
    // After every climbing iteration, the counter count_M22
    // is subtracted by one, after 100 iterations, count_M22 resets to zero,
    // and climbing process stops.
    // In the current situation, multiple climbing processes need to be performed
    // before the robot's front legs climb on to the stairs.
    
    // For testing mode
    // The previous control parameters will be overwirtten here
    // amp_cmd = 0.4;
    // turn_cmd = 0.0;
    // vel_cmd = 0.5;
    // vel4_cmd = 0.0;

    // Output informations to Terminal every 3 loops
    // Informations include control parameters and state machine
    if (iter_info < 3){
      iter_info += 1;
    }  else {
      iter_info = 0;
      ROS_INFO_NAMED(
          "keyboard_control",
          "vel_cmd: %f | turn_cmd: %f | amp_com: %f | state1: %d | state2: %d | count_M2: %d | count_M22: %d",
          vel_cmd, turn_cmd, amp_cmd, state_M, state_M2, count_M2, count_M22);
    }

    msg.angular_velocities.resize(4);
    // Forward Velocity
    msg.angular_velocities[0] = vel_cmd;  // 1;
    // Turning angle
    msg.angular_velocities[1] = turn_cmd;  // -4.8-4.8;
    // Amplitude; range unknown
    msg.angular_velocities[2] = amp_cmd;  // -4 to 4;
    // not used
    msg.angular_velocities[3] = vel4_cmd;

    prop_speeds.publish(msg);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "controller_node");
  ROS_INFO_NAMED("controller", "Controller started!");
  controllerNode n;
  ros::spin();
}
