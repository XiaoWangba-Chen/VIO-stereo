#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int MAX_NUM_OF_CAM = 2;
extern int NUM_OF_CAM;


extern std::string IMAGE_TOPIC;
extern std::string IMAGE1_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern double STEREO_BASELINE_X;
extern double STEREO_FB_THRESHOLD;
extern double STEREO_EPIPOLAR_THRESHOLD;
extern double STEREO_MIN_DEPTH;
extern double STEREO_MAX_DEPTH;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;
extern int STEREO_USE_DEPTH_INIT;
extern int STEREO_USE_METRIC_INIT;
extern int STEREO_USE_RIGHT_FACTORS;

void readParameters(ros::NodeHandle &n);
