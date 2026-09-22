#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

struct CameraObservation
{
    int camera_id = 0;
    Vector3d point = Vector3d::Zero();
    Vector2d uv = Vector2d::Zero();
    Vector2d velocity = Vector2d::Zero();
};

struct FeatureFrameObservation
{
    bool has_left = false;
    CameraObservation left;
    bool has_right = false;
    CameraObservation right;
    bool has_stereo_depth = false;
    double stereo_depth = std::numeric_limits<double>::quiet_NaN();
};

typedef map<int, FeatureFrameObservation> FeatureFrameObservations;

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const FeatureFrameObservation &observation, double td)
    {
        ROS_ASSERT(observation.has_left);
        point = observation.left.point;
        uv = observation.left.uv;
        velocity = observation.left.velocity;
        is_stereo = observation.has_right;
        right_point = observation.right.point;
        right_uv = observation.right.uv;
        right_velocity = observation.right.velocity;
        has_stereo_depth = observation.has_stereo_depth;
        stereo_depth = observation.stereo_depth;
        cur_td = td;
        z = 0.0;
        is_used = false;
        parallax = 0.0;
        dep_gradient = 0.0;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    bool is_stereo;
    Vector3d right_point;
    Vector2d right_uv;
    Vector2d right_velocity;
    bool has_stereo_depth;
    double stereo_depth;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    bool stereo_depth_initialized;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), is_outlier(false), is_margin(false), estimated_depth(-1.0),
          solve_flag(0), stereo_depth_initialized(false)
    {
    }

    bool initializeStereoDepth()
    {
        if (stereo_depth_initialized || feature_per_frame.empty())
            return false;
        const FeaturePerFrame &anchor = feature_per_frame.front();
        if (!anchor.has_stereo_depth || !std::isfinite(anchor.stereo_depth) || anchor.stereo_depth <= 0.0)
            return false;
        estimated_depth = anchor.stereo_depth;
        stereo_depth_initialized = true;
        return true;
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const FeatureFrameObservations &image, double td);
    int initializeStereoDepths();
    int initializeNewStereoDepths(int frame_count);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[MAX_NUM_OF_CAM];
};

#endif
