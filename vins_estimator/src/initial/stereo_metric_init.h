#pragma once

#include <map>
#include <string>

#include <eigen3/Eigen/Dense>

#include "../feature_manager.h"

struct StereoMetricPnPStats
{
    int correspondences = 0;
    int inliers = 0;
    double median_reprojection_error_px = 0.0;
};

// The metric initialization map is expressed in the first left-camera frame.
using StereoMetricLandmarks = std::map<int, Eigen::Vector3d>;

int countValidStereoMetricObservations(const FeatureFrameObservations &observations,
                                       double min_depth,
                                       double max_depth);

int addStereoMetricLandmarks(const FeatureFrameObservations &observations,
                             const Eigen::Matrix3d &R_wc,
                             const Eigen::Vector3d &P_wc,
                             double min_depth,
                             double max_depth,
                             StereoMetricLandmarks &landmarks);

bool solveStereoMetricPnP(const StereoMetricLandmarks &landmarks,
                          const FeatureFrameObservations &observations,
                          Eigen::Matrix3d &R_wc,
                          Eigen::Vector3d &P_wc,
                          StereoMetricPnPStats &stats,
                          std::string &failure_reason);
