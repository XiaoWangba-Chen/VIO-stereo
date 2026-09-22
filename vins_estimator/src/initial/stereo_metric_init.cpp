#include "stereo_metric_init.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "../parameters.h"

namespace
{
bool finiteVector(const Eigen::Vector3d &value)
{
    return value.array().isFinite().all();
}

bool validImagePoint(const Eigen::Vector2d &point)
{
    return point.array().isFinite().all() && point.x() >= 0.0 && point.x() < COL &&
           point.y() >= 0.0 && point.y() < ROW;
}

bool validStereoObservation(const FeatureFrameObservation &observation,
                            double min_depth,
                            double max_depth)
{
    const double depth = observation.stereo_depth;
    return observation.has_left && observation.has_right && observation.has_stereo_depth &&
           std::isfinite(depth) && depth >= min_depth && depth <= max_depth &&
           finiteVector(observation.left.point) && observation.left.point.z() > 0.0 &&
           validImagePoint(observation.left.uv) && validImagePoint(observation.right.uv);
}
}

int countValidStereoMetricObservations(const FeatureFrameObservations &observations,
                                       double min_depth,
                                       double max_depth)
{
    int valid = 0;
    for (const auto &id_observation : observations)
    {
        const FeatureFrameObservation &observation = id_observation.second;
        if (validStereoObservation(observation, min_depth, max_depth))
            ++valid;
    }
    return valid;
}

int addStereoMetricLandmarks(const FeatureFrameObservations &observations,
                             const Eigen::Matrix3d &R_wc,
                             const Eigen::Vector3d &P_wc,
                             double min_depth,
                             double max_depth,
                             StereoMetricLandmarks &landmarks)
{
    int added = 0;
    for (const auto &id_observation : observations)
    {
        const int feature_id = id_observation.first;
        const FeatureFrameObservation &observation = id_observation.second;
        if (landmarks.count(feature_id) != 0 || !observation.has_left ||
            !observation.has_right || !observation.has_stereo_depth)
            continue;
        const double depth = observation.stereo_depth;
        if (!validStereoObservation(observation, min_depth, max_depth))
            continue;

        const Eigen::Vector3d point_camera =
            observation.left.point / observation.left.point.z() * depth;
        const Eigen::Vector3d point_world = R_wc * point_camera + P_wc;
        if (!finiteVector(point_world))
            continue;
        landmarks.emplace(feature_id, point_world);
        ++added;
    }
    return added;
}

bool solveStereoMetricPnP(const StereoMetricLandmarks &landmarks,
                          const FeatureFrameObservations &observations,
                          Eigen::Matrix3d &R_wc,
                          Eigen::Vector3d &P_wc,
                          StereoMetricPnPStats &stats,
                          std::string &failure_reason)
{
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(observations.size());
    image_points.reserve(observations.size());
    for (const auto &id_observation : observations)
    {
        const auto landmark = landmarks.find(id_observation.first);
        const FeatureFrameObservation &observation = id_observation.second;
        if (landmark == landmarks.end() || !observation.has_left ||
            !finiteVector(landmark->second) || !finiteVector(observation.left.point) ||
            observation.left.point.z() <= 0.0)
            continue;
        const Eigen::Vector3d normalized = observation.left.point / observation.left.point.z();
        object_points.emplace_back(landmark->second.x(), landmark->second.y(), landmark->second.z());
        image_points.emplace_back(normalized.x(), normalized.y());
    }

    stats = StereoMetricPnPStats();
    stats.correspondences = static_cast<int>(object_points.size());
    if (stats.correspondences < 20)
    {
        failure_reason = "PnP has fewer than 20 3D-2D correspondences";
        return false;
    }

    const Eigen::Matrix3d R_cw_guess = R_wc.transpose();
    const Eigen::Vector3d t_cw_guess = -R_cw_guess * P_wc;
    cv::Mat rotation_guess;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::eigen2cv(R_cw_guess, rotation_guess);
    cv::Rodrigues(rotation_guess, rvec);
    cv::eigen2cv(t_cw_guess, tvec);
    const cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat distortion;
    cv::Mat inlier_indices;
    const bool solved = cv::solvePnPRansac(
        object_points, image_points, K, distortion, rvec, tvec, true,
        100, 2.0 / FOCAL_LENGTH, 0.99, inlier_indices, cv::SOLVEPNP_ITERATIVE);
    if (!solved)
    {
        failure_reason = "PnP RANSAC failed";
        return false;
    }
    stats.inliers = inlier_indices.rows;
    if (stats.inliers < 15)
    {
        failure_reason = "PnP has fewer than 15 RANSAC inliers";
        return false;
    }

    cv::Mat rotation_cw_cv;
    cv::Rodrigues(rvec, rotation_cw_cv);
    Eigen::Matrix3d R_cw;
    Eigen::Vector3d t_cw;
    cv::cv2eigen(rotation_cw_cv, R_cw);
    cv::cv2eigen(tvec, t_cw);
    if (!R_cw.array().isFinite().all() || !finiteVector(t_cw))
    {
        failure_reason = "PnP returned NaN or Inf";
        return false;
    }

    std::vector<double> errors_px;
    errors_px.reserve(stats.inliers);
    for (int row = 0; row < inlier_indices.rows; ++row)
    {
        const int index = inlier_indices.at<int>(row, 0);
        const Eigen::Vector3d point_world(object_points[index].x,
                                          object_points[index].y,
                                          object_points[index].z);
        const Eigen::Vector3d point_camera = R_cw * point_world + t_cw;
        if (!finiteVector(point_camera) || point_camera.z() <= 0.0)
        {
            failure_reason = "PnP inlier has non-positive camera depth";
            return false;
        }
        const Eigen::Vector2d projected = point_camera.head<2>() / point_camera.z();
        const Eigen::Vector2d measured(image_points[index].x, image_points[index].y);
        errors_px.push_back((projected - measured).norm() * FOCAL_LENGTH);
    }
    const std::size_t middle = errors_px.size() / 2;
    std::nth_element(errors_px.begin(), errors_px.begin() + middle, errors_px.end());
    stats.median_reprojection_error_px = errors_px[middle];
    if (!std::isfinite(stats.median_reprojection_error_px) ||
        stats.median_reprojection_error_px > 2.0)
    {
        failure_reason = "PnP median reprojection error exceeds 2 px";
        return false;
    }

    R_wc = R_cw.transpose();
    P_wc = -R_wc * t_cw;
    if (!R_wc.array().isFinite().all() || !finiteVector(P_wc))
    {
        failure_reason = "PnP camera pose is not finite";
        return false;
    }
    return true;
}
