#include <gtest/gtest.h>

#include "../src/initial/stereo_metric_init.h"

double ROW = 480.0;
double COL = 848.0;

namespace
{
FeatureFrameObservation makeObservation(const Eigen::Vector3d &point_camera,
                                        double stereo_depth)
{
    FeatureFrameObservation observation;
    observation.has_left = true;
    observation.has_right = true;
    observation.has_stereo_depth = true;
    observation.stereo_depth = stereo_depth;
    observation.left.camera_id = 0;
    observation.left.point = point_camera / point_camera.z();
    observation.left.uv = observation.left.point.head<2>() * FOCAL_LENGTH +
                          Eigen::Vector2d(424.0, 240.0);
    observation.right.camera_id = 1;
    observation.right.point = observation.left.point;
    observation.right.uv = observation.left.uv - Eigen::Vector2d(5.0, 0.0);
    return observation;
}
}

TEST(StereoMetricInit, BuildsOneMetricLandmarkPerFeature)
{
    FeatureFrameObservations observations;
    observations.emplace(7, makeObservation(Eigen::Vector3d(0.4, -0.2, 4.0), 4.0));
    StereoMetricLandmarks landmarks;
    EXPECT_EQ(1, addStereoMetricLandmarks(observations, Eigen::Matrix3d::Identity(),
                                          Eigen::Vector3d::Zero(), 0.2, 15.0, landmarks));
    EXPECT_EQ(0, addStereoMetricLandmarks(observations, Eigen::Matrix3d::Identity(),
                                          Eigen::Vector3d::Zero(), 0.2, 15.0, landmarks));
    ASSERT_EQ(1u, landmarks.size());
    EXPECT_NEAR(4.0, landmarks.at(7).z(), 1e-12);
}

TEST(StereoMetricInit, RejectsDepthOutsideConfiguredRange)
{
    FeatureFrameObservations observations;
    observations.emplace(1, makeObservation(Eigen::Vector3d(0.0, 0.0, 0.1), 0.1));
    observations.emplace(2, makeObservation(Eigen::Vector3d(0.0, 0.0, 16.0), 16.0));
    StereoMetricLandmarks landmarks;
    EXPECT_EQ(0, addStereoMetricLandmarks(observations, Eigen::Matrix3d::Identity(),
                                          Eigen::Vector3d::Zero(), 0.2, 15.0, landmarks));
}

TEST(StereoMetricInit, RecoversMetricPoseWithPnPRansac)
{
    StereoMetricLandmarks landmarks;
    FeatureFrameObservations observations;
    const Eigen::Matrix3d expected_rotation =
        Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Vector3d expected_position(0.25, -0.04, 0.08);
    for (int id = 0; id < 30; ++id)
    {
        const Eigen::Vector3d point_world(-1.0 + 0.08 * id,
                                          -0.45 + 0.15 * (id % 7),
                                          3.0 + 0.12 * (id % 9));
        landmarks.emplace(id, point_world);
        const Eigen::Vector3d point_camera =
            expected_rotation.transpose() * (point_world - expected_position);
        observations.emplace(id, makeObservation(point_camera, point_camera.z()));
    }

    Eigen::Matrix3d recovered_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d recovered_position = Eigen::Vector3d::Zero();
    StereoMetricPnPStats stats;
    std::string reason;
    ASSERT_TRUE(solveStereoMetricPnP(landmarks, observations, recovered_rotation,
                                     recovered_position, stats, reason)) << reason;
    EXPECT_GE(stats.correspondences, 20);
    EXPECT_GE(stats.inliers, 15);
    EXPECT_LE(stats.median_reprojection_error_px, 2.0);
    EXPECT_LT((recovered_position - expected_position).norm(), 1e-3);
    EXPECT_LT((recovered_rotation.transpose() * expected_rotation -
               Eigen::Matrix3d::Identity()).norm(), 1e-3);
}
