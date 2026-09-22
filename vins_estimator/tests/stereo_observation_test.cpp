#include <gtest/gtest.h>

#include "../src/feature_manager.h"

namespace
{
FeatureFrameObservation makeObservation(bool with_right, double depth)
{
    FeatureFrameObservation observation;
    observation.has_left = true;
    observation.left.camera_id = 0;
    observation.left.point << 0.1, -0.2, 1.0;
    if (with_right)
    {
        observation.has_right = true;
        observation.right.camera_id = 1;
        observation.right.point << 0.08, -0.2, 1.0;
    }
    if (std::isfinite(depth))
    {
        observation.has_stereo_depth = true;
        observation.stereo_depth = depth;
    }
    return observation;
}
}

TEST(StereoObservation, KeepsRightObservationOnSameFrame)
{
    FeaturePerFrame frame(makeObservation(true, 2.5), 0.0);
    EXPECT_TRUE(frame.is_stereo);
    EXPECT_TRUE(frame.has_stereo_depth);
    EXPECT_DOUBLE_EQ(2.5, frame.stereo_depth);
    EXPECT_DOUBLE_EQ(0.08, frame.right_point.x());
}

TEST(StereoObservation, MissingDepthFallsBack)
{
    FeaturePerId feature(7, 0);
    feature.feature_per_frame.emplace_back(
        makeObservation(false, std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_FALSE(feature.initializeStereoDepth());
    EXPECT_LT(feature.estimated_depth, 0.0);
}

TEST(StereoObservation, InitializesOnlyOnceFromAnchor)
{
    FeaturePerId feature(8, 0);
    feature.feature_per_frame.emplace_back(makeObservation(true, 3.0), 0.0);
    ASSERT_TRUE(feature.initializeStereoDepth());
    EXPECT_DOUBLE_EQ(3.0, feature.estimated_depth);
    feature.feature_per_frame.front().stereo_depth = 4.0;
    EXPECT_FALSE(feature.initializeStereoDepth());
    EXPECT_DOUBLE_EQ(3.0, feature.estimated_depth);
}

TEST(StereoObservation, RemovingRightObservationPreservesLeftTrack)
{
    FeaturePerFrame frame(makeObservation(true, 2.5), 0.0);
    const Eigen::Vector3d left_point = frame.point;
    const Eigen::Vector2d left_uv = frame.uv;
    frame.removeRightObservation();

    EXPECT_FALSE(frame.is_stereo);
    EXPECT_FALSE(frame.has_stereo_depth);
    EXPECT_TRUE(std::isnan(frame.stereo_depth));
    EXPECT_TRUE(frame.point.isApprox(left_point));
    EXPECT_TRUE(frame.uv.isApprox(left_uv));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
