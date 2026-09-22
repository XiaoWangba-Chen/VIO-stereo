#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "../src/factor/projection_factor.h"
#include "../src/factor/projection_stereo_factor.h"
#include "../src/utility/utility.h"

namespace
{
typedef std::array<double, 7> Pose;

Pose makePose(const Eigen::Vector3d &translation = Eigen::Vector3d::Zero(),
              const Eigen::Quaterniond &rotation = Eigen::Quaterniond::Identity())
{
    Pose pose = {{translation.x(), translation.y(), translation.z(),
                  rotation.x(), rotation.y(), rotation.z(), rotation.w()}};
    return pose;
}

void perturbPose(Pose &pose, int local_index, double epsilon)
{
    if (local_index < 3)
    {
        pose[local_index] += epsilon;
        return;
    }
    Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]);
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta[local_index - 3] = epsilon;
    q = q * Utility::deltaQ(delta);
    pose[3] = q.x();
    pose[4] = q.y();
    pose[5] = q.z();
    pose[6] = q.w();
}

template <typename Factor>
Eigen::Vector2d evaluate(Factor &factor, const std::vector<const double *> &parameters)
{
    Eigen::Vector2d residual;
    factor.Evaluate(parameters.data(), residual.data(), nullptr);
    return residual;
}
}

TEST(ProjectionStereoFactor, IdealGeometryHasZeroResidual)
{
    ProjectionFactor::sqrt_info.setIdentity();
    Pose extrinsic0 = makePose();
    Pose extrinsic1 = makePose(Eigen::Vector3d(0.1, 0.0, 0.0));
    double inverse_depth = 0.5;
    ProjectionOneFrameTwoCamFactor factor(Eigen::Vector3d(0.1, -0.05, 1.0),
                                          Eigen::Vector3d(0.05, -0.05, 1.0));
    const std::vector<const double *> parameters = {
        extrinsic0.data(), extrinsic1.data(), &inverse_depth};
    EXPECT_LT(evaluate(factor, parameters).norm(), 1e-12);
}

TEST(ProjectionStereoFactor, SameFrameAnalyticJacobiansMatchFiniteDifference)
{
    ProjectionFactor::sqrt_info.setIdentity();
    Pose extrinsic0 = makePose(Eigen::Vector3d(0.01, -0.02, 0.03),
                               Eigen::Quaterniond(Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY())));
    Pose extrinsic1 = makePose(Eigen::Vector3d(0.11, 0.01, -0.02),
                               Eigen::Quaterniond(Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitZ())));
    double inverse_depth = 0.4;
    ProjectionOneFrameTwoCamFactor factor(Eigen::Vector3d(0.12, -0.08, 1.0),
                                          Eigen::Vector3d(0.07, -0.09, 1.0));
    std::vector<const double *> parameters = {extrinsic0.data(), extrinsic1.data(), &inverse_depth};
    double jacobian0[14], jacobian1[14], jacobian_depth[2];
    double *jacobians[] = {jacobian0, jacobian1, jacobian_depth};
    double residual[2];
    factor.Evaluate(parameters.data(), residual, jacobians);

    const double epsilon = 1e-7;
    for (int block = 0; block < 2; ++block)
    {
        for (int column = 0; column < 6; ++column)
        {
            Pose perturbed = block == 0 ? extrinsic0 : extrinsic1;
            perturbPose(perturbed, column, epsilon);
            std::vector<const double *> changed = parameters;
            changed[block] = perturbed.data();
            const Eigen::Vector2d numeric =
                (evaluate(factor, changed) - Eigen::Map<Eigen::Vector2d>(residual)) / epsilon;
            const double *analytic = block == 0 ? jacobian0 : jacobian1;
            EXPECT_NEAR(numeric.x(), analytic[column], 2e-5);
            EXPECT_NEAR(numeric.y(), analytic[7 + column], 2e-5);
        }
    }
    double changed_depth = inverse_depth + epsilon;
    std::vector<const double *> changed = parameters;
    changed[2] = &changed_depth;
    const Eigen::Vector2d numeric =
        (evaluate(factor, changed) - Eigen::Map<Eigen::Vector2d>(residual)) / epsilon;
    EXPECT_NEAR(numeric.x(), jacobian_depth[0], 2e-5);
    EXPECT_NEAR(numeric.y(), jacobian_depth[1], 2e-5);
}

TEST(ProjectionStereoFactor, TwoFrameAnalyticJacobiansMatchFiniteDifference)
{
    ProjectionFactor::sqrt_info.setIdentity();
    Pose pose_i = makePose(Eigen::Vector3d(0.02, -0.01, 0.03),
                           Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ())));
    Pose pose_j = makePose(Eigen::Vector3d(0.2, 0.01, -0.02),
                           Eigen::Quaterniond(Eigen::AngleAxisd(-0.04, Eigen::Vector3d::UnitY())));
    Pose extrinsic0 = makePose(Eigen::Vector3d(0.01, -0.02, 0.03),
                               Eigen::Quaterniond(Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitX())));
    Pose extrinsic1 = makePose(Eigen::Vector3d(0.11, 0.01, -0.01),
                               Eigen::Quaterniond(Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitZ())));
    double inverse_depth = 0.35;
    ProjectionTwoFrameTwoCamFactor factor(Eigen::Vector3d(0.15, -0.06, 1.0),
                                          Eigen::Vector3d(-0.01, -0.05, 1.0));
    std::vector<const double *> parameters = {
        pose_i.data(), pose_j.data(), extrinsic0.data(), extrinsic1.data(), &inverse_depth};
    double jacobian_storage[4][14];
    double jacobian_depth[2];
    double *jacobians[] = {jacobian_storage[0], jacobian_storage[1],
                           jacobian_storage[2], jacobian_storage[3], jacobian_depth};
    double residual[2];
    factor.Evaluate(parameters.data(), residual, jacobians);

    const double epsilon = 1e-7;
    for (int block = 0; block < 4; ++block)
    {
        for (int column = 0; column < 6; ++column)
        {
            Pose perturbed;
            if (block == 0) perturbed = pose_i;
            if (block == 1) perturbed = pose_j;
            if (block == 2) perturbed = extrinsic0;
            if (block == 3) perturbed = extrinsic1;
            perturbPose(perturbed, column, epsilon);
            std::vector<const double *> changed = parameters;
            changed[block] = perturbed.data();
            const Eigen::Vector2d numeric =
                (evaluate(factor, changed) - Eigen::Map<Eigen::Vector2d>(residual)) / epsilon;
            EXPECT_NEAR(numeric.x(), jacobian_storage[block][column], 3e-5);
            EXPECT_NEAR(numeric.y(), jacobian_storage[block][7 + column], 3e-5);
        }
    }
    double changed_depth = inverse_depth + epsilon;
    std::vector<const double *> changed = parameters;
    changed[4] = &changed_depth;
    const Eigen::Vector2d numeric =
        (evaluate(factor, changed) - Eigen::Map<Eigen::Vector2d>(residual)) / epsilon;
    EXPECT_NEAR(numeric.x(), jacobian_depth[0], 3e-5);
    EXPECT_NEAR(numeric.y(), jacobian_depth[1], 3e-5);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
