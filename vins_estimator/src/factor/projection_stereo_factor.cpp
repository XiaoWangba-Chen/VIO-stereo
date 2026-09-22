#include "projection_stereo_factor.h"

#include "projection_factor.h"
#include "../utility/utility.h"

#include <cmath>

namespace
{
Eigen::Quaterniond quaternionFromPose(const double *pose)
{
    return Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]);
}

Eigen::Matrix<double, 2, 3> projectionJacobian(const Eigen::Vector3d &point,
                                               const Eigen::Vector3d &observation,
                                               Eigen::Matrix<double, 2, 3> tangent_base)
{
#ifdef UNIT_SPHERE_ERROR
    const double norm = point.norm();
    const Eigen::Matrix3d norm_jacobian =
        Eigen::Matrix3d::Identity() / norm - point * point.transpose() / std::pow(norm, 3);
    return ProjectionFactor::sqrt_info * tangent_base * norm_jacobian;
#else
    (void)observation;
    (void)tangent_base;
    const double depth = point.z();
    Eigen::Matrix<double, 2, 3> reduce;
    reduce << 1.0 / depth, 0.0, -point.x() / (depth * depth),
              0.0, 1.0 / depth, -point.y() / (depth * depth);
    return ProjectionFactor::sqrt_info * reduce;
#endif
}

Eigen::Vector2d projectionResidual(const Eigen::Vector3d &point,
                                   const Eigen::Vector3d &observation,
                                   const Eigen::Matrix<double, 2, 3> &tangent_base)
{
#ifdef UNIT_SPHERE_ERROR
    return ProjectionFactor::sqrt_info * tangent_base *
           (point.normalized() - observation.normalized());
#else
    return ProjectionFactor::sqrt_info *
           ((point / point.z()).head<2>() - observation.head<2>());
#endif
}

Eigen::Matrix<double, 2, 3> makeTangentBase(const Eigen::Vector3d &observation)
{
    Eigen::Matrix<double, 2, 3> tangent_base = Eigen::Matrix<double, 2, 3>::Zero();
#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d axis(0.0, 0.0, 1.0);
    const Eigen::Vector3d normalized = observation.normalized();
    if (normalized == axis)
        axis << 1.0, 0.0, 0.0;
    const Eigen::Vector3d b1 = (axis - normalized * normalized.dot(axis)).normalized();
    tangent_base.row(0) = b1.transpose();
    tangent_base.row(1) = normalized.cross(b1).transpose();
#else
    (void)observation;
#endif
    return tangent_base;
}
}

ProjectionOneFrameTwoCamFactor::ProjectionOneFrameTwoCamFactor(
    const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j), tangent_base(makeTangentBase(_pts_j))
{
}

bool ProjectionOneFrameTwoCamFactor::Evaluate(double const *const *parameters,
                                               double *residuals,
                                               double **jacobians) const
{
    const Eigen::Vector3d tic0(parameters[0][0], parameters[0][1], parameters[0][2]);
    const Eigen::Matrix3d ric0 = quaternionFromPose(parameters[0]).toRotationMatrix();
    const Eigen::Vector3d tic1(parameters[1][0], parameters[1][1], parameters[1][2]);
    const Eigen::Matrix3d ric1 = quaternionFromPose(parameters[1]).toRotationMatrix();
    const double inverse_depth = parameters[2][0];

    const Eigen::Vector3d point_camera0 = pts_i / inverse_depth;
    const Eigen::Vector3d point_camera1 =
        ric1.transpose() * (ric0 * point_camera0 + tic0 - tic1);
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = projectionResidual(point_camera1, pts_j, tangent_base);

    if (!jacobians)
        return true;

    const Eigen::Matrix<double, 2, 3> reduce =
        projectionJacobian(point_camera1, pts_j, tangent_base);
    if (jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
        jacobian.leftCols<3>() = reduce * ric1.transpose();
        jacobian.middleCols<3>(3) = reduce * ric1.transpose() * ric0 *
                                    -Utility::skewSymmetric(point_camera0);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[1]);
        jacobian.leftCols<3>() = reduce * -ric1.transpose();
        jacobian.middleCols<3>(3) = reduce * Utility::skewSymmetric(point_camera1);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[2])
    {
        Eigen::Map<Eigen::Vector2d> jacobian(jacobians[2]);
        jacobian = reduce * ric1.transpose() * ric0 * pts_i *
                   (-1.0 / (inverse_depth * inverse_depth));
    }
    return true;
}

ProjectionTwoFrameTwoCamFactor::ProjectionTwoFrameTwoCamFactor(
    const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j)
    : pts_i(_pts_i), pts_j(_pts_j), tangent_base(makeTangentBase(_pts_j))
{
}

bool ProjectionTwoFrameTwoCamFactor::Evaluate(double const *const *parameters,
                                               double *residuals,
                                               double **jacobians) const
{
    const Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    const Eigen::Matrix3d Ri = quaternionFromPose(parameters[0]).toRotationMatrix();
    const Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    const Eigen::Matrix3d Rj = quaternionFromPose(parameters[1]).toRotationMatrix();
    const Eigen::Vector3d tic0(parameters[2][0], parameters[2][1], parameters[2][2]);
    const Eigen::Matrix3d ric0 = quaternionFromPose(parameters[2]).toRotationMatrix();
    const Eigen::Vector3d tic1(parameters[3][0], parameters[3][1], parameters[3][2]);
    const Eigen::Matrix3d ric1 = quaternionFromPose(parameters[3]).toRotationMatrix();
    const double inverse_depth = parameters[4][0];

    const Eigen::Vector3d point_camera0 = pts_i / inverse_depth;
    const Eigen::Vector3d point_imu_i = ric0 * point_camera0 + tic0;
    const Eigen::Vector3d point_world = Ri * point_imu_i + Pi;
    const Eigen::Vector3d point_imu_j = Rj.transpose() * (point_world - Pj);
    const Eigen::Vector3d point_camera1 = ric1.transpose() * (point_imu_j - tic1);
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = projectionResidual(point_camera1, pts_j, tangent_base);

    if (!jacobians)
        return true;

    const Eigen::Matrix<double, 2, 3> reduce =
        projectionJacobian(point_camera1, pts_j, tangent_base);
    if (jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[0]);
        jacobian.leftCols<3>() = reduce * ric1.transpose() * Rj.transpose();
        jacobian.middleCols<3>(3) = reduce * ric1.transpose() * Rj.transpose() * Ri *
                                    -Utility::skewSymmetric(point_imu_i);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[1]);
        jacobian.leftCols<3>() = reduce * ric1.transpose() * -Rj.transpose();
        jacobian.middleCols<3>(3) = reduce * ric1.transpose() *
                                    Utility::skewSymmetric(point_imu_j);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[2])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[2]);
        jacobian.leftCols<3>() = reduce * ric1.transpose() * Rj.transpose() * Ri;
        jacobian.middleCols<3>(3) = reduce * ric1.transpose() * Rj.transpose() * Ri * ric0 *
                                    -Utility::skewSymmetric(point_camera0);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[3])
    {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian(jacobians[3]);
        jacobian.leftCols<3>() = reduce * -ric1.transpose();
        jacobian.middleCols<3>(3) = reduce * Utility::skewSymmetric(point_camera1);
        jacobian.rightCols<1>().setZero();
    }
    if (jacobians[4])
    {
        Eigen::Map<Eigen::Vector2d> jacobian(jacobians[4]);
        jacobian = reduce * ric1.transpose() * Rj.transpose() * Ri * ric0 * pts_i *
                   (-1.0 / (inverse_depth * inverse_depth));
    }
    return true;
}
