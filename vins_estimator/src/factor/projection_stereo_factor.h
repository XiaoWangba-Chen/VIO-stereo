#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
  public:
    ProjectionOneFrameTwoCamFactor(const Eigen::Vector3d &_pts_i,
                                   const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
};

class ProjectionTwoFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1>
{
  public:
    ProjectionTwoFrameTwoCamFactor(const Eigen::Vector3d &_pts_i,
                                   const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
};
