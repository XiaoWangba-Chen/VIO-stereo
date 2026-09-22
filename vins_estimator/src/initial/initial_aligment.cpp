#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

namespace
{
bool solveMetricGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                              const Vector3d &initial_bg,
                              Vector3d &solved_bg,
                              string &failure_reason)
{
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    for (auto frame_i = all_image_frame.begin();
         frame_i != all_image_frame.end() && next(frame_i) != all_image_frame.end(); ++frame_i)
    {
        auto frame_j = next(frame_i);
        if (!frame_j->second.pre_integration || frame_j->second.pre_integration->sum_dt <= 0.0)
        {
            failure_reason = "missing or empty IMU pre-integration";
            return false;
        }
        const Matrix3d jacobian =
            frame_j->second.pre_integration->jacobian.block<3, 3>(O_R, O_BG);
        const Quaterniond visual_delta(frame_i->second.R.transpose() * frame_j->second.R);
        const Vector3d residual =
            2.0 * (frame_j->second.pre_integration->delta_q.inverse() * visual_delta).vec();
        A.noalias() += jacobian.transpose() * jacobian;
        b.noalias() += jacobian.transpose() * residual;
    }
    Eigen::LDLT<Matrix3d> solver(A);
    if (solver.info() != Eigen::Success)
    {
        failure_reason = "gyroscope bias normal equation is singular";
        return false;
    }
    const Vector3d delta_bg = solver.solve(b);
    solved_bg = initial_bg + delta_bg;
    if (solver.info() != Eigen::Success || !solved_bg.array().isFinite().all())
    {
        failure_reason = "gyroscope bias is not finite";
        return false;
    }
    ROS_INFO_STREAM("stereo metric init gyroscope bias " << solved_bg.transpose());
    for (auto frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); ++frame_i)
    {
        if (frame_i->second.pre_integration)
            frame_i->second.pre_integration->repropagate(Vector3d::Zero(), solved_bg);
    }
    return true;
}

bool solveMetricLinearSystem(map<double, ImageFrame> &all_image_frame,
                             Vector3d &g,
                             VectorXd &x,
                             string &failure_reason)
{
    const int frame_count = static_cast<int>(all_image_frame.size());
    const int state_size = frame_count * 3 + 3;
    MatrixXd A = MatrixXd::Zero(state_size, state_size);
    VectorXd b = VectorXd::Zero(state_size);
    int index = 0;
    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end();
         ++frame_i, ++index)
    {
        const auto frame_j = next(frame_i);
        const double dt = frame_j->second.pre_integration->sum_dt;
        MatrixXd equation = MatrixXd::Zero(6, 9);
        VectorXd rhs = VectorXd::Zero(6);
        equation.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        equation.block<3, 3>(0, 6) =
            frame_i->second.R.transpose() * (dt * dt / 2.0) * Matrix3d::Identity();
        rhs.head<3>() = frame_j->second.pre_integration->delta_p +
            frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
            frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
        equation.block<3, 3>(3, 0) = -Matrix3d::Identity();
        equation.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        equation.block<3, 3>(3, 6) =
            frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        rhs.tail<3>() = frame_j->second.pre_integration->delta_v;

        const MatrixXd normal = equation.transpose() * equation;
        const VectorXd normal_rhs = equation.transpose() * rhs;
        A.block<6, 6>(index * 3, index * 3) += normal.topLeftCorner<6, 6>();
        b.segment<6>(index * 3) += normal_rhs.head<6>();
        A.bottomRightCorner<3, 3>() += normal.bottomRightCorner<3, 3>();
        b.tail<3>() += normal_rhs.tail<3>();
        A.block(index * 3, state_size - 3, 6, 3) += normal.topRightCorner(6, 3);
        A.block(state_size - 3, index * 3, 3, 6) += normal.bottomLeftCorner(3, 6);
    }
    A *= 1000.0;
    b *= 1000.0;
    Eigen::LDLT<MatrixXd> solver(A);
    if (solver.info() != Eigen::Success)
    {
        failure_reason = "fixed-scale velocity/gravity system is singular";
        return false;
    }
    x = solver.solve(b);
    if (solver.info() != Eigen::Success || !x.array().isFinite().all())
    {
        failure_reason = "fixed-scale velocity/gravity solution is not finite";
        return false;
    }
    g = x.tail<3>();
    if (!g.array().isFinite().all() || std::abs(g.norm() - G.norm()) > 1.0)
    {
        failure_reason = "fixed-scale gravity norm differs from configuration by more than 1 m/s^2";
        return false;
    }

    Vector3d refined_g = g.normalized() * G.norm();
    VectorXd refined_solution;
    for (int iteration = 0; iteration < 4; ++iteration)
    {
        const MatrixXd basis = TangentBasis(refined_g);
        const int refined_size = frame_count * 3 + 2;
        MatrixXd refined_A = MatrixXd::Zero(refined_size, refined_size);
        VectorXd refined_b = VectorXd::Zero(refined_size);
        int refined_index = 0;
        for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end();
             ++frame_i, ++refined_index)
        {
            const auto frame_j = next(frame_i);
            const double dt = frame_j->second.pre_integration->sum_dt;
            MatrixXd equation = MatrixXd::Zero(6, 8);
            VectorXd rhs = VectorXd::Zero(6);
            equation.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            equation.block<3, 2>(0, 6) =
                frame_i->second.R.transpose() * (dt * dt / 2.0) * basis;
            rhs.head<3>() = frame_j->second.pre_integration->delta_p +
                frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
                frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) -
                frame_i->second.R.transpose() * (dt * dt / 2.0) * refined_g;
            equation.block<3, 3>(3, 0) = -Matrix3d::Identity();
            equation.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            equation.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * basis;
            rhs.tail<3>() = frame_j->second.pre_integration->delta_v -
                frame_i->second.R.transpose() * dt * refined_g;

            const MatrixXd normal = equation.transpose() * equation;
            const VectorXd normal_rhs = equation.transpose() * rhs;
            refined_A.block<6, 6>(refined_index * 3, refined_index * 3) +=
                normal.topLeftCorner<6, 6>();
            refined_b.segment<6>(refined_index * 3) += normal_rhs.head<6>();
            refined_A.bottomRightCorner<2, 2>() += normal.bottomRightCorner<2, 2>();
            refined_b.tail<2>() += normal_rhs.tail<2>();
            refined_A.block(refined_index * 3, refined_size - 2, 6, 2) +=
                normal.topRightCorner(6, 2);
            refined_A.block(refined_size - 2, refined_index * 3, 2, 6) +=
                normal.bottomLeftCorner(2, 6);
        }
        refined_A *= 1000.0;
        refined_b *= 1000.0;
        Eigen::LDLT<MatrixXd> refined_solver(refined_A);
        if (refined_solver.info() != Eigen::Success)
        {
            failure_reason = "fixed-scale gravity refinement system is singular";
            return false;
        }
        refined_solution = refined_solver.solve(refined_b);
        if (!refined_solution.array().isFinite().all())
        {
            failure_reason = "refined fixed-scale alignment is not finite";
            return false;
        }
        refined_g = (refined_g + basis * refined_solution.tail<2>()).normalized() * G.norm();
    }
    g = refined_g;
    x.resize(state_size);
    x.head(frame_count * 3) = refined_solution.head(frame_count * 3);
    x.tail<3>() = g;
    return true;
}
}

bool MetricVisualIMUAlignment(map<double, ImageFrame> &all_image_frame,
                              const Vector3d &initial_bg,
                              Vector3d &solved_bg,
                              Vector3d &g,
                              VectorXd &x,
                              string &failure_reason)
{
    if (all_image_frame.size() < 2)
    {
        failure_reason = "metric initialization window has fewer than two frames";
        return false;
    }
    if (!solveMetricGyroscopeBias(all_image_frame, initial_bg, solved_bg, failure_reason))
        return false;
    return solveMetricLinearSystem(all_image_frame, g, x, failure_reason);
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
