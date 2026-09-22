#include "parameters.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;
int NUM_OF_CAM = 1;
int FEATURE_CAMERA_COUNT = 1;
int STEREO_USE_DEPTH_INIT = 0;
int STEREO_USE_METRIC_INIT = 0;
int STEREO_USE_RIGHT_FACTORS = 0;
double STEREO_MIN_DEPTH = 0.2;
double STEREO_MAX_DEPTH = 15.0;
double STEREO_EPIPOLAR_THRESHOLD = 1.0;
double STEREO_BACKEND_REPROJECTION_THRESHOLD = 3.0;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

namespace
{
int readOptionalInt(const cv::FileStorage &settings, const std::string &name, int default_value)
{
    const cv::FileNode node = settings[name];
    return node.empty() ? default_value : static_cast<int>(node);
}

double readOptionalDouble(const cv::FileStorage &settings, const std::string &name,
                          double default_value)
{
    const cv::FileNode node = settings[name];
    return node.empty() ? default_value : static_cast<double>(node);
}

void readExtrinsic(const cv::FileStorage &settings, int camera_id,
                   Eigen::Matrix3d &rotation, Eigen::Vector3d &translation)
{
    const std::string suffix = camera_id == 0 ? "" : std::to_string(camera_id);
    const std::string rotation_name = "extrinsicRotation" + suffix;
    const std::string translation_name = "extrinsicTranslation" + suffix;
    if (settings[rotation_name].empty() || settings[translation_name].empty())
        throw std::runtime_error("missing stereo extrinsic: " + rotation_name + " or " + translation_name);

    cv::Mat cv_rotation;
    cv::Mat cv_translation;
    settings[rotation_name] >> cv_rotation;
    settings[translation_name] >> cv_translation;
    if (cv_rotation.rows != 3 || cv_rotation.cols != 3 ||
        cv_translation.rows != 3 || cv_translation.cols != 1)
        throw std::runtime_error("invalid dimensions for " + rotation_name + " or " + translation_name);

    cv::cv2eigen(cv_rotation, rotation);
    cv::cv2eigen(cv_translation, translation);
    rotation = Eigen::Quaterniond(rotation).normalized().toRotationMatrix();
}

void validateStereoExtrinsics()
{
    const Eigen::Matrix3d relative_rotation = RIC[1].transpose() * RIC[0];
    const double cosine = std::max(-1.0, std::min(1.0, (relative_rotation.trace() - 1.0) / 2.0));
    const double angle_degrees = std::acos(cosine) * 180.0 / 3.14159265358979323846;
    const Eigen::Vector3d baseline_cam0 = RIC[0].transpose() * (TIC[1] - TIC[0]);
    const double baseline = baseline_cam0.norm();
    const double transverse = baseline_cam0.tail<2>().norm();

    if (angle_degrees > 1.0)
        throw std::runtime_error("stereo relative rotation exceeds 1 degree");
    if (baseline <= 0.01)
        throw std::runtime_error("stereo baseline must be greater than 1 cm");
    if (baseline_cam0.x() <= 0.0)
        throw std::runtime_error("stereo baseline x component must be positive");
    if (transverse > baseline * 0.05)
        throw std::runtime_error("stereo baseline y/z component exceeds 5 percent");

    ROS_INFO("validated stereo extrinsics: baseline=[%.9f %.9f %.9f] m, length=%.9f m, rotation=%.6f deg",
             baseline_cam0.x(), baseline_cam0.y(), baseline_cam0.z(), baseline, angle_degrees);
}
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        throw std::runtime_error("cannot open settings file: " + config_file);
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;
    FEATURE_CAMERA_COUNT = readOptionalInt(fsSettings, "num_of_cam", 1);
    if (FEATURE_CAMERA_COUNT < 1 || FEATURE_CAMERA_COUNT > MAX_NUM_OF_CAM)
        throw std::runtime_error("num_of_cam must be 1 or 2");
    // Phase B0 transports two-camera IDs but deliberately keeps the estimator
    // and Ceres problem monocular. Later phases will activate camera 1 here.
    NUM_OF_CAM = 1;
    STEREO_USE_DEPTH_INIT = readOptionalInt(fsSettings, "stereo_use_depth_init", 0);
    STEREO_USE_METRIC_INIT = readOptionalInt(fsSettings, "stereo_use_metric_init", 0);
    STEREO_USE_RIGHT_FACTORS = readOptionalInt(fsSettings, "stereo_use_right_factors", 0);
    STEREO_MIN_DEPTH = readOptionalDouble(fsSettings, "stereo_min_depth", 0.2);
    STEREO_MAX_DEPTH = readOptionalDouble(fsSettings, "stereo_max_depth", 15.0);
    STEREO_EPIPOLAR_THRESHOLD = readOptionalDouble(fsSettings, "stereo_epipolar_threshold", 1.0);
    STEREO_BACKEND_REPROJECTION_THRESHOLD =
        readOptionalDouble(fsSettings, "stereo_backend_reprojection_threshold", 3.0);
    if ((STEREO_USE_DEPTH_INIT != 0 && STEREO_USE_DEPTH_INIT != 1) ||
        (STEREO_USE_METRIC_INIT != 0 && STEREO_USE_METRIC_INIT != 1) ||
        (STEREO_USE_RIGHT_FACTORS != 0 && STEREO_USE_RIGHT_FACTORS != 1))
        throw std::runtime_error("stereo mode switches must be 0 or 1");
    if (STEREO_USE_DEPTH_INIT && FEATURE_CAMERA_COUNT != 2)
        throw std::runtime_error("stereo_use_depth_init=1 requires num_of_cam=2");
    if (STEREO_USE_RIGHT_FACTORS && FEATURE_CAMERA_COUNT != 2)
        throw std::runtime_error("stereo_use_right_factors=1 requires num_of_cam=2");
    if (STEREO_USE_METRIC_INIT && FEATURE_CAMERA_COUNT != 2)
        throw std::runtime_error("stereo_use_metric_init=1 requires num_of_cam=2");
    if (STEREO_USE_METRIC_INIT && !STEREO_USE_DEPTH_INIT)
        throw std::runtime_error("stereo_use_metric_init=1 requires stereo_use_depth_init=1");
    if (STEREO_MIN_DEPTH <= 0.0 || STEREO_MAX_DEPTH <= STEREO_MIN_DEPTH)
        throw std::runtime_error("invalid stereo depth range");
    if (STEREO_EPIPOLAR_THRESHOLD <= 0.0 ||
        STEREO_BACKEND_REPROJECTION_THRESHOLD <= 0.0)
        throw std::runtime_error("stereo observation thresholds must be positive");
    NUM_OF_CAM = STEREO_USE_RIGHT_FACTORS ? 2 : 1;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    RIC.clear();
    TIC.clear();
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        for (int camera_id = 0; camera_id < FEATURE_CAMERA_COUNT; camera_id++)
        {
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            readExtrinsic(fsSettings, camera_id, eigen_R, eigen_T);
            RIC.push_back(eigen_R);
            TIC.push_back(eigen_T);
            ROS_INFO_STREAM("Extrinsic_R camera " << camera_id << ":" << std::endl << RIC[camera_id]);
            ROS_INFO_STREAM("Extrinsic_T camera " << camera_id << ": " << TIC[camera_id].transpose());
        }
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }

    if (FEATURE_CAMERA_COUNT == 2)
    {
        if (ESTIMATE_EXTRINSIC != 0)
            throw std::runtime_error("phase-1 stereo requires estimate_extrinsic=0");
        if (ESTIMATE_TD != 0)
            throw std::runtime_error("phase-1 stereo requires estimate_td=0");
        if (ROLLING_SHUTTER != 0)
            throw std::runtime_error("phase-1 stereo requires rolling_shutter=0");
        validateStereoExtrinsics();
        if (STEREO_USE_RIGHT_FACTORS)
            ROS_INFO("stereo estimator uses fixed camera 0 and camera 1 reprojection factors");
        else
            ROS_INFO("stereo estimator uses camera 0 residuals only; feature message camera stride is 2");
    }
    
    fsSettings.release();
}
