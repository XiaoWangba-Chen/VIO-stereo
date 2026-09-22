#include "parameters.h"

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <stdexcept>

std::string IMAGE_TOPIC;
std::string IMAGE1_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
double STEREO_BASELINE_X = 0.0;
double STEREO_FB_THRESHOLD = 0.5;
double STEREO_EPIPOLAR_THRESHOLD = 1.0;
double STEREO_MIN_DEPTH = 0.2;
double STEREO_MAX_DEPTH = 15.0;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;
int NUM_OF_CAM = 1;
int STEREO_USE_DEPTH_INIT = 0;
int STEREO_USE_METRIC_INIT = 0;
int STEREO_USE_RIGHT_FACTORS = 0;

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
std::string resolveConfigPath(const std::string &config_file, const std::string &path)
{
    if (path.empty() || path[0] == '/')
        return path;
    const std::string::size_type separator = config_file.find_last_of("/\\");
    if (separator == std::string::npos)
        return path;
    return config_file.substr(0, separator + 1) + path;
}

int readOptionalInt(const cv::FileStorage &settings, const std::string &name, int default_value)
{
    const cv::FileNode node = settings[name];
    return node.empty() ? default_value : static_cast<int>(node);
}

double readOptionalDouble(const cv::FileStorage &settings, const std::string &name, double default_value)
{
    const cv::FileNode node = settings[name];
    return node.empty() ? default_value : static_cast<double>(node);
}

std::string readRequiredString(const cv::FileStorage &settings, const std::string &name)
{
    std::string value;
    const cv::FileNode node = settings[name];
    if (!node.empty())
        node >> value;
    if (value.empty())
        throw std::runtime_error("missing required stereo setting: " + name);
    return value;
}

struct PinholeMetadata
{
    int width;
    int height;
    double fx;
    double fy;
    double cx;
    double cy;
};

PinholeMetadata readPinholeMetadata(const std::string &path)
{
    cv::FileStorage calibration(path, cv::FileStorage::READ);
    if (!calibration.isOpened())
        throw std::runtime_error("cannot open camera calibration: " + path);

    std::string model_type;
    calibration["model_type"] >> model_type;
    if (model_type != "PINHOLE" && model_type != "pinhole")
        throw std::runtime_error("stereo camera model must be PINHOLE: " + path);

    PinholeMetadata result;
    result.width = static_cast<int>(calibration["image_width"]);
    result.height = static_cast<int>(calibration["image_height"]);
    const cv::FileNode projection = calibration["projection_parameters"];
    result.fx = static_cast<double>(projection["fx"]);
    result.fy = static_cast<double>(projection["fy"]);
    result.cx = static_cast<double>(projection["cx"]);
    result.cy = static_cast<double>(projection["cy"]);
    return result;
}

void validateStereoIntrinsics(const std::string &cam0_path, const std::string &cam1_path)
{
    const PinholeMetadata cam0 = readPinholeMetadata(cam0_path);
    const PinholeMetadata cam1 = readPinholeMetadata(cam1_path);
    if (cam0.width != cam1.width || cam0.height != cam1.height)
        throw std::runtime_error("stereo calibration resolutions differ");
    if (cam0.width != COL || cam0.height != ROW)
        throw std::runtime_error("stereo calibration resolution differs from the main configuration");

    const double fx_relative = std::abs(cam0.fx - cam1.fx) / std::max(std::abs(cam0.fx), 1.0);
    const double fy_relative = std::abs(cam0.fy - cam1.fy) / std::max(std::abs(cam0.fy), 1.0);
    if (fx_relative > 0.01 || fy_relative > 0.01)
        throw std::runtime_error("stereo focal lengths differ by more than 1 percent");
    if (std::abs(cam0.cy - cam1.cy) > 1.0)
        throw std::runtime_error("stereo principal-point y coordinates differ by more than 1 pixel");

    ROS_INFO("validated stereo intrinsics: %dx%d, fx=(%.9f, %.9f), fy=(%.9f, %.9f), cy=(%.9f, %.9f)",
             cam0.width, cam0.height, cam0.fx, cam1.fx, cam0.fy, cam1.fy, cam0.cy, cam1.cy);
}

double readAndValidateStereoBaseline(const cv::FileStorage &settings)
{
    Eigen::Matrix3d rotations[2];
    Eigen::Vector3d translations[2];
    for (int camera_id = 0; camera_id < 2; ++camera_id)
    {
        const std::string suffix = camera_id == 0 ? "" : "1";
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
            throw std::runtime_error("invalid stereo extrinsic dimensions");
        cv::cv2eigen(cv_rotation, rotations[camera_id]);
        cv::cv2eigen(cv_translation, translations[camera_id]);
        rotations[camera_id] = Eigen::Quaterniond(rotations[camera_id]).normalized().toRotationMatrix();
    }

    const Eigen::Matrix3d relative_rotation = rotations[1].transpose() * rotations[0];
    const double cosine = std::max(-1.0, std::min(1.0, (relative_rotation.trace() - 1.0) / 2.0));
    const double angle_degrees = std::acos(cosine) * 180.0 / 3.14159265358979323846;
    const Eigen::Vector3d baseline_cam0 = rotations[0].transpose() * (translations[1] - translations[0]);
    const double baseline = baseline_cam0.norm();
    if (angle_degrees > 1.0)
        throw std::runtime_error("stereo relative rotation exceeds 1 degree");
    if (baseline <= 0.01)
        throw std::runtime_error("stereo baseline must be greater than 1 cm");
    if (baseline_cam0.x() <= 0.0)
        throw std::runtime_error("stereo baseline x component must be positive");
    if (baseline_cam0.tail<2>().norm() > baseline * 0.05)
        throw std::runtime_error("stereo baseline y/z component exceeds 5 percent");

    ROS_INFO("stereo matcher baseline x=%.9f m, rotation=%.6f deg",
             baseline_cam0.x(), angle_degrees);
    return baseline_cam0.x();
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
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    NUM_OF_CAM = readOptionalInt(fsSettings, "num_of_cam", 1);
    if (NUM_OF_CAM < 1 || NUM_OF_CAM > MAX_NUM_OF_CAM)
        throw std::runtime_error("num_of_cam must be 1 or 2");
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.clear();
    std::string cam0_calib;
    if (!fsSettings["cam0_calib"].empty())
        fsSettings["cam0_calib"] >> cam0_calib;
    CAM_NAMES.push_back(cam0_calib.empty() ? config_file : resolveConfigPath(config_file, cam0_calib));

    STEREO_USE_DEPTH_INIT = readOptionalInt(fsSettings, "stereo_use_depth_init", 0);
    STEREO_USE_METRIC_INIT = readOptionalInt(fsSettings, "stereo_use_metric_init", 0);
    STEREO_USE_RIGHT_FACTORS = readOptionalInt(fsSettings, "stereo_use_right_factors", 0);
    STEREO_TRACK = readOptionalInt(fsSettings, "stereo_track", 0);
    STEREO_FB_THRESHOLD = readOptionalDouble(fsSettings, "stereo_fb_threshold", 0.5);
    STEREO_EPIPOLAR_THRESHOLD = readOptionalDouble(fsSettings, "stereo_epipolar_threshold", 1.0);
    STEREO_MIN_DEPTH = readOptionalDouble(fsSettings, "stereo_min_depth", 0.2);
    STEREO_MAX_DEPTH = readOptionalDouble(fsSettings, "stereo_max_depth", 15.0);
    if (STEREO_TRACK != 0 && STEREO_TRACK != 1)
        throw std::runtime_error("stereo_track must be 0 or 1");
    if (STEREO_FB_THRESHOLD <= 0.0 || STEREO_EPIPOLAR_THRESHOLD <= 0.0 ||
        STEREO_MIN_DEPTH <= 0.0 || STEREO_MAX_DEPTH <= STEREO_MIN_DEPTH)
        throw std::runtime_error("invalid stereo matching thresholds or depth range");
    if (STEREO_TRACK && NUM_OF_CAM != 2)
        throw std::runtime_error("stereo_track=1 requires num_of_cam=2");
    if (STEREO_USE_DEPTH_INIT && NUM_OF_CAM != 2)
        throw std::runtime_error("stereo_use_depth_init=1 requires num_of_cam=2");
    if (STEREO_USE_METRIC_INIT && NUM_OF_CAM != 2)
        throw std::runtime_error("stereo_use_metric_init=1 requires num_of_cam=2");
    if (STEREO_USE_METRIC_INIT && !STEREO_USE_DEPTH_INIT)
        throw std::runtime_error("stereo_use_metric_init=1 requires stereo_use_depth_init=1");
    if (STEREO_USE_RIGHT_FACTORS && NUM_OF_CAM != 2)
        throw std::runtime_error("stereo_use_right_factors=1 requires num_of_cam=2");

    if (NUM_OF_CAM == 2)
    {
        IMAGE1_TOPIC = readRequiredString(fsSettings, "image1_topic");
        const std::string cam1_calib = resolveConfigPath(config_file, readRequiredString(fsSettings, "cam1_calib"));
        CAM_NAMES.push_back(cam1_calib);
        if (STEREO_USE_DEPTH_INIT && !STEREO_TRACK)
            throw std::runtime_error("stereo_use_depth_init=1 requires stereo_track=1");
        if (STEREO_USE_RIGHT_FACTORS && !STEREO_TRACK)
            throw std::runtime_error("stereo_use_right_factors=1 requires stereo_track=1");
        validateStereoIntrinsics(CAM_NAMES[0], CAM_NAMES[1]);
        if (STEREO_TRACK)
            STEREO_BASELINE_X = readAndValidateStereoBaseline(fsSettings);
    }

    WINDOW_SIZE = 20;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
