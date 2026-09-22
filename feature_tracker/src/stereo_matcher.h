#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include <opencv2/core/core.hpp>

#include "camodocal/camera_models/Camera.h"

enum StereoRejectCode
{
    STEREO_ACCEPTED = 0,
    STEREO_REJECT_FLOW = 1,
    STEREO_REJECT_BORDER = 2,
    STEREO_REJECT_FB = 3,
    STEREO_REJECT_EPIPOLAR = 4,
    STEREO_REJECT_DISPARITY = 5,
    STEREO_REJECT_DEPTH_RANGE = 6,
    STEREO_REJECT_CODE_COUNT = 7
};

struct StereoMatcherOptions
{
    double baseline_x = 0.0;
    double fb_threshold = 0.5;
    double epipolar_threshold = 1.0;
    double min_depth = 0.2;
    double max_depth = 15.0;
    bool equalize = false;
};

struct StereoCandidateInput
{
    bool forward_ok = false;
    bool backward_ok = false;
    cv::Point2f left_pixel;
    cv::Point2f right_pixel;
    cv::Point2f backward_pixel;
    cv::Point2f left_normalized;
    cv::Point2f right_normalized;
    int image_width = 0;
    int image_height = 0;
};

struct StereoCandidateEvaluation
{
    StereoRejectCode reject_code = STEREO_REJECT_FLOW;
    float fb_error = std::numeric_limits<float>::quiet_NaN();
    float epipolar_error = std::numeric_limits<float>::quiet_NaN();
    float depth = std::numeric_limits<float>::quiet_NaN();
};

struct StereoMatch
{
    int feature_id = -1;
    cv::Point2f left_pixel;
    cv::Point2f left_normalized;
    cv::Point2f right_pixel;
    cv::Point2f right_normalized;
    cv::Point2f right_velocity;
    float depth = std::numeric_limits<float>::quiet_NaN();
    float fb_error = std::numeric_limits<float>::quiet_NaN();
    float epipolar_error = std::numeric_limits<float>::quiet_NaN();
    StereoRejectCode reject_code = STEREO_REJECT_FLOW;
};

struct StereoMatchStats
{
    uint64_t frames = 0;
    uint64_t candidates = 0;
    std::array<uint64_t, STEREO_REJECT_CODE_COUNT> outcomes{{0, 0, 0, 0, 0, 0, 0}};
};

StereoCandidateEvaluation evaluateStereoCandidate(const StereoCandidateInput &input,
                                                   const StereoMatcherOptions &options);

class StereoMatcher
{
  public:
    StereoMatcher(const camodocal::CameraPtr &right_camera,
                  const StereoMatcherOptions &options);

    std::vector<StereoMatch> match(const cv::Mat &left_image,
                                   const cv::Mat &right_image,
                                   const std::vector<cv::Point2f> &left_pixels,
                                   const std::vector<cv::Point2f> &left_normalized,
                                   const std::vector<int> &feature_ids,
                                   double timestamp);

    void reset();

    const StereoMatchStats &stats() const;

  private:
    struct StereoPixelHistory
    {
        cv::Point2f left_pixel;
        cv::Point2f right_pixel;
    };

    camodocal::CameraPtr right_camera_;
    StereoMatcherOptions options_;
    std::map<int, StereoPixelHistory> previous_pixel_history_;
    std::map<int, cv::Point2f> previous_right_normalized_;
    float previous_median_disparity_ = std::numeric_limits<float>::quiet_NaN();
    double previous_timestamp_ = -1.0;
    StereoMatchStats stats_;
};
