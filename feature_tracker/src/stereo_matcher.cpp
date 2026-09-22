#include "stereo_matcher.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <eigen3/Eigen/Dense>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace
{
bool pointInBorder(const cv::Point2f &point, int width, int height)
{
    const int border = 1;
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    return border <= x && x < width - border && border <= y && y < height - border;
}

float quietNaN()
{
    return std::numeric_limits<float>::quiet_NaN();
}
}

StereoCandidateEvaluation evaluateStereoCandidate(const StereoCandidateInput &input,
                                                   const StereoMatcherOptions &options)
{
    StereoCandidateEvaluation result;
    if (!input.forward_ok)
    {
        result.reject_code = STEREO_REJECT_FLOW;
        return result;
    }
    if (!pointInBorder(input.right_pixel, input.image_width, input.image_height))
    {
        result.reject_code = STEREO_REJECT_BORDER;
        return result;
    }
    if (!input.backward_ok)
    {
        result.reject_code = STEREO_REJECT_FLOW;
        return result;
    }

    result.fb_error = cv::norm(input.left_pixel - input.backward_pixel);
    if (!std::isfinite(result.fb_error) || result.fb_error > options.fb_threshold)
    {
        result.reject_code = STEREO_REJECT_FB;
        return result;
    }

    result.epipolar_error = std::abs(input.left_pixel.y - input.right_pixel.y);
    if (!std::isfinite(result.epipolar_error) ||
        result.epipolar_error > options.epipolar_threshold)
    {
        result.reject_code = STEREO_REJECT_EPIPOLAR;
        return result;
    }

    const double disparity = input.left_normalized.x - input.right_normalized.x;
    if (!std::isfinite(disparity) || disparity <= 0.0)
    {
        result.reject_code = STEREO_REJECT_DISPARITY;
        return result;
    }

    result.depth = static_cast<float>(options.baseline_x / disparity);
    if (!std::isfinite(result.depth) || result.depth < options.min_depth ||
        result.depth > options.max_depth)
    {
        result.reject_code = STEREO_REJECT_DEPTH_RANGE;
        return result;
    }

    result.reject_code = STEREO_ACCEPTED;
    return result;
}

StereoMatcher::StereoMatcher(const camodocal::CameraPtr &right_camera,
                             const StereoMatcherOptions &options)
    : right_camera_(right_camera), options_(options)
{
    if (!right_camera_)
        throw std::invalid_argument("right camera is required for stereo matching");
}

std::vector<StereoMatch> StereoMatcher::match(const cv::Mat &left_image,
                                              const cv::Mat &right_image,
                                              const std::vector<cv::Point2f> &left_pixels,
                                              const std::vector<cv::Point2f> &left_normalized,
                                              const std::vector<int> &feature_ids,
                                              double timestamp)
{
    if (left_pixels.size() != left_normalized.size() || left_pixels.size() != feature_ids.size())
        throw std::invalid_argument("stereo matcher input vectors have different sizes");
    if (left_image.empty() || right_image.empty() || left_image.size() != right_image.size())
        throw std::invalid_argument("stereo matcher images are empty or have different sizes");

    stats_.frames++;
    stats_.candidates += left_pixels.size();

    std::vector<StereoMatch> matches(left_pixels.size());
    if (left_pixels.empty())
    {
        previous_right_normalized_.clear();
        previous_timestamp_ = timestamp;
        return matches;
    }

    cv::Mat prepared_right;
    if (options_.equalize)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(right_image, prepared_right);
    }
    else
    {
        prepared_right = right_image;
    }

    std::vector<cv::Point2f> right_pixels = left_pixels;
    for (std::size_t index = 0; index < feature_ids.size(); ++index)
    {
        const std::map<int, StereoPixelHistory>::const_iterator previous =
            previous_pixel_history_.find(feature_ids[index]);
        if (previous != previous_pixel_history_.end())
        {
            right_pixels[index] = previous->second.right_pixel +
                (left_pixels[index] - previous->second.left_pixel);
        }
        else if (std::isfinite(previous_median_disparity_) && previous_median_disparity_ > 0.0f)
        {
            right_pixels[index].x -= previous_median_disparity_;
        }
    }
    std::vector<uchar> forward_status;
    std::vector<float> forward_error;
    cv::calcOpticalFlowPyrLK(left_image, prepared_right, left_pixels, right_pixels,
                             forward_status, forward_error, cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                              30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<cv::Point2f> reverse_input;
    std::vector<std::size_t> reverse_indices;
    for (std::size_t index = 0; index < left_pixels.size(); ++index)
    {
        if (forward_status[index] &&
            pointInBorder(right_pixels[index], right_image.cols, right_image.rows))
        {
            reverse_input.push_back(right_pixels[index]);
            reverse_indices.push_back(index);
        }
    }

    std::vector<cv::Point2f> backward_pixels(left_pixels.size(), cv::Point2f(quietNaN(), quietNaN()));
    std::vector<uchar> backward_status(left_pixels.size(), 0);
    if (!reverse_input.empty())
    {
        std::vector<cv::Point2f> reverse_output;
        reverse_output.reserve(reverse_indices.size());
        for (const std::size_t original_index : reverse_indices)
            reverse_output.push_back(left_pixels[original_index]);
        std::vector<uchar> reverse_status;
        std::vector<float> reverse_error;
        cv::calcOpticalFlowPyrLK(prepared_right, left_image, reverse_input, reverse_output,
                                 reverse_status, reverse_error, cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                                  30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        for (std::size_t reverse_index = 0; reverse_index < reverse_indices.size(); ++reverse_index)
        {
            const std::size_t original_index = reverse_indices[reverse_index];
            backward_status[original_index] = reverse_status[reverse_index];
            backward_pixels[original_index] = reverse_output[reverse_index];
        }
    }

    const double delta_time = timestamp - previous_timestamp_;
    const bool velocity_time_valid = previous_timestamp_ >= 0.0 && delta_time > 0.0;
    std::map<int, cv::Point2f> current_right_normalized;
    std::map<int, StereoPixelHistory> current_pixel_history;
    std::vector<float> accepted_disparities;

    for (std::size_t index = 0; index < matches.size(); ++index)
    {
        StereoMatch &match = matches[index];
        match.feature_id = feature_ids[index];
        match.left_pixel = left_pixels[index];
        match.left_normalized = left_normalized[index];
        match.right_pixel = cv::Point2f(quietNaN(), quietNaN());
        match.right_normalized = cv::Point2f(quietNaN(), quietNaN());
        match.right_velocity = cv::Point2f(0.0f, 0.0f);

        StereoCandidateInput candidate;
        candidate.forward_ok = forward_status[index] != 0;
        candidate.backward_ok = backward_status[index] != 0;
        candidate.left_pixel = left_pixels[index];
        candidate.right_pixel = right_pixels[index];
        candidate.backward_pixel = backward_pixels[index];
        candidate.left_normalized = left_normalized[index];
        candidate.image_width = right_image.cols;
        candidate.image_height = right_image.rows;

        if (candidate.forward_ok)
            match.right_pixel = right_pixels[index];

        if (candidate.forward_ok && pointInBorder(candidate.right_pixel, right_image.cols, right_image.rows))
        {
            Eigen::Vector3d ray;
            right_camera_->liftProjective(
                Eigen::Vector2d(candidate.right_pixel.x, candidate.right_pixel.y), ray);
            candidate.right_normalized = cv::Point2f(ray.x() / ray.z(), ray.y() / ray.z());
            match.right_normalized = candidate.right_normalized;
        }

        const StereoCandidateEvaluation evaluation = evaluateStereoCandidate(candidate, options_);
        match.reject_code = evaluation.reject_code;
        match.fb_error = evaluation.fb_error;
        match.epipolar_error = evaluation.epipolar_error;
        match.depth = evaluation.depth;
        stats_.outcomes[static_cast<std::size_t>(match.reject_code)]++;

        if (match.reject_code == STEREO_ACCEPTED)
        {
            const std::map<int, cv::Point2f>::const_iterator previous =
                previous_right_normalized_.find(match.feature_id);
            if (velocity_time_valid && previous != previous_right_normalized_.end())
                match.right_velocity = (match.right_normalized - previous->second) / delta_time;
            current_right_normalized[match.feature_id] = match.right_normalized;
            StereoPixelHistory history;
            history.left_pixel = match.left_pixel;
            history.right_pixel = match.right_pixel;
            current_pixel_history[match.feature_id] = history;
            accepted_disparities.push_back(match.left_pixel.x - match.right_pixel.x);
        }
    }

    if (!accepted_disparities.empty())
    {
        const std::size_t middle = accepted_disparities.size() / 2;
        std::nth_element(accepted_disparities.begin(), accepted_disparities.begin() + middle,
                         accepted_disparities.end());
        previous_median_disparity_ = accepted_disparities[middle];
    }

    previous_pixel_history_.swap(current_pixel_history);
    previous_right_normalized_.swap(current_right_normalized);
    previous_timestamp_ = timestamp;
    return matches;
}

void StereoMatcher::reset()
{
    previous_pixel_history_.clear();
    previous_right_normalized_.clear();
    previous_median_disparity_ = quietNaN();
    previous_timestamp_ = -1.0;
}

const StereoMatchStats &StereoMatcher::stats() const
{
    return stats_;
}
