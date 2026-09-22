#include <gtest/gtest.h>

#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/camera_models/PinholeCamera.h"
#include "stereo_matcher.h"

namespace
{
const int kWidth = 320;
const int kHeight = 240;
const double kFocal = 200.0;

StereoMatcherOptions defaultOptions()
{
    StereoMatcherOptions options;
    options.baseline_x = 0.1;
    options.fb_threshold = 0.5;
    options.epipolar_threshold = 1.0;
    options.min_depth = 0.2;
    options.max_depth = 15.0;
    return options;
}

StereoCandidateInput validCandidate()
{
    StereoCandidateInput input;
    input.forward_ok = true;
    input.backward_ok = true;
    input.left_pixel = cv::Point2f(120.0f, 80.0f);
    input.right_pixel = cv::Point2f(110.0f, 80.0f);
    input.backward_pixel = input.left_pixel;
    input.left_normalized = cv::Point2f(0.1f, 0.0f);
    input.right_normalized = cv::Point2f(0.05f, 0.0f);
    input.image_width = kWidth;
    input.image_height = kHeight;
    return input;
}

cv::Mat translateImage(const cv::Mat &image, double x)
{
    cv::Mat result;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, x, 0.0, 1.0, 0.0);
    cv::warpAffine(image, result, transform, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    return result;
}
}

TEST(StereoCandidateEvaluation, AcceptsValidGeometry)
{
    const StereoCandidateEvaluation result = evaluateStereoCandidate(validCandidate(), defaultOptions());
    EXPECT_EQ(STEREO_ACCEPTED, result.reject_code);
    EXPECT_NEAR(2.0, result.depth, 1e-6);
}

TEST(StereoCandidateEvaluation, RejectsInStableOrder)
{
    StereoCandidateInput input = validCandidate();
    input.forward_ok = false;
    EXPECT_EQ(STEREO_REJECT_FLOW, evaluateStereoCandidate(input, defaultOptions()).reject_code);

    input = validCandidate();
    input.right_pixel.x = 0.0f;
    EXPECT_EQ(STEREO_REJECT_BORDER, evaluateStereoCandidate(input, defaultOptions()).reject_code);

    input = validCandidate();
    input.backward_pixel.x += 0.6f;
    EXPECT_EQ(STEREO_REJECT_FB, evaluateStereoCandidate(input, defaultOptions()).reject_code);

    input = validCandidate();
    input.right_pixel.y += 1.1f;
    EXPECT_EQ(STEREO_REJECT_EPIPOLAR, evaluateStereoCandidate(input, defaultOptions()).reject_code);

    input = validCandidate();
    input.right_normalized.x = input.left_normalized.x;
    EXPECT_EQ(STEREO_REJECT_DISPARITY, evaluateStereoCandidate(input, defaultOptions()).reject_code);

    input = validCandidate();
    input.right_normalized.x = 0.099f;
    EXPECT_EQ(STEREO_REJECT_DEPTH_RANGE, evaluateStereoCandidate(input, defaultOptions()).reject_code);
}

TEST(StereoMatcher, TracksHorizontalShiftWithoutMutatingLeftState)
{
    cv::Mat left(kHeight, kWidth, CV_8UC1);
    cv::randu(left, 0, 255);
    cv::GaussianBlur(left, left, cv::Size(3, 3), 0.8);
    const cv::Mat right = translateImage(left, -8.0);

    std::vector<cv::Point2f> pixels;
    std::vector<cv::Point2f> normalized;
    std::vector<int> ids;
    for (int y = 50; y <= 190; y += 35)
    {
        for (int x = 70; x <= 250; x += 45)
        {
            pixels.push_back(cv::Point2f(x, y));
            normalized.push_back(cv::Point2f((x - 160.0) / kFocal, (y - 120.0) / kFocal));
            ids.push_back(static_cast<int>(ids.size()) + 10);
        }
    }
    const std::vector<cv::Point2f> original_pixels = pixels;
    const std::vector<int> original_ids = ids;

    camodocal::CameraPtr camera(new camodocal::PinholeCamera(
        "right", kWidth, kHeight, 0.0, 0.0, 0.0, 0.0,
        kFocal, kFocal, 160.0, 120.0));
    StereoMatcher matcher(camera, defaultOptions());
    const std::vector<StereoMatch> matches = matcher.match(
        left, right, pixels, normalized, ids, 1.0);

    ASSERT_EQ(pixels.size(), matches.size());
    EXPECT_EQ(original_pixels, pixels);
    EXPECT_EQ(original_ids, ids);
    int accepted = 0;
    for (std::size_t index = 0; index < matches.size(); ++index)
    {
        EXPECT_EQ(ids[index], matches[index].feature_id);
        if (matches[index].reject_code != STEREO_ACCEPTED)
            continue;
        accepted++;
        EXPECT_NEAR(pixels[index].x - 8.0, matches[index].right_pixel.x, 0.25);
        EXPECT_NEAR(2.5, matches[index].depth, 0.08);
        EXPECT_FLOAT_EQ(0.0f, matches[index].right_velocity.x);
    }
    EXPECT_GE(accepted, static_cast<int>(matches.size() * 4 / 5));
}

TEST(StereoMatcher, VelocityRequiresConsecutiveMatchesAndResetClearsIt)
{
    cv::Mat base(kHeight, kWidth, CV_8UC1);
    cv::randu(base, 0, 255);
    cv::GaussianBlur(base, base, cv::Size(3, 3), 0.8);
    camodocal::CameraPtr camera(new camodocal::PinholeCamera(
        "right", kWidth, kHeight, 0.0, 0.0, 0.0, 0.0,
        kFocal, kFocal, 160.0, 120.0));
    StereoMatcher matcher(camera, defaultOptions());

    std::vector<cv::Point2f> pixels(1, cv::Point2f(150.0f, 120.0f));
    std::vector<cv::Point2f> normalized(1, cv::Point2f(-0.05f, 0.0f));
    std::vector<int> ids(1, 42);
    ASSERT_EQ(STEREO_ACCEPTED,
              matcher.match(base, translateImage(base, -8.0), pixels, normalized, ids, 1.0)[0].reject_code);

    const cv::Mat shifted_left = translateImage(base, 2.0);
    pixels[0].x += 2.0f;
    normalized[0].x += 2.0 / kFocal;
    const StereoMatch second = matcher.match(
        shifted_left, translateImage(base, -6.0), pixels, normalized, ids, 1.1)[0];
    ASSERT_EQ(STEREO_ACCEPTED, second.reject_code);
    EXPECT_NEAR((2.0 / kFocal) / 0.1, second.right_velocity.x, 0.02);

    matcher.reset();
    const StereoMatch after_reset = matcher.match(
        shifted_left, translateImage(base, -6.0), pixels, normalized, ids, 1.2)[0];
    ASSERT_EQ(STEREO_ACCEPTED, after_reset.reject_code);
    EXPECT_FLOAT_EQ(0.0f, after_reset.right_velocity.x);
    EXPECT_FLOAT_EQ(0.0f, after_reset.right_velocity.y);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
