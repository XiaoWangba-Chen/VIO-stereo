#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <deque>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

namespace
{
const std::size_t SYNC_QUEUE_LIMIT = 100;

struct StereoSyncStats
{
    uint64_t left_received = 0;
    uint64_t right_received = 0;
    uint64_t paired = 0;
    uint64_t left_unpaired = 0;
    uint64_t right_unpaired = 0;
    uint64_t out_of_order = 0;
    uint64_t queue_overflow = 0;
    uint64_t restart_count = 0;
    std::size_t max_queue_depth = 0;
};

std::deque<sensor_msgs::ImageConstPtr> left_image_buf;
std::deque<sensor_msgs::ImageConstPtr> right_image_buf;
StereoSyncStats sync_stats;
ros::Time last_left_received;
ros::Time last_right_received;
bool image_dimensions_checked = false;
}

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[MAX_NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = false;

namespace
{
void printSyncStats()
{
    if (NUM_OF_CAM != 2)
        return;
    ROS_INFO("stereo sync: left=%llu right=%llu paired=%llu left_unpaired=%llu "
             "right_unpaired=%llu out_of_order=%llu overflow=%llu restarts=%llu max_queue=%zu",
             static_cast<unsigned long long>(sync_stats.left_received),
             static_cast<unsigned long long>(sync_stats.right_received),
             static_cast<unsigned long long>(sync_stats.paired),
             static_cast<unsigned long long>(sync_stats.left_unpaired),
             static_cast<unsigned long long>(sync_stats.right_unpaired),
             static_cast<unsigned long long>(sync_stats.out_of_order),
             static_cast<unsigned long long>(sync_stats.queue_overflow),
             static_cast<unsigned long long>(sync_stats.restart_count),
             sync_stats.max_queue_depth);
}

void publishRestart()
{
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);
    sync_stats.restart_count++;
}

void resetTrackerForDiscontinuity()
{
    ROS_WARN("image stream discontinued; reset the feature tracker");
    first_image_flag = true;
    first_image_time = 0;
    last_image_time = 0;
    pub_count = 1;
    init_pub = false;
    trackerData[0] = FeatureTracker();
    trackerData[0].readIntrinsicParameter(CAM_NAMES[0]);
    publishRestart();
}

cv_bridge::CvImageConstPtr toMono8(const sensor_msgs::ImageConstPtr &img_msg)
{
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image image = *img_msg;
        image.encoding = sensor_msgs::image_encodings::MONO8;
        return cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::MONO8);
    }
    return cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
}

bool validateImageDimensions(const sensor_msgs::ImageConstPtr &left_msg,
                             const sensor_msgs::ImageConstPtr &right_msg)
{
    if (left_msg->width != static_cast<uint32_t>(COL) || left_msg->height != static_cast<uint32_t>(ROW))
    {
        ROS_FATAL("left image is %ux%u but calibration is %dx%d",
                  left_msg->width, left_msg->height, COL, ROW);
        ros::shutdown();
        return false;
    }
    if (right_msg &&
        (right_msg->width != left_msg->width || right_msg->height != left_msg->height))
    {
        ROS_FATAL("stereo image sizes differ: left=%ux%u right=%ux%u",
                  left_msg->width, left_msg->height, right_msg->width, right_msg->height);
        ros::shutdown();
        return false;
    }
    image_dimensions_checked = true;
    return true;
}

void processImage(const sensor_msgs::ImageConstPtr &img_msg,
                  const sensor_msgs::ImageConstPtr &right_msg = sensor_msgs::ImageConstPtr())
{
    if (!image_dimensions_checked && !validateImageDimensions(img_msg, right_msg))
        return;

    cv_bridge::CvImageConstPtr right_ptr;
    cv_bridge::CvImageConstPtr ptr;
    try
    {
        ptr = toMono8(img_msg);
        if (right_msg)
            right_ptr = toMono8(right_msg); // Phase B0 validates decoding only.
        (void)right_ptr;
    }
    catch (const cv_bridge::Exception &error)
    {
        ROS_FATAL("failed to decode stereo image: %s", error.what());
        ros::shutdown();
        return;
    }

    const double image_time = img_msg->header.stamp.toSec();
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = image_time;
        last_image_time = image_time;
        return;
    }
    if (image_time - last_image_time > 1.0 || image_time < last_image_time)
    {
        resetTrackerForDiscontinuity();
        return;
    }
    last_image_time = image_time;

    if (round(1.0 * pub_count / (image_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        if (abs(1.0 * pub_count / (image_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = image_time;
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    TicToc t_r;
    trackerData[0].readImage(ptr->image, image_time);

#if SHOW_UNDISTORTION
    trackerData[0].showUndistortion("undistrotion_0");
#endif

    for (unsigned int i = 0; trackerData[0].updateID(i); i++)
    {
    }

    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        const auto &un_pts = trackerData[0].cur_un_pts;
        const auto &cur_pts = trackerData[0].cur_pts;
        const auto &ids = trackerData[0].ids;
        const auto &pts_velocity = trackerData[0].pts_velocity;
        for (unsigned int j = 0; j < ids.size(); j++)
        {
            if (trackerData[0].track_cnt[j] <= 1)
                continue;

            geometry_msgs::Point32 point;
            point.x = un_pts[j].x;
            point.y = un_pts[j].y;
            point.z = 1;
            feature_points->points.push_back(point);
            id_of_point.values.push_back(ids[j] * NUM_OF_CAM); // camera_id is 0 in A1/B0.
            u_of_point.values.push_back(cur_pts[j].x);
            v_of_point.values.push_back(cur_pts[j].y);
            velocity_x_of_point.values.push_back(pts_velocity[j].x);
            velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        if (!init_pub)
            init_pub = true;
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            cv_bridge::CvImagePtr color = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++)
            {
                const double len = std::min(1.0, 1.0 * trackerData[0].track_cnt[j] / WINDOW_SIZE);
                cv::circle(color->image, trackerData[0].cur_pts[j], 2,
                           cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }
            pub_match.publish(color->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

void clearStereoQueuesAndRestart()
{
    sync_stats.left_unpaired += left_image_buf.size();
    sync_stats.right_unpaired += right_image_buf.size();
    left_image_buf.clear();
    right_image_buf.clear();
    last_left_received = ros::Time();
    last_right_received = ros::Time();
    resetTrackerForDiscontinuity();
}

bool checkIncomingStamp(const ros::Time &stamp, ros::Time &last_stamp)
{
    if (!last_stamp.isZero() && stamp <= last_stamp)
    {
        sync_stats.out_of_order++;
        ROS_WARN("discard out-of-order stereo image at %.9f", stamp.toSec());
        return false;
    }
    if (!last_stamp.isZero() && (stamp - last_stamp).toSec() > 1.0)
        clearStereoQueuesAndRestart();
    last_stamp = stamp;
    return true;
}

void drainStereoQueues()
{
    while (!left_image_buf.empty() && !right_image_buf.empty())
    {
        const ros::Time left_stamp = left_image_buf.front()->header.stamp;
        const ros::Time right_stamp = right_image_buf.front()->header.stamp;
        if (left_stamp == right_stamp)
        {
            const sensor_msgs::ImageConstPtr left = left_image_buf.front();
            const sensor_msgs::ImageConstPtr right = right_image_buf.front();
            left_image_buf.pop_front();
            right_image_buf.pop_front();
            sync_stats.paired++;
            processImage(left, right);
            if (sync_stats.paired % 300 == 0)
                printSyncStats();
        }
        else if (left_stamp < right_stamp)
        {
            left_image_buf.pop_front();
            sync_stats.left_unpaired++;
        }
        else
        {
            right_image_buf.pop_front();
            sync_stats.right_unpaired++;
        }
    }
}

void enforceQueueLimit(std::deque<sensor_msgs::ImageConstPtr> &queue, uint64_t &unpaired_count)
{
    while (queue.size() > SYNC_QUEUE_LIMIT)
    {
        queue.pop_front();
        sync_stats.queue_overflow++;
        unpaired_count++;
    }
}

void leftImageCallback(const sensor_msgs::ImageConstPtr &image)
{
    sync_stats.left_received++;
    if (!checkIncomingStamp(image->header.stamp, last_left_received))
        return;
    left_image_buf.push_back(image);
    enforceQueueLimit(left_image_buf, sync_stats.left_unpaired);
    sync_stats.max_queue_depth = std::max(sync_stats.max_queue_depth, left_image_buf.size());
    drainStereoQueues();
}

void monoImageCallback(const sensor_msgs::ImageConstPtr &image)
{
    processImage(image);
}

void rightImageCallback(const sensor_msgs::ImageConstPtr &image)
{
    sync_stats.right_received++;
    if (!checkIncomingStamp(image->header.stamp, last_right_received))
        return;
    right_image_buf.push_back(image);
    enforceQueueLimit(right_image_buf, sync_stats.right_unpaired);
    sync_stats.max_queue_depth = std::max(sync_stats.max_queue_depth, right_image_buf.size());
    drainStereoQueues();
}
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle node("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    try
    {
        readParameters(node);
    }
    catch (const std::exception &error)
    {
        ROS_FATAL("invalid feature tracker configuration: %s", error.what());
        return 1;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if (FISHEYE)
    {
        trackerData[0].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if (!trackerData[0].fisheye_mask.data)
        {
            ROS_FATAL("failed to load fisheye mask");
            return 1;
        }
    }

    pub_img = node.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = node.advertise<sensor_msgs::Image>("feature_img", 1000);
    pub_restart = node.advertise<std_msgs::Bool>("restart", 1000);

    ros::Subscriber mono_subscriber;
    ros::Subscriber left_subscriber;
    ros::Subscriber right_subscriber;
    if (NUM_OF_CAM == 1)
    {
        mono_subscriber = node.subscribe(IMAGE_TOPIC, 100, monoImageCallback);
    }
    else
    {
        left_subscriber = node.subscribe(IMAGE_TOPIC, 100, leftImageCallback);
        right_subscriber = node.subscribe(IMAGE1_TOPIC, 100, rightImageCallback);
        ROS_INFO("strict stereo synchronization enabled: left=%s right=%s",
                 IMAGE_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
    }

    ros::spin();
    sync_stats.left_unpaired += left_image_buf.size();
    sync_stats.right_unpaired += right_image_buf.size();
    left_image_buf.clear();
    right_image_buf.clear();
    printSyncStats();
    if (NUM_OF_CAM == 2)
    {
        std::fprintf(stdout,
                     "FINAL stereo sync: left=%llu right=%llu paired=%llu left_unpaired=%llu "
                     "right_unpaired=%llu out_of_order=%llu overflow=%llu restarts=%llu max_queue=%zu\n",
                     static_cast<unsigned long long>(sync_stats.left_received),
                     static_cast<unsigned long long>(sync_stats.right_received),
                     static_cast<unsigned long long>(sync_stats.paired),
                     static_cast<unsigned long long>(sync_stats.left_unpaired),
                     static_cast<unsigned long long>(sync_stats.right_unpaired),
                     static_cast<unsigned long long>(sync_stats.out_of_order),
                     static_cast<unsigned long long>(sync_stats.queue_overflow),
                     static_cast<unsigned long long>(sync_stats.restart_count),
                     sync_stats.max_queue_depth);
        std::fflush(stdout);
    }
    return 0;
}
