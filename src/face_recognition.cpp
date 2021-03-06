#include <rgbd/View.h>
#include <opencv2/highgui/highgui.hpp>

#include <geolib/datatypes.h>
#include <geolib/ros/msg_conversions.h>

#include <openface_ros/DetectFace.h>
#include <openface_ros/LearnFace.h>

#include <ed_perception/RecognizePerson.h>
#include <ed_perception/LearnPerson.h>

#include <ed/kinect/image_buffer.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <ros/service_client.h>
#include <ros/node_handle.h>
#include <ros/console.h>

#include <std_srvs/Empty.h>

ImageBuffer image_buffer;
ros::ServiceClient client_detect_face;
ros::ServiceClient client_learn_face;
ros::ServiceClient client_clear_faces;

// ----------------------------------------------------------------------------------------------------

bool srvClearPersons(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res)
{
    return client_clear_faces.call(req, res);
}

// ----------------------------------------------------------------------------------------------------

bool srvLearnPerson(ed_perception::LearnPerson::Request& req, ed_perception::LearnPerson::Response& res)
{
    rgbd::ImageConstPtr image;
    geo::Pose3D sensor_pose;

    if (!image_buffer.waitForRecentImage("/map", image, sensor_pose, 2.0))
    {
        res.error_msg = "Could not capture image";
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }

    openface_ros::LearnFace srv;
    srv.request.name = req.person_name;

    cv_bridge::CvImage image_msg;
    image_msg.encoding = sensor_msgs::image_encodings::BGR8;
    image_msg.image = image->getRGBImage();
    srv.request.image = *image_msg.toImageMsg();

    if (!client_learn_face.call(srv))
    {
        res.error_msg = "Could not call openface server (learn)";
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }

    if (!srv.response.error_msg.empty())
    {
        res.error_msg = "OpenFace server (learn) responded with error: " + srv.response.error_msg;
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }

    return true;
}

// ----------------------------------------------------------------------------------------------------

bool srvRecognizePerson(ed_perception::RecognizePerson::Request& req, ed_perception::RecognizePerson::Response& res)
{
    rgbd::ImageConstPtr image;
    geo::Pose3D sensor_pose;

    if (!image_buffer.waitForRecentImage("/map", image, sensor_pose, 2.0))
    {
        res.error_msg = "Could not capture image";
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }

    ROS_INFO_STREAM("srvRecognizePerson: sensor pose: " << sensor_pose);

    openface_ros::DetectFace srv;

    cv_bridge::CvImage image_msg;
    image_msg.encoding = sensor_msgs::image_encodings::BGR8;
    image_msg.image = image->getRGBImage();
    srv.request.image = *image_msg.toImageMsg();
    srv.request.external_api_request = req.external_api_request;

//    cv::imshow("face_detection", image->getRGBImage());
//    cv::waitKey();

    if (!client_detect_face.call(srv))
    {
        res.error_msg = "Could not call OpenFace server (detect)";
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }

    if (!srv.response.error_msg.empty())
    {
        res.error_msg = "OpenFace server (detect) responded with error: " + srv.response.error_msg;
        ROS_ERROR("%s", res.error_msg.c_str());
        return true;
    }


    ROS_INFO_STREAM("Openface found " << srv.response.face_detections.size() << " faces");

    for(unsigned int i = 0; i < srv.response.face_detections.size(); ++i)
    {
        const openface_ros::FaceDetection& det_of = srv.response.face_detections[i];

        res.person_detections.push_back(ed_perception::PersonDetection());
        ed_perception::PersonDetection& det = res.person_detections.back();

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Find the name with the lowest error

        double min_error = 1e9;
        for(unsigned int j = 0; j < det_of.names.size(); ++j)
        {
            if (det_of.l2_distances[j] < min_error)
            {
                min_error = det_of.l2_distances[j];
                det.name = det_of.names[j];
            }
        }

        det.name_score = -min_error;
        det.gender_score = det_of.gender_score;
        det.age_score = det_of.age_score;

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get face depth ROI

        const cv::Mat& depth = image->getDepthImage();

        cv::Rect roi_rgb(det_of.x, det_of.y, det_of.width, det_of.height);
        float rgb_depth_width_ratio = (float)(depth.cols) / image->getRGBImage().cols;
        cv::Rect roi_depth(rgb_depth_width_ratio * roi_rgb.tl(), rgb_depth_width_ratio * roi_rgb.br());
        cv::Point roi_depth_center = 0.5 * (roi_depth.tl() + roi_depth.br());

        cv::Rect roi_depth_capped(cv::Point(std::max(0, roi_depth.x),
                                            std::max(0, roi_depth.y)),
                                  cv::Point(std::min(depth.cols - 1, roi_depth.br().x),
                                            std::min(depth.rows - 1, roi_depth.br().y)));

        cv::Mat face_depth = depth(roi_depth_capped);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get median depth of face ROI

        std::vector<float> depths;
        for(unsigned int j = 0; j < face_depth.cols * face_depth.rows; ++j)
        {
            float d = face_depth.at<float>(j);
            if (d > 0 && d == d)
                depths.push_back(d);
        }

        if (depths.empty())
        {
            res.person_detections.pop_back();
            continue;
        }

        std::sort(depths.begin(), depths.end());
        float median_depth = depths[depths.size() / 2];

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine face location

        rgbd::View view(*image, image->getDepthImage().cols);
        geo::Vec3 face_pos = view.getRasterizer().project2Dto3D(roi_depth_center.x, roi_depth_center.y) * median_depth;

        geo::Pose3D pose = geo::Pose3D::identity();
        pose.t = face_pos;

        geo::Pose3D pose_MAP = sensor_pose * pose;

        ROS_INFO_STREAM("Face " << i << ": position in camera frame: " << face_pos << ", in map frame: " << pose_MAP.t);

        det.pose.header.frame_id = "/map";
        geo::convert(pose_MAP, det.pose.pose);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Save images

        std::stringstream ss_filename;
        ss_filename << "/tmp/face-" << i;

        cv::Mat debug_rgb = image->getRGBImage().clone();
        cv::rectangle(debug_rgb, roi_rgb, cv::Scalar(255, 255, 255), 2);

        cv::Mat debug_depth = depth.clone();
        cv::rectangle(debug_depth, roi_depth_capped, cv::Scalar(10), 2);

        cv::imwrite(ss_filename.str() + "-rgb.jpg", debug_rgb);
        cv::imwrite(ss_filename.str() + "-depth.jpg", debug_depth / 10);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // TODO

        det.age = det_of.age;

        if (det_of.gender_is_male)
            det.gender = ed_perception::PersonDetection::MALE;
        else
            det.gender = ed_perception::PersonDetection::FEMALE;

        det.body_pose = "";
    }

    return true;
}

// ----------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    ros::init(argc, argv, "face_recognition");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    std::string rgbd_topic;
    if (!nh_private.getParam("camera_topic", rgbd_topic))
    {
        ROS_ERROR("Parameters 'camera_topic' not set");
        return 0;
    }

    image_buffer.initialize(rgbd_topic);

    client_learn_face = nh.serviceClient<openface_ros::LearnFace>("face/learn");
    client_detect_face = nh.serviceClient<openface_ros::DetectFace>("face/detect");
    client_clear_faces = nh.serviceClient<std_srvs::Empty>("face/clear");

    ros::ServiceServer srv_learn_person = nh.advertiseService("learn_person", srvLearnPerson);
    ros::ServiceServer srv_recognize_person = nh.advertiseService("recognize_person", srvRecognizePerson);
    ros::ServiceServer srv_clear_persons = nh.advertiseService("clear_persons", srvClearPersons);

    ros::spin();

    return 0;
}
