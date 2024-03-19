//
// Created by sxj on 2024/3/17.
//

#ifndef UNTITLED10_KEYPOINT_H
#define UNTITLED10_KEYPOINT_H
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"


using namespace std;
using namespace cv;

struct Keypoints {
    float x;
    float y;
    float score;

    Keypoints() : x(0), y(0), score(0) {}

    Keypoints(float x, float y, float score) : x(x), y(y), score(score) {}
};

struct Box {
    float center_x;
    float center_y;
    float scale_x;
    float scale_y;
    float scale_prob;
    float score;

    Box() : center_x(0), center_y(0), scale_x(0), scale_y(0), scale_prob(0), score(0) {}

    Box(float center_x, float center_y, float scale_x, float scale_y, float scale_prob, float score) :
            center_x(center_x), center_y(center_y), scale_x(scale_x), scale_y(scale_y), scale_prob(scale_prob),
            score(score) {}
};



class Keypoint {
public:
    Keypoint();
    ~Keypoint();

private:

    void bbox_xywh2cs(float bbox[], float aspect_ratio, float padding, float pixel_std, float *center, float *scale);
    void rotate_point(float *pt, float angle_rad, float *rotated_pt);
    void _get_3rd_point(cv::Point2f a, cv::Point2f b, float *direction);
    void get_affine_transform(float *center, float *scale, float rot, float *output_size, float *shift, bool inv,
                              cv::Mat &trans);
    void
    transform_preds(std::vector <cv::Point2f> coords, std::vector <Keypoints> &target_coords, float *center, float *scale,
                    int w, int h, bool use_udp = false);

public:
    int  DetectImage(cv::Mat &rgb,float bbox[]);
    int  initModel(const char *modelPath);
private:
    bool flip_test = true;
    bool heap_map = false;
    float keypoint_score = 0.3f;

    float *imageBytes;
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
     int32_t modelWidth_;
    int32_t modelHeight_;
    float meanRgb[3] = {0, 0, 0};
    float stdRgb[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};

};


#endif //UNTITLED10_KEYPOINT_H
