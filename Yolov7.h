//
// Created by sxj on 2024/3/17.
//
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "iostream"
#include "Keypoint.h"

using namespace std;
using namespace cv;

#ifndef UNTITLED10_YOLOV7_H
#define UNTITLED10_YOLOV7_H
#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 200
#define OBJ_CLASS_NUM     80
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)
using namespace std;

typedef struct _BOX_RECT {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
    char name[OBJ_NAME_MAX_SIZE];
    int class_index;
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t {
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;





class Yolov7 {
public:
    Yolov7();
    ~Yolov7();

public:
    int InitModel(const char *modelPath,const char *modelPath_1);
   int DetectImage();

private:
    float sigmoid(float x);
    float unsigmoid(float y);
    int process_fp(float *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
                   float threshold);
    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                           float ymax1);
    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> &order, float threshold);
    int quick_sort_indice_inverse(
            std::vector<float> &input,
            int left,
            int right,
            std::vector<int> &indices);
    int clamp(float val, int min, int max);
    int post_process_fp(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,
                        int h_offset, int w_offset, float resize_scale, float conf_threshold, float nms_threshold,
                        detect_result_group_t *group, const char *labels[]);
    void letterbox(cv::Mat rgb,cv::Mat &img_resize,int target_width,int target_height);
private:
    const int anchor0[6] = {12,16, 19,36, 40,28};
    const int anchor1[6] = {36,75, 76,55, 72,146};
    const int anchor2[6] = {142,110, 192,243, 459,401};
    float resize_scale = 0;
    int h_pad=0;
    int w_pad=0;
    const float nms_threshold = 0.6;
    const float conf_threshold = 0.25;
    // data standardization
    float meanRgb[3] = {0, 0, 0};
    float stdRgb[3]  = {1/255.0f, 1/255.0f, 1/255.0f};
    Keypoint *item;
    const int32_t target_width = 640;
    const int32_t target_height = 640;
     const char* image_process_mode="letter_box";
    const char *labels[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                            "traffic light",
                            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                            "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                            "suitcase", "frisbee",
                            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                            "skateboard", "surfboard",
                            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                            "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                            "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                            "scissors", "teddy bear",
                            "hair drier", "toothbrush"};
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    int32_t modelWidth_;
    int32_t modelHeight_;

};


#endif //UNTITLED10_YOLOV7_H
