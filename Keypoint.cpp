//
// Created by sxj on 2024/3/17.
//

#include "Keypoint.h"
#include "common.h"
Keypoint::Keypoint(){}
Keypoint::~Keypoint(){}

void Keypoint::bbox_xywh2cs(float bbox[], float aspect_ratio, float padding, float pixel_std, float *center, float *scale) {
    float x = bbox[0];
    float y = bbox[1];
    float w = bbox[2];
    float h = bbox[3];
    *center = x + w * 0.5;
    *(center + 1) = y + h * 0.5;
    if (w > aspect_ratio * h)
        h = w * 1.0 / aspect_ratio;
    else if (w < aspect_ratio * h)
        w = h * aspect_ratio;


    *scale = (w / pixel_std) * padding;
    *(scale + 1) = (h / pixel_std) * padding;
}

void Keypoint::rotate_point(float *pt, float angle_rad, float *rotated_pt) {
    float sn = sin(angle_rad);
    float cs = cos(angle_rad);
    float new_x = pt[0] * cs - pt[1] * sn;
    float new_y = pt[0] * sn + pt[1] * cs;
    rotated_pt[0] = new_x;
    rotated_pt[1] = new_y;

}

void Keypoint::_get_3rd_point(cv::Point2f a, cv::Point2f b, float *direction) {

    float direction_0 = a.x - b.x;
    float direction_1 = a.y - b.y;
    direction[0] = b.x - direction_1;
    direction[1] = b.y + direction_0;


}

void Keypoint::get_affine_transform(float *center, float *scale, float rot, float *output_size, float *shift, bool inv,
                          cv::Mat &trans) {
    float scale_tmp[] = {0, 0};
    scale_tmp[0] = scale[0] * 200.0;
    scale_tmp[1] = scale[1] * 200.0;
    float src_w = scale_tmp[0];
    float dst_w = output_size[0];
    float dst_h = output_size[1];
    float rot_rad = M_PI * rot / 180;
    float pt[] = {0, 0};
    pt[0] = 0;
    pt[1] = src_w * (-0.5);
    float src_dir[] = {0, 0};
    rotate_point(pt, rot_rad, src_dir);
    float dst_dir[] = {0, 0};
    dst_dir[0] = 0;
    dst_dir[1] = dst_w * (-0.5);
    cv::Point2f src[3] = {cv::Point2f(0, 0), cv::Point2f(0, 0), cv::Point2f(0, 0)};
    src[0] = cv::Point2f(center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]);
    src[1] = cv::Point2f(center[0] + src_dir[0] + scale_tmp[0] * shift[0],
                         center[1] + src_dir[1] + scale_tmp[1] * shift[1]);
    float direction_src[] = {0, 0};
    _get_3rd_point(src[0], src[1], direction_src);
    src[2] = cv::Point2f(direction_src[0], direction_src[1]);
    cv::Point2f dst[3] = {cv::Point2f(0, 0), cv::Point2f(0, 0), cv::Point2f(0, 0)};
    dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
    dst[1] = cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
    float direction_dst[] = {0, 0};
    _get_3rd_point(dst[0], dst[1], direction_dst);
    dst[2] = cv::Point2f(direction_dst[0], direction_dst[1]);

    if (inv) {
        trans = cv::getAffineTransform(dst, src);
    } else {
        trans = cv::getAffineTransform(src, dst);
    }


}


void
Keypoint::transform_preds(std::vector <cv::Point2f> coords, std::vector <Keypoints> &target_coords, float *center, float *scale,
                int w, int h, bool use_udp ) {
    float scale_x[] = {0, 0};
    float temp_scale[] = {scale[0] * 200, scale[1] * 200};
    if (use_udp) {
        scale_x[0] = temp_scale[0] / (w - 1);
        scale_x[1] = temp_scale[1] / (h - 1);
    } else {
        scale_x[0] = temp_scale[0] / w;
        scale_x[1] = temp_scale[1] / h;
    }
    for (int i = 0; i < coords.size(); i++) {
        target_coords[i].x = coords[i].x * scale_x[0] + center[0] - temp_scale[0] * 0.5;
        target_coords[i].y = coords[i].y * scale_x[1] + center[1] - temp_scale[1] * 0.5;
    }

}

int Keypoint::initModel(const char *modelPath){
//
//    AclLiteError ret = aclResource_.Init();
//    if (ret == FAILED) {
//        ACLLITE_LOG_ERROR("Keypoint resource init failed, errorCode is %d", ret);
//        return
//                FAILED;
//    }
//
//    ret = aclrtGetRunMode(&runMode_);
//    if (ret == FAILED) {
//        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
//        return
//                FAILED;
//    }
//
//// init dvpp resource
//    ret = imageProcess_.Init();
//    if (ret == FAILED) {
//        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
//        return
//                FAILED;
//    }

// load model from file
    auto ret = model_.Init(modelPath);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return
                FAILED;
    }


// data standardization

// create malloc of image, which is shape with NCHW
//const float meanRgb[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
//const float stdRgb[3] = {(1 / 0.229f / 255.f), (1 / 0.224f / 255.f), (1 / 0.225f / 255.f)};



    return 0;
}
int Keypoint::DetectImage(cv::Mat &bgr,float bbox[]) {

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB
    );

    float image_target_w = 256;
    float image_target_h = 256;
    float padding = 1.25;
    float pixel_std = 200;
    float aspect_ratio = image_target_h / image_target_w;
    //float bbox[] = {13.711652, 26.188112, 293.61298, 227.78246, 9.995332e-01};// 需要检测框架　这个矩形框来自检测框架的坐标　x y w h score
    bbox[2] = bbox[2] - bbox[0];
    bbox[3] = bbox[3] - bbox[1];
    float center[2] = {0, 0};
    float scale[2] = {0, 0};
    bbox_xywh2cs(bbox, aspect_ratio, padding, pixel_std, center, scale
    );
    float rot = 0;
    float shift[] = {0, 0};
    bool inv = false;
    float output_size[] = {image_target_h, image_target_w};
    cv::Mat trans;
    get_affine_transform(center, scale, rot, output_size, shift, inv, trans
    );
    std::cout << trans <<
              std::endl;
    std::cout << center[0] << " " << center[1] << " " << scale[0] << " " << scale[1] <<
              std::endl;
    cv::Mat detect_image;//= cv::Mat::zeros(image_target_w ,image_target_h, CV_8UC3);
    cv::warpAffine(rgb, detect_image, trans, cv::Size(image_target_h, image_target_w), cv::INTER_LINEAR
    );
//cv::imwrite("te.jpg",detect_image);
    std::cout << detect_image.cols << " " << detect_image.rows <<
              std::endl;


// inference
    bool release = false;
//SampleYOLOV7 sampleYOLO(modelPath, target_width, target_height);




    int32_t channel = detect_image.channels();
    int32_t resizeHeight = detect_image.rows;
    int32_t resizeWeight = detect_image.cols;
    imageBytes = (float *) malloc(channel * image_target_w * image_target_h * sizeof(float));
    memset(imageBytes,
           0,
           channel * image_target_h
           * image_target_w * sizeof(float));

// image to bytes with shape HWC to CHW, and switch channel BGR to RGB

    for (
            int c = 0;
            c < channel;
            ++c) {
        for (
                int h = 0;
                h < resizeHeight;
                ++h) {
            for (
                    int w = 0;
                    w < resizeWeight;
                    ++w) {
                int dstIdx = c * resizeHeight * resizeWeight + h * resizeWeight + w;

                imageBytes[dstIdx] = static_cast
                        <float>(
                        (detect_image
                                 .
                                         at<cv::Vec3b>(h, w
                                 )[c] -
                         1.0f * meanRgb[c]) * 1.0f * stdRgb[c] );
            }
        }
    }


    std::vector <InferenceOutput> inferOutputs;
    int ret = model_.CreateInput(static_cast<void *>(imageBytes),
                             channel * image_target_w * image_target_h * sizeof(float));
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return
                FAILED;
    }

// inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return
                FAILED;
    }

// for()
    float *data = static_cast<float *>(inferOutputs[0].data.get());
//输出维度
    int shape_d = 1;
    int shape_c = 20;
    int shape_w = 64;
    int shape_h = 64;
    std::vector<float> vec_heap;
    for (
            int i = 0;
            i < shape_c * shape_h * shape_w;
            i++) {
        vec_heap.
                push_back(data[i]);
    }


    std::vector <Keypoints> all_preds;
    std::vector<int> idx;
    for (
            int i = 0;
            i < shape_c;
            i++) {
        auto begin = vec_heap.begin() + i * shape_w * shape_h;
        auto end = vec_heap.begin() + (i + 1) * shape_w * shape_h;
        float maxValue = *max_element(begin, end);
        int maxPosition = max_element(begin, end) - begin;
        all_preds.
                emplace_back(Keypoints(0, 0, maxValue)
        );
        idx.
                emplace_back(maxPosition);
    }
    std::vector <cv::Point2f> vec_point;
    for (
            int i = 0;
            i < idx.

                    size();

            i++) {
        int x = idx[i] % shape_w;
        int y = idx[i] / shape_w;
        vec_point.
                emplace_back(cv::Point2f(x, y)
        );
    }


    for (
            int i = 0;
            i < shape_c;
            i++) {
        int px = vec_point[i].x;
        int py = vec_point[i].y;
        if (px > 1 && px < shape_w - 1 && py > 1 && py < shape_h - 1) {
            float diff_0 = vec_heap[py * shape_w + px + 1] - vec_heap[py * shape_w + px - 1];
            float diff_1 = vec_heap[(py + 1) * shape_w + px] - vec_heap[(py - 1) * shape_w + px];
            vec_point[i].x += diff_0 == 0 ? 0 : (diff_0 > 0) ? 0.25 : -0.25;
            vec_point[i].y += diff_1 == 0 ? 0 : (diff_1 > 0) ? 0.25 : -0.25;
        }
    }
    std::vector <Box> all_boxes;
    if (heap_map) {
        all_boxes.
                emplace_back(Box(center[0], center[1], scale[0], scale[1], scale[0] * scale[1] * 400, bbox[4])
        );
    }
    transform_preds(vec_point, all_preds, center, scale, shape_w, shape_h
    );
//0 L_Eye  1 R_Eye 2 L_EarBase 3 R_EarBase 4 Nose 5 Throat 6 TailBase 7 Withers 8 L_F_Elbow 9 R_F_Elbow 10 L_B_Elbow 11 R_B_Elbow
// 12 L_F_Knee 13 R_F_Knee 14 L_B_Knee 15 R_B_Knee 16 L_F_Paw 17 R_F_Paw 18 L_B_Paw 19  R_B_Paw

    int skeleton[][2] = {{0,  1},
                         {0,  2},
                         {1,  3},
                         {0,  4},
                         {1,  4},
                         {4,  5},
                         {5,  7},
                         {5,  8},
                         {5,  9},
                         {6,  7},
                         {6,  10},
                         {6,  11},
                         {8,  12},
                         {9,  13},
                         {10, 14},
                         {11, 15},
                         {12, 16},
                         {13, 17},
                         {14, 18},
                         {15, 19}};

    cv::rectangle(bgr, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  cv::Scalar(255, 0, 0)
    );
    for (
            int i = 0;
            i < all_preds.

                    size();

            i++) {
        if (all_preds[i].score > keypoint_score) {
            cv::circle(bgr, cv::Point(all_preds[i].x, all_preds[i].y),
                       3, cv::Scalar(0, 255, 120), -1);//画点，其实就是实心圆
        }
    }
    for (
            int i = 0;
            i < sizeof(skeleton) / sizeof(sizeof(skeleton[1])); i++) {
        int x0 = all_preds[skeleton[i][0]].x;
        int y0 = all_preds[skeleton[i][0]].y;
        int x1 = all_preds[skeleton[i][1]].x;
        int y1 = all_preds[skeleton[i][1]].y;

        cv::line(bgr, cv::Point(x0, y0), cv::Point(x1, y1),
                 cv::Scalar(0, 255, 0),
                 1);

    }

  return 0;
}
