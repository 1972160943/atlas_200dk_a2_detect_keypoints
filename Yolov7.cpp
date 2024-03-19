//
// Created by sxj on 2024/3/17.
//

#include "Yolov7.h"
#include "common.h"
#include "Keypoint.h"
extern Queue<cv::Mat> iq0;  //实例化iq
extern Queue<cv::Mat> iq1;  //实例化iq
extern std::mutex mx;  //全局锁mx
extern std::condition_variable cv_;  //条件变量cv

Yolov7::Yolov7(){
     item=new Keypoint();
}
Yolov7::~Yolov7(){
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int Yolov7::InitModel(const char *modelPath,const char *modelPath_1){


    AclLiteError    ret = aclResource_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }


      ret=item->initModel(modelPath_1);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("keypoint resource init failed, errorCode is %d", ret);
        return FAILED;
    }else{
        printf("keypoint resource  successfully %d", ret);
     }


    // load model from file
    ret = model_.Init(modelPath);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }

    return 0;
}


float Yolov7::sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

float Yolov7::unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

int Yolov7::process_fp(float *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
               std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
               float threshold) {

    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres_sigmoid = unsigmoid(threshold);
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_sigmoid) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float *in_ptr = input + offset;
                    float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
                    float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
                    float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
                    float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;
                    box_x = (box_x + j) * (float) stride;
                    box_y = (box_y + i) * (float) stride;
                    box_w = box_w * box_w * (float) anchor[a * 2];
                    box_h = box_h * box_h * (float) anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    float box_conf_f32 = sigmoid(box_confidence);
                    float class_prob_f32 = sigmoid(maxClassProbs);
                    boxScores.push_back(box_conf_f32 * class_prob_f32);
                    classId.push_back(maxClassId);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

float Yolov7::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                       float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int Yolov7::nms(int validCount, std::vector<float> &outputLocations, std::vector<int> &order, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int Yolov7::quick_sort_indice_inverse(
        std::vector<float> &input,
        int left,
        int right,
        std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

int Yolov7::clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

int Yolov7::post_process_fp(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,
                    int h_offset, int w_offset, float resize_scale, float conf_threshold, float nms_threshold,
                    detect_result_group_t *group, const char *labels[]) {
    memset(group, 0, sizeof(detect_result_group_t));
    std::vector<float> filterBoxes;
    std::vector<float> boxesScore;
    std::vector<int> classId;
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = process_fp(input0, (int *) anchor0, grid_h0, grid_w0, model_in_h, model_in_w,
                             stride0, filterBoxes, boxesScore, classId, conf_threshold);

    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process_fp(input1, (int *) anchor1, grid_h1, grid_w1, model_in_h, model_in_w,
                             stride1, filterBoxes, boxesScore, classId, conf_threshold);

    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process_fp(input2, (int *) anchor2, grid_h2, grid_w2, model_in_h, model_in_w,
                             stride2, filterBoxes, boxesScore, classId, conf_threshold);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(boxesScore, 0, validCount - 1, indexArray);

    nms(validCount, filterBoxes, indexArray, nms_threshold);

    int last_count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {

        if (indexArray[i] == -1 || boxesScore[i] < conf_threshold || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];

        group->results[last_count].box.left = (int) ((clamp(x1, 0, model_in_w) - w_offset) / resize_scale);
        group->results[last_count].box.top = (int) ((clamp(y1, 0, model_in_h) - h_offset) / resize_scale);
        group->results[last_count].box.right = (int) ((clamp(x2, 0, model_in_w) - w_offset) / resize_scale);
        group->results[last_count].box.bottom = (int) ((clamp(y2, 0, model_in_h) - h_offset) / resize_scale);
        group->results[last_count].prop = boxesScore[i];
        group->results[last_count].class_index = id;
        const char *label = labels[id];
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

        // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left, group->results[last_count].box.top,
        //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
        last_count++;
    }
    group->count = last_count;

    return 0;
}


void Yolov7::letterbox(cv::Mat rgb,cv::Mat &img_resize,int target_width,int target_height){

    float shape_0=rgb.rows;
    float shape_1=rgb.cols;
    float new_shape_0=target_height;
    float new_shape_1=target_width;
    float r=std::min(new_shape_0/shape_0,new_shape_1/shape_1);
    float new_unpad_0=int(round(shape_1*r));
    float new_unpad_1=int(round(shape_0*r));
    float dw=new_shape_1-new_unpad_0;
    float dh=new_shape_0-new_unpad_1;
    dw=dw/2;
    dh=dh/2;
    cv::Mat copy_rgb=rgb.clone();
    if(int(shape_0)!=int(new_unpad_0)&&int(shape_1)!=int(new_unpad_1)){
        cv::resize(copy_rgb,img_resize,cv::Size(new_unpad_0,new_unpad_1));
        copy_rgb=img_resize;
    }
    int top=int(round(dh-0.1));
    int bottom=int(round(dh+0.1));
    int left=int(round(dw-0.1));
    int right=int(round(dw+0.1));
    cv::copyMakeBorder(copy_rgb, img_resize,top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

}
int Yolov7::DetectImage(){

    cv::Mat bgr;

    {
        std::unique_lock<std::mutex> lock(mx);
        cv_.wait(lock, []() -> bool { return !iq0.Empty(); });//lambda表达式中为真退出等待不为NULL时，退出wait

        iq0.Front(bgr);

        cv_.notify_all();
    }


    ///
    if (!bgr.data) {
         return -1;
    }
    cv::Mat rgb;
    //BGR->RGB
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_resize;
    float correction[2] = {0, 0};
    float scale_factor[] = {0, 0};
    int width=rgb.cols;
    int height=rgb.rows;
    // Letter box resize
    float img_wh_ratio = (float) width / height;
    float input_wh_ratio = (float) target_width /  target_height;
    int resize_width;
    int resize_height;
    if (img_wh_ratio >= input_wh_ratio) {
        //pad height dim
        resize_scale = (float) target_width /  width;
        resize_width = target_width;
        resize_height =(float) height * resize_scale;
        w_pad = 0;
        h_pad = (target_height - resize_height) / 2;
    } else {
        //pad width dim
        resize_scale = (float) target_height / height;
        resize_width = (float) width * resize_scale;
        resize_height = target_height;
        w_pad = (target_width - resize_width) / 2;;
        h_pad = 0;
    }
    if(strcmp(image_process_mode,"letter_box")==0){
        letterbox(rgb,img_resize,target_width,target_height);
    }else {
        cv::resize(rgb, img_resize, cv::Size(target_width, target_height));
    }
    int32_t channel = img_resize.channels();
    int32_t resizeHeight = img_resize.rows;
    int32_t resizeWeight = img_resize.cols;

    // data standardization
    float meanRgb[3] = {0, 0, 0};
    float stdRgb[3]  = {1/255.0f, 1/255.0f, 1/255.0f};

    // create malloc of image, which is shape with NCHW
    auto imageBytes = (float*)malloc(channel * resizeHeight * resizeWeight * sizeof(float));
    memset(imageBytes, 0, channel * resizeHeight * resizeWeight * sizeof(float));

    uint8_t bgrToRgb=2;
    // image to bytes with shape HWC to CHW, and switch channel BGR to RGB
    for (int c = 0; c < channel; ++c)
    {
        for (int h = 0; h < resizeHeight; ++h)
        {
            for (int w = 0; w < resizeWeight; ++w)
            {
                int dstIdx = (bgrToRgb - c) * resizeHeight * resizeWeight + h * resizeWeight + w;
                imageBytes[dstIdx] =  static_cast<float>((img_resize.at<cv::Vec3b>(h, w)[c] -
                                                          1.0f*meanRgb[c]) * 1.0f*stdRgb[c] );
            }
        }
    }

    std::vector <InferenceOutput> inferOutputs;
    AclLiteError ret = model_.CreateInput(static_cast<void *>(imageBytes), channel * resizeHeight * resizeWeight * sizeof(float));
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }



    detect_result_group_t detect_result_group;

    post_process_fp(static_cast<float *>(inferOutputs[0].data.get()), static_cast<float *>(inferOutputs[1].data.get()), static_cast<float *>(inferOutputs[2].data.get()), target_width,
                    target_height,
                    h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, &detect_result_group, labels);

    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        if(strcmp(det_result->name,"person")==0){
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int bx1 = det_result->box.left;
        int by1 = det_result->box.top;
        int bx2 = det_result->box.right;
        int by2 = det_result->box.bottom;
        cv::rectangle(bgr, cv::Point(bx1, by1), cv::Point(bx2, by2), cv::Scalar(231, 232, 143));  //两点的方式
        char text[256];
        sprintf(text, "%s %.1f%% ", det_result->name, det_result->prop * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = bx1;
        int y = by1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;


        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(0, 0, 255), -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        float bbox[]={(float)bx1,(float)by1,(float)bx2-bx1,(float)by2-by1,det_result->prop};
        item->DetectImage(bgr,bbox);
       cv::imwrite("../csxj731533730_yolov7_batchsize_1.jpg", bgr);
        }
    }

    return 0;
}