

#include "Cam.h"
#include "Yolov7.h"
#include "thread"
void SendFrameThread() {
    Cam *item_cam = new Cam();

    int ret = item_cam->init_paramter();
    if (ret != 0) {
        printf("initial camera fail \n");
        return;
    }

    while (true) {

        int ret = item_cam->detect_image();
        if (ret != 0) {
            printf("get frame error \n");
        }
    }

    item_cam->close();

    delete item_cam;
}

void InterfaceFrameThread(const char *modelPath,const char *modelPath_1 ) {


    Yolov7 *item=new Yolov7();
    item->InitModel(modelPath,modelPath_1);
    while (1){
        item->DetectImage();
    }
}

int main() {

    const char *modelPath = "../model/yolov7_batch_size_1.om";
    const char *modelPath_1 = "../model/end2end.om";
    const int thread_num = 2;
    std::array<thread, thread_num> threads;
    threads = {
            thread(SendFrameThread),
            thread(InterfaceFrameThread,modelPath,modelPath_1),

    };

    for (int i = 0; i < thread_num; i++) {
        threads[i].join();
    }


    return SUCCESS;
}