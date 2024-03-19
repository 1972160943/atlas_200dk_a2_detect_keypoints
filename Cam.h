//
// Created by ubuntu on 2023/4/30.
//

#ifndef RK3588_MACHINE_CAM_H
#define RK3588_MACHINE_CAM_H

#include <iostream>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include<sys/types.h>
#include<fcntl.h>
#include<stdlib.h>
#include <zconf.h>
#include<linux/videodev2.h>
#include<sys/mman.h>
#include <jpeglib.h>
#include <cstring>
#include <linux/fb.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include "string"
#include "stdio.h"
#include "common.h"

using namespace std;

class Cam {
public:
    Cam();
    ~Cam();
public:
    int init_paramter();
    int close();
    int detect_image();

private:
    unsigned char rgb_data[WIDTH * HEIGHT * 3];
    int fd=-1;
    unsigned char *mptr[4];
    unsigned int mptr_len[4];
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

private:
    int read_JPEG_file(unsigned char *jpegData, unsigned char *rgbdata, int size);
};


#endif //RK3588_MACHINE_CAM_H
