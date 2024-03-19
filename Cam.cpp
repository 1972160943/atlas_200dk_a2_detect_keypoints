//
// Created by ubuntu on 2023/4/30.
//


#include "Cam.h"
#include "common.h"
Queue<cv::Mat> iq0;  //实例化iq
Queue<cv::Mat> iq1;  //实例化iq
std::mutex mx;  //全局锁mx
std::condition_variable cv_;  //条件变量cv

Cam::Cam() {}

Cam::~Cam() {

}

int Cam::read_JPEG_file(unsigned char *jpegData, unsigned char *rgbdata, int size) {
    struct jpeg_error_mgr jerr;
    struct jpeg_decompress_struct cinfo;
    cinfo.err = jpeg_std_error(&jerr);
    //1创建解码对象并且初始化
    jpeg_create_decompress(&cinfo);
    //2.装备解码的数据
    //jpeg_stdio_src(&cinfo, infile);
    jpeg_mem_src(&cinfo, (unsigned char *) jpegData, size);
    //3.获取jpeg图片文件的参数
    (void) jpeg_read_header(&cinfo, TRUE);
    /* Step 4: set parameters for decompression */
    //5.开始解码
    (void) jpeg_start_decompress(&cinfo);
    //6.申请存储一行数据的内存空间
    int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *buffer = (unsigned char *) (malloc(row_stride));
    int i = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        //printf("****%d\n",i);
        (void) jpeg_read_scanlines(&cinfo, &buffer, 1);
        memcpy(rgbdata + i * WIDTH * 3, buffer, row_stride);
        i++;
    }
    //7.解码完成
    (void) jpeg_finish_decompress(&cinfo);
    //8.释放解码对象
    jpeg_destroy_decompress(&cinfo);
    free(buffer);
    return 1;
}

int Cam::init_paramter() {
    //打开设备
    fd = open("/dev/video0", O_RDWR);
    if (fd < 0) {
        perror("open device fail\n");
        return -1;
    }
    // 获取摄像头支持的格式
    //获取摄像头支持的格式
    struct v4l2_fmtdesc v4fmt;
    v4fmt.index = 0;
    v4fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    int ret = ioctl(fd, VIDIOC_ENUM_FMT, &v4fmt);
    if (ret < 0) {
        perror("acquire fail\n");
        return -1;
    }
    printf("%s\n", v4fmt.description);
    unsigned char *p = (unsigned char *) &v4fmt.pixelformat;
    printf("%c %c %c %c\n", p[0], p[1], p[2], p[3]);
    // 设置摄像头支持的格式
    //设置摄像头支持的格式
    struct v4l2_format vFormat;
    vFormat.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vFormat.fmt.pix.width = WIDTH;
    vFormat.fmt.pix.height = HEIGHT;
    vFormat.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    ret = ioctl(fd, VIDIOC_S_FMT, &vFormat);
    if (ret < 0) {
        perror("set fail\n");
        return -1;
    }
    //申请内核空间
    struct v4l2_requestbuffers vqbuff;
    vqbuff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vqbuff.count = 4;
    vqbuff.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(fd, VIDIOC_REQBUFS, &vqbuff);
    if (ret < 0) {
        perror("buff fail\n");
        return -1;
    }
    //申请内存空间
    struct v4l2_buffer mptrbuff;
    mptrbuff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;



    for (int i = 0; i < 4; i++) {
        mptrbuff.index = i;
        ret = ioctl(fd, VIDIOC_QUERYBUF, &mptrbuff);
        if (ret < 0) {
            perror("require buff fail\n");
            return -1;
        }
        mptr[i] = (unsigned char *) mmap(NULL, mptrbuff.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                                         mptrbuff.m.offset);
        mptr_len[i] = mptrbuff.length;
        //通知完毕
        ret = ioctl(fd, VIDIOC_QBUF, &mptrbuff);
        if (ret < 0) {
            perror("put fail");
            return -1;
        }

    }
    // 开始采集

    ret = ioctl(fd, VIDIOC_STREAMON, &type);
    if (ret < 0) {
        perror("open fail");
        return -1;
    }

    return 0;
}




//int yy=0;
int Cam::detect_image() {

    struct v4l2_buffer readbuff;
    readbuff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int ret = ioctl(fd, VIDIOC_DQBUF, &readbuff);
    if (ret < 0) {
        perror("read fail");
    }
    read_JPEG_file(mptr[readbuff.index], rgb_data, readbuff.length);
    cv::Mat bgrImg(HEIGHT,WIDTH,CV_8UC3,rgb_data);
    cv::cvtColor(bgrImg, bgrImg, cv::COLOR_BGR2RGB);
    //printf("----opencv show image----\n");

     //bgrImg=cv::imread("../fire000000.jpg");
    //break;
    //通知内核 已经使用完
    ret = ioctl(fd, VIDIOC_QBUF, &readbuff);
    if (ret < 0) {
        perror("put equee fail\n");

    }




    std::unique_lock<std::mutex> lock(mx);  //类似于智能指针的智能锁
    cv_.wait(lock, []()->bool {return !iq0.Full(); });  //lambda表达式
    iq0.Push(bgrImg);  //上述lambda表达式为真退出，所以就不为full时为退出

    cv_.notify_all();


    return 0;
}

int Cam::close() {
    //停止采集
    int ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
    if (ret < 0) {
        perror("stop eque@e fail\n");

    }

    //释放资源、
    for (int i = 0; i < 4; i++) {
        munmap(mptr[i], mptr_len[i]);
    }
    //关闭设备
   // close(fd);
    printf("close device successfully\n");


    return 0;
}