/*
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>

#include "NvJpegDecoder.h"
#include "NvEglRenderer.h"
#include "NvUtils.h"
#include "NvCudaProc.h"
#include "nvbuf_utils.h"

#include "camera_v4l2_mjpg.h"

#ifdef ENABLE_VALGRIND
#include <valgrind/memcheck.h>
#endif

static bool quit = false;

using namespace std;

static void
print_usage(void)
{
    printf("\n\tUsage: camera_v4l2_cuda [OPTIONS]\n\n"
           "\tExample: \n"
           "\t./camera_v4l2_cuda -d /dev/video0 -s 640x480 -f MJPG -n 30 -c\n\n"
           "\tSupported options:\n"
           "\t-d\t\tSet V4l2 video device node\n"
           "\t-s\t\tSet output resolution of video device\n"
           "\t-f\t\tSet output pixel format of video device (supports only MJPG)\n"
           "\t-r\t\tSet renderer frame rate (30 fps by default)\n"
           "\t-n\t\tSave the n-th frame before VIC processing\n"
           "\t-c\t\tEnable CUDA aglorithm (draw a black box in the upper left corner)\n"
           "\t-m\t\tEnable level : 0 for silente , 1 for parser, 2 for decoder and 3 for display \n"
           "\t-l\t\tEnable valgrind check ( memory leak )\n"
           "\t-v\t\tEnable verbose message\n"
           "\t-h\t\tPrint this usage\n\n"
           "\tNOTE: It runs infinitely until you terminate it with <ctrl+c>\n");
}

static bool
parse_cmdline(context_t * ctx, int argc, char **argv)
{
    int c;

    if (argc < 2)
    {
        print_usage();
        exit(EXIT_SUCCESS);
    }

                    
    while ((c = getopt(argc, argv, "d:s:f:r:n:m:l:cvh")) != -1)
    {
        switch (c)
        {
            case 'd':
                ctx->cam_devname = optarg;
                break;
            case 's':
                if (sscanf(optarg, "%dx%d",
                            &ctx->cam_w, &ctx->cam_h) != 2)
                {
                    print_usage();
                    return false;
                }
                break;
            case 'f':
                if (strcmp(optarg, "MJPG") == 0)
                    ctx->cam_pixfmt = V4L2_PIX_FMT_MJPEG;
                else 
                {
                    print_usage();
//                     return false;
                }
                sprintf(ctx->cam_file, "camera.%s", optarg);
                break;
            case 'r':
                ctx->fps = strtol(optarg, NULL, 10);
                break;
            case 'n':
                ctx->save_n_frame = strtol(optarg, NULL, 10);
                break;
            case 'c':
                ctx->enable_cuda = true;
                break;
            case 'm':
                ctx->enable_silence = strtol(optarg, NULL, 10);
                break;
            case 'l':
                ctx->enable_valgrind = true;
                break;
            case 'v':
                ctx->enable_verbose = true;
                break;
            case 'h':
                print_usage();
                exit(EXIT_SUCCESS);
                break;
            default:
                print_usage();
                return false;
        }
    }

    return true;
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->cam_devname = "/dev/video0";
    ctx->cam_fd = -1;
    ctx->cam_pixfmt = V4L2_PIX_FMT_MJPEG ; // V4L2_PIX_FMT_YUYV;
    ctx->cam_w = 640;
    ctx->cam_h = 480;
    ctx->frame = 0;
    ctx->save_n_frame = 0;

    ctx->g_buff = NULL;
    ctx->renderer = NULL;
    ctx->fps = 30;
    ctx->enable_silence = 3;
    ctx->enable_valgrind = false;
    ctx->enable_cuda = false;
    ctx->egl_image = NULL;
    ctx->egl_display = EGL_NO_DISPLAY;

    ctx->enable_verbose = false;
}

static nv_color_fmt nvcolor_fmt[] =
{
    // TODO add more pixel format mapping
    {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
    {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
    {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
    {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
    {V4L2_PIX_FMT_YUV420M, NvBufferColorFormat_YUV420},
};

static NvBufferColorFormat
get_nvbuff_color_fmt(unsigned int v4l2_pixfmt)
{
    unsigned i;

    for (i = 0; i < sizeof(nvcolor_fmt) / sizeof(nvcolor_fmt[0]); i++)
    {
        if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
            return nvcolor_fmt[i].nvbuff_color;
    }

    return NvBufferColorFormat_Invalid;
}

static bool
save_frame_to_file(context_t * ctx, struct v4l2_buffer * buf)
{
    int file;

    file = open(ctx->cam_file, O_CREAT | O_WRONLY | O_APPEND | O_TRUNC,
            S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);

    if (-1 == file)
        ERROR_RETURN("Failed to open file for frame saving");

    if (-1 == write(file, ctx->g_buff[buf->index].start,
                ctx->g_buff[buf->index].size))
    {
        close(file);
        ERROR_RETURN("Failed to write frame into file");
    }

    close(file);

    return true;
}

static bool
camera_initialize(context_t * ctx)
{
    struct v4l2_format fmt;

    // Open camera device
    ctx->cam_fd = open(ctx->cam_devname, O_RDWR);
    if (ctx->cam_fd == -1)
        ERROR_RETURN("Failed to open camera device %s: %s (%d)",
                ctx->cam_devname, strerror(errno), errno);

    // Set camera output format
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
//     fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to set camera output format: %s (%d)",
                strerror(errno), errno);

    // Get the real format in case the desired is not supported
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to get camera output format: %s (%d)",
                strerror(errno), errno);
    if (fmt.fmt.pix.width != ctx->cam_w ||
            fmt.fmt.pix.height != ctx->cam_h ||
            fmt.fmt.pix.pixelformat != ctx->cam_pixfmt)
    {
        WARN("The desired format is not supported");
        ctx->cam_w = fmt.fmt.pix.width;
        ctx->cam_h = fmt.fmt.pix.height;
        ctx->cam_pixfmt =fmt.fmt.pix.pixelformat;
    }

    INFO("Camera ouput format: (%d x %d)  stride: %d, imagesize: %d\n",
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fmt.fmt.pix.bytesperline,
            fmt.fmt.pix.sizeimage);

    return true;
}



static bool CaptureInitialize( context_t * ctx  ) {
    
    
        ctx->m_JpegDec = NvJPEGDecoder::createJPEGDecoder("jpegdec_0");
        
        
        if( ctx->m_JpegDec == nullptr ) {
            ERROR_RETURN("Could not create Jpeg Decoder");
        }
    
        ERROR_RETURN("Create Jpeg Decoder jpegdec_0 done");
            
    return ( ctx->m_JpegDec != nullptr );
    
}




static bool
display_initialize(context_t * ctx)
{
    // Create EGL renderer

    
    bool lret = true;
    

    uint32_t display_width = 0 , display_height = 0 , x_offset = 0 , y_offset = 0; // x_fpsoffset, y_fpsoffset;
    
    NvEglRenderer::getDisplayResolution(display_width, display_height);
    
    
    	// Crop and resizing the streaming window based on display size
	x_offset = 0; //(display_width / 16) * 3;
	y_offset = 0;
	ctx->x_Display_Offset = display_width; //10;
	ctx->y_Display_Offset = display_height; // - 10;
    
    
    double dcoef = (double) ( ((double)display_width) / ((double)display_height) );
    double ccoef = (double) ( ((double)ctx->cam_w) / ((double)ctx->cam_h) );
    
    uint32_t x = 0;
    uint32_t y = 0;
    if( dcoef < ccoef   )  {
        x = display_width ;
        y = x/ ccoef;
    }
    else{
        y = display_height ;
        x = y * ccoef;
    }
    

    x_offset = (display_width - x ) /2.0 ;
    y_offset = (display_height - y ) /2.0 ;
    

	ctx->x_Display_Offset = x; //10;
	ctx->y_Display_Offset = y; // - 10;
    ctx->renderer = nullptr;
//         ERROR_RETURN("Create Egl Renderer on display"); 
    ctx->renderer = NvEglRenderer::createEglRenderer("eglrenderer_0", x, y, x_offset, y_offset);
    
    if( ctx->renderer != nullptr ) {    
        
        // TX2 : Enable render profiling information
        ctx->renderer->enableProfiling();
        
        printf("Create NvEglRenderer::createEglRenderer : eglrenderer_0 done" );
    }
    else {
        ERROR_RETURN("Could not create renderer");        
    }
	
            
//     return lret && ( ctx->renderer != nullptr );
    
    if (!ctx->renderer) {
        ERROR_RETURN("Failed to create EGL renderer");
    }
    
    ctx->renderer->setFPS(ctx->fps);

    if (ctx->enable_cuda)
    {
        // Get defalut EGL display
        ctx->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (ctx->egl_display == EGL_NO_DISPLAY)
            ERROR_RETURN("Failed to get EGL display connection");

        // Init EGL display connection
        if (!eglInitialize(ctx->egl_display, NULL, NULL))
            ERROR_RETURN("Failed to initialize EGL display connection");
    }

    return true;
}

static bool
init_components(context_t * ctx)
{
    if (!camera_initialize(ctx))
        ERROR_RETURN("Failed to initialize camera device");

    if (!display_initialize(ctx)) {
//         ERROR_RETURN("Failed to initialize display");
         printf("Failed to initialize display\n");
    }
    INFO("Initialize v4l2 components successfully");
    return true;
}

static bool
request_camera_buff(context_t *ctx)
{
    
    bool lret = true;
    // Request camera v4l2 buffer
	/* Buffer allocation
	 * Buffer can be allocated either from capture driver or
	 * user pointer can be used
	 */
	/* Request for MAX_BUFFER input buffers. As far as Physically contiguous
	 * memory is available, driver can allocate as many buffers as
	 * possible. If memory is not available, it returns number of
	 * buffers it has allocated in count member of reqbuf.
	 * HERE count = number of buffer to be allocated.
	 * type = type of device for which buffers are to be allocated.
	 * memory = type of the buffers requested i.e. driver allocated or
	 * user pointer */
    
	memset(&ctx->cam_req, 0, sizeof (ctx->cam_req));
    ctx->cam_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    uint32_t counterbuffer = UVC_BUFFER_NUM ;
	ctx->cam_req.count = counterbuffer;
	ctx->cam_req.type = ctx->cam_type;
	ctx->cam_req.memory = V4L2_MEMORY_MMAP;
    
	if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &ctx->cam_req) < 0) {
		ERROR_RETURN("CAPTURE: VIDIOC_REQBUFS");
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)", strerror(errno), errno);
		lret = false;
	}
	

	
    if (ctx->cam_req.count != counterbuffer) {
        ERROR_RETURN("V4l2 buffer number is %i not as desired ( %i )" , (uint32_t)ctx->cam_req.count, (uint32_t)counterbuffer);
        lret = false;
    }
	
    printf("V4l2 buffer number is %i ( %i )\n" , (uint32_t)ctx->cam_req.count, (uint32_t)counterbuffer);

	/* Mmap the buffers
	 * To access driver allocated buffer in application space, they have
	 * to be mmapped in the application space using mmap system call */
	for (uint32_t i = 0; i < ctx->cam_req.count; i++)	{
        
        if( lret ) {
            memset(&ctx->cam_buf,0, sizeof(ctx->cam_buf));
            ctx->cam_buf.type = ctx->cam_type;
            ctx->cam_buf.index = i;
            ctx->cam_buf.memory = V4L2_MEMORY_MMAP;
            int  err_chk;
            if((err_chk=ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &ctx->cam_buf)) < 0) {
                ERROR_RETURN("CAPTURE: VIDIOC_QUERYBUF");
                ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                            strerror(errno), errno);
                lret = false;
            }
            else {
                
                printf("CAPTURE: Prepare MMAP buffer\n");
                
                ctx->buffers[i].cam_offset = (size_t) ctx->cam_buf.m.offset;
                ctx->buffers[i].cam_length = ctx->cam_buf.length;
                ctx->buffers[i].cam_start = (unsigned char *)mmap(NULL, ctx->cam_buf.length,
                                PROT_READ | PROT_WRITE, MAP_SHARED, ctx->cam_fd,
                                ctx->buffers[i].cam_offset);
                
                if (ctx->buffers[i].cam_start == MAP_FAILED) {
                    ERROR_RETURN("Cannot mmap = %i buffer\n", i);
                    lret = false;
                }
                else {
                    /* Enqueue buffers
                    * Before starting streaming, all the buffers needs to be
                    * en-queued in the driver incoming queue. These buffers will
                    * be used by thedrive for storing captured frames. */
                    if(ioctl(ctx->cam_fd, VIDIOC_QBUF, &ctx->cam_buf) < 0) {
                        ERROR_RETURN("CAPTURE: VIDIOC_QBUF :" );
                        ERROR_RETURN("- Failed to request v4l2 buffers: %s (%d)",strerror(errno), errno);
                        lret = false;
                    }
                }
            }
        }
        
        printf("V4l2 buffer %i %i is %s\n" ,(uint32_t) i ,(uint32_t) ctx->cam_req.count, (lret? "OK":"NOK"));
	}
	
	return lret;

}


/*
 * 
 * @brief DMABUF and YUV note use here 
 * 
 */
static bool
prepare_buffers(context_t * ctx)
{
    
    
    INFO("Succeed in preparing stream buffers\n");
    NvBufferCreateParams input_params = {0};

    // Allocate global buffer context
    ctx->g_buff = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    if (ctx->g_buff == NULL)
        ERROR_RETURN("Failed to allocate global buffer context");

    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = ctx->cam_w;
    input_params.height = ctx->cam_h;
    input_params.layout = NvBufferLayout_Pitch;

    // Create buffer and provide it with camera
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        int fd;
        NvBufferParams params = {0};

        input_params.colorFormat = NvBufferColorFormat_YVYU ; // get_nvbuff_color_fmt(ctx->cam_pixfmt);
        input_params.nvbuf_tag = NvBufferTag_CAMERA;
        if (-1 == NvBufferCreateEx(&fd, &input_params))
            ERROR_RETURN("Failed to create NvBuffer");

        ctx->g_buff[index].dmabuff_fd = fd;

        if (-1 == NvBufferGetParams(fd, &params))
            ERROR_RETURN("Failed to get NvBuffer parameters");

        // TODO add multi-planar support
        // Currently it supports only YUV422 interlaced single-planar
        if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                    (void**)&ctx->g_buff[index].start))
            ERROR_RETURN("Failed to map buffer");

    }

    input_params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);
    input_params.nvbuf_tag = NvBufferTag_NONE;
    // Create Render buffer
    if (-1 == NvBufferCreateEx(&ctx->render_dmabuf_fd, &input_params))
        ERROR_RETURN("Failed to create NvBuffer");

    if (!request_camera_buff(ctx))
        ERROR_RETURN("Failed to set up camera buff");

    INFO("Succeed in preparing stream buffers\n");
    return true;
}

static bool
start_stream(context_t * ctx)
{
    usleep(200);
    enum v4l2_buf_type type;

    // Start v4l2 streaming
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0) {
        ERROR_RETURN("Failed to start streaming: %s (%d)", strerror(errno), errno);
//         printf("Failed to start streaming: %s (%d)", strerror(errno), errno);
    }
    

    usleep(200);

    INFO("Camera video streaming on ...\n");
    return true;
}



static uint64_t parseDataImage( context_t * _ctx ) {
    uint64_t length = _ctx->cam_buf.bytesused;
    
    
    
    if(*(_ctx->buffers[_ctx->cam_buf.index].cam_start) != 0xFF || *(_ctx->buffers[_ctx->cam_buf.index].cam_start + 1) != 0xD8)
    {
        printf("CAMERA : WRONG Data Image ( index = %d )", _ctx->cam_buf.index  );
//             lskip= true;
        length = 0;
    }
    else {
            
            for(;;) {
                if(*(_ctx->buffers[_ctx->cam_buf.index].cam_start + length) == 0xD9) {
                    if(*(_ctx->buffers[_ctx->cam_buf.index].cam_start + length - 1) == 0xFF) {
                        length++;
                        break;
                    }
                    *(_ctx->buffers[_ctx->cam_buf.index].cam_start + length - 1) = 0xFF;
                    length++;
                    break;
                }
                length--;
            }
    }
    return length;
}



static void
signal_handle(int signum)
{
    printf("Quit due to exit command from user!\n");
    quit = true;
}

static bool
cuda_postprocess(context_t *ctx, int fd)
{
    if (ctx->enable_cuda)
    {
        // Create EGLImage from dmabuf fd
        ctx->egl_image = NvEGLImageFromFd(ctx->egl_display, fd);
        if (ctx->egl_image == NULL)
            ERROR_RETURN("Failed to map dmabuf fd (0x%X) to EGLImage",
                    ctx->render_dmabuf_fd);

        // Running algo process with EGLImage via GPU multi cores
        HandleEGLImage(&ctx->egl_image);

        // Destroy EGLImage
        NvDestroyEGLImage(ctx->egl_display, ctx->egl_image);
        ctx->egl_image = NULL;
    }

    return true;
}

static bool
start_capture(context_t * ctx)
{
    
    
    NvJPEGDecoder *m_JpegDec = NvJPEGDecoder::createJPEGDecoder("jdec_0");
    struct sigaction sig_action;
    struct pollfd fds[1];
    NvBufferTransformParams transParams;

    // Ensure a clean shutdown if user types <ctrl+c>
    sig_action.sa_handler = signal_handle;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sigaction(SIGINT, &sig_action, NULL);

    // Init the NvBufferTransformParams
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Smart;

    // Enable render profiling information
    ctx->renderer->enableProfiling();

    
	//struct timeval cur_tv, ;
	//struct timeval prev_tv;
	struct timeval tv_start;
	
	//gettimeofday(&prev_tv, NULL);
	//gettimeofday(&cur_tv, NULL);
	gettimeofday(&tv_start, NULL);
    
    printf(" CAMERA : Silence level : %d \n" , ctx->enable_silence);
                            
    fds[0].fd = ctx->cam_fd;
    fds[0].events = POLLIN;
    while (poll(fds, 1, 5000) > 0 && !quit)
    {
        if (fds[0].revents & POLLIN) {
            struct v4l2_buffer v4l2_buf;

            // Dequeue camera buff
            memset(&ctx->cam_buf, 0, sizeof(v4l2_buf));
            ctx->cam_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ctx->cam_buf.memory = V4L2_MEMORY_MMAP;
            if (ioctl(ctx->cam_fd, VIDIOC_DQBUF, &ctx->cam_buf) < 0)
                ERROR_RETURN("Failed to dequeue camera buff: %s (%d)",
                        strerror(errno), errno);


//             dev->disp_buf.type = dev->disp_type;
//             dev->disp_buf.memory = V4L2_MEMORY_USERPTR;
//             dev->disp_buf.index = dev->cam_buf.index;	
//             dev->disp_buf.m.userptr = (unsigned long) dev->buffers[dev->cam_buf.index].cam_offset;
//             dev->disp_buf.length = dev->buffers[dev->cam_buf.index].cam_length;
// 
//             dev->disp_buf.timestamp.tv_sec = tv_start.tv_sec;
// 	        dev->disp_buf.timestamp.tv_usec = tv_start.tv_usec + (1000000L / 30 * stream_count);
//             
// 		if (ioctl(dev->cam_fd, VIDIOC_QBUF, &dev->disp_buf) < 0) {
// 	        	perror("DISPLAY: stream_video_userptr VIDIOC_QBUF failed");
// 	        }
            
            
            ctx->frame++;
            
            int jret = -1;
            uint8_t *ptr_y = nullptr;
            int fd = -1;
                    

            int length = 0;
            
            
            if ( 0 < ctx->enable_silence ) {
                length = parseDataImage(ctx); //->cam_buf.bytesused;
            
            
                if( 0 < length ) {
                                    ptr_y = (uint8_t *)malloc((length+1)*sizeof(uint8_t));
                    memset(ptr_y ,'\0',length+1);
                    memcpy(ptr_y , ctx->buffers[ctx->cam_buf.index].cam_start, length);
                    int in_file_size = length;
                            
                    if ( 1 < ctx->enable_silence ) {
                        
                        uint32_t pixfmt, width, height;
                        if( ptr_y != nullptr ) {
                            
                                // printf(" CAMERA : decodeToFd %d\n", length );
                            jret = m_JpegDec->decodeToFd(fd, (uint8_t*)ptr_y, length, pixfmt, width, height);
                            
        //             cuda_postprocess(ctx, ctx->render_dmabuf_fd);
                            
                            
                            if ( 2 < ctx->enable_silence ) {
                                ctx->renderer->render(fd);
                            }

                            
                        }
                        else{
                            printf(" CAMERA : malloc => frame is NULL " );
                        }       

                    }
                    
                    if(ptr_y != nullptr) {
                        free(ptr_y);
                        ptr_y= nullptr;
                    }
                    
                }
                else{
                    printf(" CAMERA : parseDataImage => frame is empty " );
                }
            }
            // Enqueue camera buff
            if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &ctx->cam_buf))
                ERROR_RETURN("Failed to queue camera buffers: %s (%d)",
                        strerror(errno), errno);
        }
        

#ifdef ENABLE_VALGRIND
        if ( 2 < ctx->enable_valgrind ) {
            VALGRIND_DO_QUICK_LEAK_CHECK ;
        }
#endif
    }

    // Print profiling information when streaming stops.
    ctx->renderer->printProfilingStats();

    return true;
}

static bool
stop_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    // Stop v4l2 streaming
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type))
        ERROR_RETURN("Failed to stop streaming: %s (%d)",
                strerror(errno), errno);

    INFO("Camera video streaming off ...\n");
    return true;
}


int
main(int argc, char *argv[])
{
    context_t ctx;
    int error = 0;

    set_defaults(&ctx);

    CHECK_ERROR(parse_cmdline(&ctx, argc, argv), ended,
            "Invalid options specified");

    CHECK_ERROR(init_components(&ctx), cleanup,
            "Failed to initialize v4l2 components");

//     CHECK_ERROR(prepare_buffers(&ctx), cleanup,
//             "Failed to prepare v4l2 buffs");

        CHECK_ERROR(request_camera_buff(&ctx), cleanup,
             "Failed to request v4l2 buffs");
    
    
    CHECK_ERROR(start_stream(&ctx), cleanup,
            "Failed to start streaming");

    CHECK_ERROR(start_capture(&ctx), cleanup,
            "Failed to start capturing")

    CHECK_ERROR(stop_stream(&ctx), cleanup,
            "Failed to stop streaming");

cleanup:
    if (ctx.cam_fd > 0)
        close(ctx.cam_fd);

    if (ctx.renderer != NULL)
        delete ctx.renderer;

    if (ctx.egl_display && !eglTerminate(ctx.egl_display))
        printf("Failed to terminate EGL display connection\n");

    if (ctx.g_buff != NULL)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++)
            if (ctx.g_buff[i].dmabuff_fd)
                NvBufferDestroy(ctx.g_buff[i].dmabuff_fd);
        free(ctx.g_buff);
    }

    NvBufferDestroy(ctx.render_dmabuf_fd);

    if (error)
        printf("App run failed\n");
    else
        printf("App run was successful\n");

               if( ctx.buffers != nullptr) {
                for (int i = 0; i < UVC_BUFFER_NUM; i++) {
                    
                    if(ctx.buffers[i].cam_start){
                        munmap(ctx.buffers[i].cam_start, ctx.buffers[i].cam_length);
                    }
                    if(ctx.buffers[i].disp_start)
                        munmap(ctx.buffers[i].disp_start, ctx.buffers[i].disp_length);
                    ctx.buffers[i].cam_start = nullptr;
                    ctx.buffers[i].disp_start = nullptr;
                }
    //             ctxbuffers = nullptr;
            }
    
ended:
    
    return -error;
}
