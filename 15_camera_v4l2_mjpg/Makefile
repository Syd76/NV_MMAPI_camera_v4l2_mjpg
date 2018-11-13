###############################################################################
#
# Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


### Tegra rules :
# export TEGRA_ARMABI=aarch64-linux-gnu ; make -j4 ENABLE_VALGRIND=yes

### Capture only
# ./camera_v4l2_mjpg -d /dev/video0 -s 3840x2160 -v -m 0

### Capture+parse JPEG
# ./camera_v4l2_mjpg -d /dev/video0 -s 3840x2160 -v -m 1

### Capture + parse + decoder
# ./camera_v4l2_mjpg uda -d /dev/video0 -s 3840x2160 -v -m 2

### Capture + parse + decoder + Display
# ./camera_v4l2_mjpg -d /dev/video0 -s 3840x2160 -v -m 3


$(info $$ENABLE_VALGRIND for ${ENABLE_VALGRIND} )

include ../Rules.mk

ifeq (${ENABLE_VALGRIND},yes)
	CPPFLAGS += -DENABLE_VALGRIND
endif



APP := camera_v4l2_mjpg 

SRCS := \
	camera_v4l2_mjpg.cpp \
	$(wildcard $(CLASS_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)

OBJS += \
	$(ALGO_CUDA_DIR)/NvAnalysis.o \
	$(ALGO_CUDA_DIR)/NvCudaProc.o

all: $(APP)
	@echo "CPPFLAGS = "${CPPFLAGS}

$(CLASS_DIR)/%.o: $(CLASS_DIR)/%.cpp
	$(AT)$(MAKE) -C $(CLASS_DIR)

$(ALGO_CUDA_DIR)/%.o: $(ALGO_CUDA_DIR)/%.cpp
	$(AT)$(MAKE) -C $(ALGO_CUDA_DIR)

$(ALGO_CUDA_DIR)/%.o: $(ALGO_CUDA_DIR)/%.cu
	$(AT)$(MAKE) -C $(ALGO_CUDA_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -c $<

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CPP) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)

clean:
	$(AT)rm -rf $(APP) $(OBJS)
