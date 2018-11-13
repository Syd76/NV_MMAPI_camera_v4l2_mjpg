#!/bin/bash


# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
        echo "** Trapped CTRL-C"
        pkill -9 ./camera_v4l2_mjpg
}


valgrind -v --log-file="valgrind_memcheck2.log" --tool=memcheck --leak-check=yes --gen-suppressions=yes --show-reachable=yes --num-callers=20 --track-fds=yes --track-origins=yes  ./camera_v4l2_mjpg -d /dev/video0 -s 3840x2160 -m 3 -l -v






