

# Nvidia MMAPI : camera_v4l2_mjpg
====


### sample for Nvidia MMAPI  (TX2) from "12_camera_v4l2_cuda"
====



## Valgring : 

- enable Valgrind : for Macro **VALGRIND_DO_QUICK_LEAK_CHECK** in code :

*export TEGRA_ARMABI=aarch64-linux-gnu ; make -j4 **ENABLE_VALGRIND=yes***


## Option :

* "-l" enable VALGRIND_DO_QUICK_LEAK_CHECK in code ( must be compiled with "ENABLE_VALGRIND=yes" makefile directive )

* "-m" used for level stream ( capture to display )

## LOG

* ldd_camera_v4l2_mjpg.log 
	- result of "lld camera_v4l2_mjpg"
* nv_tegra_release.log 
	- Nvidia release 
* valgrind_memcheck2.log
	- log from capture_valgrind.sh and VALGRIND_DO_QUICK_LEAK_CHECK macro
