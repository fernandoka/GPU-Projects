#ifndef _KERNEL_H

#define _KERNEL_H

#define NTHREADS2D 32

#ifdef __cplusplus
extern "C"
#endif
void cannyGPU(float *im, float *image_out, 
	float level,
	int height, int width);

#endif
