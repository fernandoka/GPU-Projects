#ifndef _OCL_H
#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#define _OCL_H

int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width);
#endif
