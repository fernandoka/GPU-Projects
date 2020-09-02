#ifndef _OCL_H
#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#define _OCL_H

int nbodiesOCL(float *m, float*x, float *y, float *z,
	float *vx,float *vy, float *vz ,
	int nBodies,float dt,const int nIters);
#endif
