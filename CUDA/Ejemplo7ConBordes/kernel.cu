#include <cuda.h>
#include <math.h>

//Debug with printf
#include <stdio.h>

#include "kernel.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

double get_time2(){
	static struct timeval 	tv0;
	double time_, mytime;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	mytime = time_/1000000;
	return(mytime);
}

/* EDGE AND HYSTERESIS_THRESHOLDING */

__global__ void edgeAndHysteresisThresholding(float *image_out, float *G, float *phi, int lowthres, int hithres, int height, int width){
	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int ii,jj;
	
	float gAux[3][3];
	int pedge,goOut;

	if( (i < height-3 && j < width-3) && (i > 2 && j > 2) ){
			
			//Con esto va un poco más rapido
			gAux[0][0] = G[(i-1)*width+j-1];			
			gAux[0][1] = G[(i-1)*width+j];			
			gAux[0][2] = G[(i-1)*width+j+1];			
			
			gAux[1][0] = G[i*width+j-1];			
			gAux[1][1] = G[i*width+j];							
			gAux[1][2] = G[i*width+j+1];			
			
			gAux[2][0] = G[(i+1)*width+j-1];
			gAux[2][1] = G[(i+1)*width+j];
			gAux[2][2] = G[(i+1)*width+j+1];

			pedge = 0;

			if(phi[i*width+j] == 0){
				if(gAux[1][1]>gAux[1][2] && gAux[1][1]>gAux[1][0]) //edge is in N-S
					pedge = 1;

			} else if(phi[i*width+j] == 45) {
				if(gAux[1][1]>gAux[2][2] && gAux[1][1]>gAux[0][0]) // edge is in NW-SE
					pedge = 1;

			} else if(phi[i*width+j] == 90) {
				if(gAux[1][1]>gAux[2][1] && gAux[1][1]>gAux[0][1]) //edge is in E-W
					pedge = 1;

			} else if(phi[i*width+j] == 135) {
				if(gAux[1][1]>gAux[2][0] && gAux[1][1]>gAux[0][2]) // edge is in NE-SW
					pedge = 1;
			}



			image_out[i*width+j] = 0;
			if(gAux[1][1]>hithres && pedge)
				image_out[i*width+j] = 255;
			else if(pedge && gAux[1][1]>=lowthres && gAux[1][1]<hithres){
				// check neighbours 3x3
				goOut = 0;
				for (ii=0;goOut==0 && ii<=2 ; ii++)
					for (jj=0;goOut==0 && jj<=2 ; jj++)
						if (gAux[ii][jj]>hithres){
							image_out[i*width+j] = 255;
							goOut = 1;						
						}
											
						
				
			}
			
							
		
	} else if(i < height && j < width)
		image_out[i*width+j] = 0;
	
}

/*IMAGE GRADIENT*/

__global__ void imageGradient(float *NR, float *G, float *phi, int height, int width,int blocksY, int blocksX){
	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	float PI = 3.141593,Gx,Gy,phiAux;

	if( (i < height-2 && j < width-2) && (i > 1 && j > 1) ){
		
		// Intensity gradient of the image
		if( by == 0 || bx == 0 || by == blocksY-1 || bx == blocksX-1){
			Gx = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		}
		else{
			__shared__ float tile[NTHREADS2D+4][NTHREADS2D+4+1];
			
			tile[ty + 2][tx + 2] = NR[i*width+j];
			
			//Corners
			if( ty == 0 && tx == 0){
				tile[0][0] = NR[(i-2)*width+j-2];	tile[2][0] = NR[i*width+j-2];	
				tile[0][1] = NR[(i-2)*width+j-1];	tile[1][1] = NR[(i-1)*width+j-1];
				tile[0][2] = NR[(i-2)*width+j];		tile[1][2] = NR[(i-1)*width+j];
				tile[1][0] = NR[(i-1)*width+j-2];	tile[2][1] = NR[i*width+j-1];
			}
			else if( ty == 0 && tx == NTHREADS2D - 1 ){
				tile[0][NTHREADS2D + 3] = NR[(i-2)*width+j+2];		tile[2][NTHREADS2D + 3] = NR[i*width+j+2];	
				tile[0][NTHREADS2D + 2] = NR[(i-2)*width+j+1];		tile[1][NTHREADS2D + 2] = NR[(i-1)*width+j+1];
				tile[0][NTHREADS2D + 1] = NR[(i-2)*width+j];		tile[1][NTHREADS2D + 1] = NR[(i-1)*width+j];
				tile[1][NTHREADS2D + 3] = NR[(i-1)*width+j+2];		tile[2][NTHREADS2D + 2] = NR[i*width+j+1];
							
			}
			else if( ty == NTHREADS2D - 1 && tx == 0 ){
				tile[NTHREADS2D + 3][0] = NR[(i+2)*width+j-2];		tile[NTHREADS2D + 1][0] = NR[i*width+j-2];	
				tile[NTHREADS2D + 3][1] = NR[(i+2)*width+j-1];		tile[NTHREADS2D + 2][1] = NR[(i+1)*width+j-1];
				tile[NTHREADS2D + 3][2] = NR[(i+2)*width+j];		tile[NTHREADS2D + 2][2] = NR[(i+1)*width+j];
				tile[NTHREADS2D + 2][0] = NR[(i+1)*width+j-2];		tile[NTHREADS2D + 1][1] = NR[i*width+j-1];
			
			}
			else if( ty == NTHREADS2D - 1 && tx == NTHREADS2D - 1 ){
				tile[NTHREADS2D + 3][NTHREADS2D + 3] = NR[(i+2)*width+j+2];	tile[NTHREADS2D + 1][NTHREADS2D + 3] = NR[i*width+j+2];	
				tile[NTHREADS2D + 3][NTHREADS2D + 2] = NR[(i+2)*width+j+1];	tile[NTHREADS2D + 2][NTHREADS2D + 2] = NR[(i+1)*width+j+1];
				tile[NTHREADS2D + 3][NTHREADS2D + 1] = NR[(i+2)*width+j];	tile[NTHREADS2D + 2][NTHREADS2D + 1] = NR[(i+1)*width+j];
				tile[NTHREADS2D + 2][NTHREADS2D + 3] = NR[(i+1)*width+j+2];	tile[NTHREADS2D + 1][NTHREADS2D + 2] = NR[i*width+j+1];

			}//Edges
			else if( ty == 0 && (tx > 0 && tx < NTHREADS2D - 1) ){
				tile[0][tx + 2] = NR[(i-2)*width+j];
				tile[1][tx + 2] = NR[(i-1)*width+j];
			}		
			else if( (ty > 0 && ty < NTHREADS2D - 1) && tx == NTHREADS2D - 1 ){
				tile[ty + 2][NTHREADS2D + 2] = NR[i*width+j+1];
				tile[ty + 2][NTHREADS2D + 3] = NR[i*width+j+2];
			}		
			else if( ty == NTHREADS2D - 1 && (tx > 0 && tx < NTHREADS2D - 1) ){
				tile[NTHREADS2D + 3][tx + 2] = NR[(i+2)*width+j];
				tile[NTHREADS2D + 2][tx + 2] = NR[(i+1)*width+j];
			}		
			else if( (ty < NTHREADS2D - 1 && ty > 0 ) && tx == 0 ){
				tile[ty + 2][0] = NR[i*width+j-2];
				tile[ty + 2][1] = NR[i*width+j-1];
			}		


			__syncthreads();

			Gx = 
				 (1.0*tile[ty][tx] +  2.0*tile[ty][tx+1] +  (-2.0)*tile[ty][tx+3] + (-1.0)*tile[ty][tx+4]
				+ 4.0*tile[ty+1][tx] +  8.0*tile[ty+1][tx+1] +  (-8.0)*tile[ty+1][tx+3] + (-4.0)*tile[ty+1][tx+4]
				+ 6.0*tile[ty+2][tx] + 12.0*tile[ty+2][tx+1] + (-12.0)*tile[ty+2][tx+3] + (-6.0)*tile[ty+2][tx+4]
				+ 4.0*tile[ty+3][tx] +  8.0*tile[ty+3][tx+1] +  (-8.0)*tile[ty+3][tx+3] + (-4.0)*tile[ty+3][tx+4]
				+ 1.0*tile[ty+4][tx] +  2.0*tile[ty+4][tx+1] +  (-2.0)*tile[ty+4][tx+3] + (-1.0)*tile[ty+4][tx+4]);


			Gy = 
				 ((-1.0)*tile[ty][tx] + (-4.0)*tile[ty][tx+1] +  (-6.0)*tile[ty][tx+2] + (-4.0)*tile[ty][tx+3] + (-1.0)*tile[ty][tx+4]
				+ (-2.0)*tile[ty+1][tx] + (-8.0)*tile[ty+1][tx+1] + (-12.0)*tile[ty+1][tx+2] + (-8.0)*tile[ty+1][tx+3] + (-2.0)*tile[ty+1][tx+4]
				+    2.0*tile[ty+3][tx] +    8.0*tile[ty+3][tx+1] +    12.0*tile[ty+3][tx+2] +    8.0*tile[ty+3][tx+3] +    2.0*tile[ty+3][tx+4]
				+    1.0*tile[ty+4][tx] +    4.0*tile[ty+4][tx+1] +     6.0*tile[ty+4][tx+2] +    4.0*tile[ty+4][tx+3] +    1.0*tile[ty+4][tx+4]);

			

		}
		
		//G = √Gx²+Gy²
		G[i*width+j] = sqrtf( (Gx*Gx) + (Gy*Gy) );	
		phiAux = fabsf( atan2f(fabsf(Gy),fabsf(Gx)) );


		if (phiAux<=PI/8 )
			phi[i*width+j] = 0;
		else if ( phiAux<= 3*(PI/8))
			phi[i*width+j] = 45;
		else if ( phiAux <= 5*(PI/8))
			phi[i*width+j] = 90;
		else if ( phiAux <= 7*(PI/8))
			phi[i*width+j] = 135;
		else phi[i*width+j] = 0;

	}
	
}


/*NOISE REDUCTION*/

__global__ void noiseReduction(float *im,float *im_out, int height, int width, int blocksY, int blocksX){
	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	if( (i < height-2 && j < width-2) && (i > 1 && j > 1) ){
		
		if( by == 0 || bx == 0 || by == blocksY-1 || bx == blocksX-1){
			im_out[i*width+j] = (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
							+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
							+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
							+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
							+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
							/159.0;		
		}
		else{
		
			__shared__ float tile[NTHREADS2D+4][NTHREADS2D+4+1];
			
			tile[ty + 2][tx + 2] = im[i*width+j];
			
			//Corners
			if( ty == 0 && tx == 0){
				tile[0][0] = im[(i-2)*width+j-2];	tile[2][0] = im[i*width+j-2];	
				tile[0][1] = im[(i-2)*width+j-1];	tile[1][1] = im[(i-1)*width+j-1];
				tile[0][2] = im[(i-2)*width+j];		tile[1][2] = im[(i-1)*width+j];
				tile[1][0] = im[(i-1)*width+j-2];	tile[2][1] = im[i*width+j-1];
			}
			else if( ty == 0 && tx == NTHREADS2D - 1 ){
				tile[0][NTHREADS2D + 3] = im[(i-2)*width+j+2];		tile[2][NTHREADS2D + 3] = im[i*width+j+2];	
				tile[0][NTHREADS2D + 2] = im[(i-2)*width+j+1];		tile[1][NTHREADS2D + 2] = im[(i-1)*width+j+1];
				tile[0][NTHREADS2D + 1] = im[(i-2)*width+j];		tile[1][NTHREADS2D + 1] = im[(i-1)*width+j];
				tile[1][NTHREADS2D + 3] = im[(i-1)*width+j+2];		tile[2][NTHREADS2D + 2] = im[i*width+j+1];
							
			}
			else if( ty == NTHREADS2D - 1 && tx == 0 ){
				tile[NTHREADS2D + 3][0] = im[(i+2)*width+j-2];		tile[NTHREADS2D + 1][0] = im[i*width+j-2];	
				tile[NTHREADS2D + 3][1] = im[(i+2)*width+j-1];		tile[NTHREADS2D + 2][1] = im[(i+1)*width+j-1];
				tile[NTHREADS2D + 3][2] = im[(i+2)*width+j];		tile[NTHREADS2D + 2][2] = im[(i+1)*width+j];
				tile[NTHREADS2D + 2][0] = im[(i+1)*width+j-2];		tile[NTHREADS2D + 1][1] = im[i*width+j-1];
			
			}
			else if( ty == NTHREADS2D - 1 && tx == NTHREADS2D - 1 ){
				tile[NTHREADS2D + 3][NTHREADS2D + 3] = im[(i+2)*width+j+2];	tile[NTHREADS2D + 1][NTHREADS2D + 3] = im[i*width+j+2];	
				tile[NTHREADS2D + 3][NTHREADS2D + 2] = im[(i+2)*width+j+1];	tile[NTHREADS2D + 2][NTHREADS2D + 2] = im[(i+1)*width+j+1];
				tile[NTHREADS2D + 3][NTHREADS2D + 1] = im[(i+2)*width+j];	tile[NTHREADS2D + 2][NTHREADS2D + 1] = im[(i+1)*width+j];
				tile[NTHREADS2D + 2][NTHREADS2D + 3] = im[(i+1)*width+j+2];	tile[NTHREADS2D + 1][NTHREADS2D + 2] = im[i*width+j+1];

			}//Edges
			else if( ty == 0 && (tx > 0 && tx < NTHREADS2D - 1) ){
				tile[0][tx + 2] = im[(i-2)*width+j];
				tile[1][tx + 2] = im[(i-1)*width+j];
			}		
			else if( (ty > 0 && ty < NTHREADS2D - 1) && tx == NTHREADS2D - 1 ){
				tile[ty + 2][NTHREADS2D + 2] = im[i*width+j+1];
				tile[ty + 2][NTHREADS2D + 3] = im[i*width+j+2];
			}		
			else if( ty == NTHREADS2D - 1 && (tx > 0 && tx < NTHREADS2D - 1) ){
				tile[NTHREADS2D + 3][tx + 2] = im[(i+2)*width+j];
				tile[NTHREADS2D + 2][tx + 2] = im[(i+1)*width+j];
			}		
			else if( (ty < NTHREADS2D - 1 && ty > 0 ) && tx == 0 ){
				tile[ty + 2][0] = im[i*width+j-2];
				tile[ty + 2][1] = im[i*width+j-1];
			}		


			__syncthreads();

			im_out[i*width+j] = (2.0*tile[ty][tx] +  4.0*tile[ty][tx+1] +  5.0*tile[ty][tx+2] +  4.0*tile[ty][tx+3] + 2.0*tile[ty][tx+4]
					+ 4.0*tile[ty+1][tx] +  9.0*tile[ty+1][tx+1] + 12.0*tile[ty+1][tx+2] +  9.0*tile[ty+1][tx+3] + 4.0*tile[ty+1][tx+4]
					+ 5.0*tile[ty+2][tx] + 12.0*tile[ty+2][tx+1] + 15.0*tile[ty+2][tx+2] + 12.0*tile[ty+2][tx+3] + 5.0*tile[ty+2][tx+4]
					+ 4.0*tile[ty+3][tx] +  9.0*tile[ty+3][tx+1] + 12.0*tile[ty+3][tx+2] +  9.0*tile[ty+3][tx+3] + 4.0*tile[ty+3][tx+4]
					+ 2.0*tile[ty+4][tx] +  4.0*tile[ty+4][tx+1] +  5.0*tile[ty+4][tx+2] +  4.0*tile[ty+4][tx+3] + 2.0*tile[ty+4][tx+4])
					/159.0;
		}
	




	}
	
}

void cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{
	float *IM_IN ,*IM_OUT,*G,*phi;


	/* Mallocs GPU */
	cudaMalloc((void**)&IM_IN, sizeof(float)*height*width);
	cudaMalloc((void**)&IM_OUT, sizeof(float)*height*width);

	cudaMalloc((void**)&G, sizeof(float)*height*width);
	cudaMalloc((void**)&phi, sizeof(float)*height*width);

	
	/* CPU->GPU */
	cudaMemcpy(IM_IN, im, sizeof(float)*height*width, cudaMemcpyHostToDevice);
	
	dim3 dimBlock(NTHREADS2D,NTHREADS2D,1);

	int blocksX,blocksY;
	blocksX = ( height % NTHREADS2D == 0 ) ? height/NTHREADS2D : height/NTHREADS2D + 1 ;
	blocksY = ( width % NTHREADS2D == 0 ) ? width/NTHREADS2D : width/NTHREADS2D + 1 ;
	dim3 dimGrid(blocksY,blocksX,1);

	printf("BlocksX: %i \n",blocksX);
	printf("BlocksY: %i \n",blocksY);
	
	double t0 = get_time2();
	noiseReduction<<<dimGrid,dimBlock>>>(IM_IN,IM_OUT, height, width,blocksY,blocksX);
	cudaThreadSynchronize();
	double t1 = get_time2();
	printf("NR=%f .s\n", t1-t0);

	t0 = get_time2();
	imageGradient<<<dimGrid,dimBlock>>>(IM_OUT, G, phi, height, width,blocksY,blocksX);
	cudaThreadSynchronize();
	t1 = get_time2();
	printf("IG=%f .s\n", t1-t0);	

	t0 = get_time2();
	edgeAndHysteresisThresholding<<<dimGrid,dimBlock>>>(IM_OUT, G, phi,level/2, 2*(level), height, width);
	cudaThreadSynchronize();
	t1 = get_time2();
	printf("EDGE/THRES=%f .s\n", t1-t0);	
		
	/* GPU->CPU */
	cudaMemcpy(image_out, IM_OUT, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
		
	// Free device memory
	cudaFree(IM_IN);
	cudaFree(IM_OUT);
	

	cudaFree(G);
	cudaFree(phi);

}
