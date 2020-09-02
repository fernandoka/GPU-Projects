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


__global__ void imageGradientHysteresisThresholding(float *NR, float *image_out, int lowthres, int hithres,int height, int width,int blocksY, int blocksX){
	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int by = blockIdx.y;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	float PI = 3.141593,Gx,Gy,phiAux,phi;

	int a,b,c,d,A,B,C,D,X,Y;
	int ii,jj,pedge,goOut;
	//Inicialize the "boolean" variables.
	A=B=C=D=a=b=c=d=0;	
	
	pedge = 0; image_out[i*width+j] = 0;
	
		if( !(by == 0 || bx == 0 || by == blocksY-1 || bx == blocksX-1) ){
			
				// Intensity gradient of the image
				__shared__ float tile[NTHREADS2D+6][NTHREADS2D+6];
				__shared__ float G_tile[NTHREADS2D+2][NTHREADS2D+2];

				tile[ty + 3][tx + 3] = NR[i*width+j];
				
				//Corners
				if( ty<3 && tx<3){
					A=1; ii=ty; jj=tx;
					tile[ty][tx] = NR[(i-3)*width+(j-3)];

				}
				else if( ty<3 && tx>NTHREADS2D-4 ){
					B=1; ii=ty; jj=tx+2;
					tile[ty][tx+6] = NR[(i-3)*width+(j+3)];
								
				}
				else if( ty>NTHREADS2D-4 && tx>NTHREADS2D-4  ){
					C=1; ii=ty+2; jj=tx+2;
					tile[ty+6][tx+6] = NR[(i+3)*width+(j+3)];
				}
				else if( ty>NTHREADS2D-4 && tx<3){
					D=1; ii=ty+2; jj=tx;
					tile[ty+6][tx] = NR[(i+3)*width+(j-3)];
				}
				//Edges
				if( ty == 0 && (tx >= 0 && tx <= NTHREADS2D - 1) ){
					tile[0][tx+3] = NR[(i-3)*width+j];	tile[1][tx+3] = NR[(i-2)*width+j];
					tile[2][tx+3] = NR[(i-1)*width+j];	b=1;		
				}		

				if( (ty >= 0 && ty <= NTHREADS2D - 1) && tx == NTHREADS2D - 1 ){
					tile[ty+3][NTHREADS2D+5] = NR[(i)*width+(j+3)];	tile[ty+3][NTHREADS2D+4] = NR[(i)*width+(j+2)];
					tile[ty+3][NTHREADS2D+3] = NR[(i)*width+(j+1)];	c=1;
				}		

				if( ty == NTHREADS2D - 1 && (tx >= 0 && tx <= NTHREADS2D - 1) ){
					tile[NTHREADS2D+5][tx+3] = NR[(i+3)*width+j];	tile[NTHREADS2D+4][tx+3] = NR[(i+2)*width+j];
					tile[NTHREADS2D+3][tx+3] = NR[(i+1)*width+j];	d=1;
				}		

				if( (ty <= NTHREADS2D - 1 && ty >= 0 ) && tx == 0 ){
					tile[ty+3][0] = NR[(i)*width+(j-3)];	tile[ty+3][1] = NR[(i)*width+(j-2)];
					tile[ty+3][2] = NR[(i)*width+(j-1)];	a=1;
				}		


				__syncthreads();
				Gx = 
					 (1.0*tile[ty+1][tx+1] +  2.0*tile[ty+1][tx+2] +  (-2.0)*tile[ty+1][tx+4] + (-1.0)*tile[ty+1][tx+5]
					+ 4.0*tile[ty+2][tx+1] +  8.0*tile[ty+2][tx+2] +  (-8.0)*tile[ty+2][tx+4] + (-4.0)*tile[ty+2][tx+5]
					+ 6.0*tile[ty+3][tx+1] + 12.0*tile[ty+3][tx+2] + (-12.0)*tile[ty+3][tx+4] + (-6.0)*tile[ty+3][tx+5]
					+ 4.0*tile[ty+4][tx+1] +  8.0*tile[ty+4][tx+2] +  (-8.0)*tile[ty+4][tx+4] + (-4.0)*tile[ty+4][tx+5]
					+ 1.0*tile[ty+5][tx+1] +  2.0*tile[ty+5][tx+2] +  (-2.0)*tile[ty+5][tx+4] + (-1.0)*tile[ty+5][tx+5]);


				Gy = 
					 ((-1.0)*tile[ty+1][tx+1] + (-4.0)*tile[ty+1][tx+2] +  (-6.0)*tile[ty+1][tx+3] + (-4.0)*tile[ty+1][tx+4] + (-1.0)*tile[ty+1][tx+5]
					+ (-2.0)*tile[ty+2][tx+1] + (-8.0)*tile[ty+2][tx+2] + (-12.0)*tile[ty+2][tx+3] + (-8.0)*tile[ty+2][tx+4] + (-2.0)*tile[ty+2][tx+5]
					+    2.0*tile[ty+4][tx+1] +    8.0*tile[ty+4][tx+2] +    12.0*tile[ty+4][tx+3] +    8.0*tile[ty+4][tx+4] +    2.0*tile[ty+4][tx+5]
					+    1.0*tile[ty+5][tx+1] +    4.0*tile[ty+5][tx+2] +     6.0*tile[ty+5][tx+3] +    4.0*tile[ty+5][tx+4] +    1.0*tile[ty+5][tx+5]);


				//G = √Gx²+Gy²
				G_tile[ty+1][tx+1] = sqrtf( (Gx*Gx) + (Gy*Gy) );	
				phiAux = fabsf( atan2f(fabsf(Gy),fabsf(Gx)) );


				if (phiAux<=PI/8 )
					phi = 0;
				else if ( phiAux<= 3*(PI/8))
					phi = 45;
				else if ( phiAux <= 5*(PI/8))
					phi = 90;
				else if ( phiAux <= 7*(PI/8))
					phi = 135;
				else phi = 0;


				if(A==1 || B==1 || C==1 || D==1){
					
					if(A==1 || B==1)
						X=ty;
					else
						X=ty+1;

					if(C==1 || B==1)
						Y=tx+1;
					else
						Y=tx;

					Gx = 
						 (1.0*tile[ii][jj] +  2.0*tile[ii][jj+1] +  (-2.0)*tile[ii][jj+3] + (-1.0)*tile[ii][jj+4]
						+ 4.0*tile[ii+1][jj] +  8.0*tile[ii+1][jj+1] +  (-8.0)*tile[ii+1][jj+3] + (-4.0)*tile[ii+1][jj+4]
						+ 6.0*tile[ii+2][jj] + 12.0*tile[ii+2][jj+1] + (-12.0)*tile[ii+2][jj+3] + (-6.0)*tile[ii+2][jj+4]
						+ 4.0*tile[ii+3][jj] +  8.0*tile[ii+3][jj+1] +  (-8.0)*tile[ii+3][jj+3] + (-4.0)*tile[ii+3][jj+4]
						+ 1.0*tile[ii+4][jj] +  2.0*tile[ii+4][jj+1] +  (-2.0)*tile[ii+4][jj+3] + (-1.0)*tile[ii+4][jj+4]);


					Gy = 
						 ((-1.0)*tile[ii][jj] + (-4.0)*tile[ii][jj+1] +  (-6.0)*tile[ii][jj+2] + (-4.0)*tile[ii][jj+3] + (-1.0)*tile[ii][jj+4]
						+ (-2.0)*tile[ii+1][jj] + (-8.0)*tile[ii+1][jj+1] + (-12.0)*tile[ii+1][jj+2] + (-8.0)*tile[ii+1][jj+3] + (-2.0)*tile[ii+1][jj+4]
						+    2.0*tile[ii+3][jj] +    8.0*tile[ii+3][jj+1] +    12.0*tile[ii+3][jj+2] +    8.0*tile[ii+3][jj+3] +    2.0*tile[ii+3][jj+4]
						+    1.0*tile[ii+4][jj] +    4.0*tile[ii+4][jj+1] +     6.0*tile[ii+4][jj+2] +    4.0*tile[ii+4][jj+3] +    1.0*tile[ii+4][jj+4]);

					
					G_tile[X][Y] = sqrtf( (Gx*Gx) + (Gy*Gy) );

				}


				else if(a==1){

					Gx = 
						 (1.0*tile[ty+1][0] +  2.0*tile[ty+1][1] +  (-2.0)*tile[ty+1][3] + (-1.0)*tile[ty+1][4]
						+ 4.0*tile[ty+2][0] +  8.0*tile[ty+2][1] +  (-8.0)*tile[ty+2][3] + (-4.0)*tile[ty+2][4]
						+ 6.0*tile[ty+3][0] + 12.0*tile[ty+3][1] + (-12.0)*tile[ty+3][3] + (-6.0)*tile[ty+3][4]
						+ 4.0*tile[ty+4][0] +  8.0*tile[ty+4][1] +  (-8.0)*tile[ty+4][3] + (-4.0)*tile[ty+4][4]
						+ 1.0*tile[ty+5][0] +  2.0*tile[ty+5][1] +  (-2.0)*tile[ty+5][3] + (-1.0)*tile[ty+5][4]);


					Gy = 
						 ((-1.0)*tile[ty+1][0] + (-4.0)*tile[ty+1][1] +  (-6.0)*tile[ty+1][2] + (-4.0)*tile[ty+1][3] + (-1.0)*tile[ty+1][4]
						+ (-2.0)*tile[ty+2][0] + (-8.0)*tile[ty+2][1] + (-12.0)*tile[ty+2][2] + (-8.0)*tile[ty+2][3] + (-2.0)*tile[ty+2][4]
						+    2.0*tile[ty+4][0] +    8.0*tile[ty+4][1] +    12.0*tile[ty+4][2] +    8.0*tile[ty+4][3] +    2.0*tile[ty+4][4]
						+    1.0*tile[ty+5][0] +    4.0*tile[ty+5][1] +     6.0*tile[ty+5][2] +    4.0*tile[ty+5][3] +    1.0*tile[ty+5][4]);

		
					G_tile[ty+1][0] = sqrtf( (Gx*Gx) + (Gy*Gy) );	
					
				}
				else if(b==1){
					Gx = 
						 (1.0*tile[0][tx+1] +  2.0*tile[0][tx+2] +  (-2.0)*tile[0][tx+4] + (-1.0)*tile[0][tx+5]
						+ 4.0*tile[1][tx+1] +  8.0*tile[1][tx+2] +  (-8.0)*tile[1][tx+4] + (-4.0)*tile[1][tx+5]
						+ 6.0*tile[2][tx+1] + 12.0*tile[2][tx+2] + (-12.0)*tile[2][tx+4] + (-6.0)*tile[2][tx+5]
						+ 4.0*tile[3][tx+1] +  8.0*tile[3][tx+2] +  (-8.0)*tile[3][tx+4] + (-4.0)*tile[3][tx+5]
						+ 1.0*tile[4][tx+1] +  2.0*tile[4][tx+2] +  (-2.0)*tile[4][tx+4] + (-1.0)*tile[4][tx+5]);


					Gy = 
						 ((-1.0)*tile[0][tx+1] + (-4.0)*tile[0][tx+2] +  (-6.0)*tile[0][tx+3] + (-4.0)*tile[0][tx+4] + (-1.0)*tile[0][tx+5]
						+ (-2.0)*tile[1][tx+1] + (-8.0)*tile[1][tx+2] + (-12.0)*tile[1][tx+3] + (-8.0)*tile[1][tx+4] + (-2.0)*tile[1][tx+5]
						+    2.0*tile[3][tx+1] +    8.0*tile[3][tx+2] +    12.0*tile[3][tx+3] +    8.0*tile[3][tx+4] +    2.0*tile[3][tx+5]
						+    1.0*tile[4][tx+1] +    4.0*tile[4][tx+2] +     6.0*tile[4][tx+3] +    4.0*tile[4][tx+4] +    1.0*tile[4][tx+5]);

		
					G_tile[0][tx+1] = sqrtf( (Gx*Gx) + (Gy*Gy) );

				}
				else if(c==1){

					Gx = 
						 (1.0*tile[ty+1][NTHREADS2D+1] +  2.0*tile[ty+1][NTHREADS2D+2] +  (-2.0)*tile[ty+1][NTHREADS2D+4] + (-1.0)*tile[ty+1][NTHREADS2D+5]
						+ 4.0*tile[ty+2][NTHREADS2D+1] +  8.0*tile[ty+2][NTHREADS2D+2] +  (-8.0)*tile[ty+2][NTHREADS2D+4] + (-4.0)*tile[ty+2][NTHREADS2D+5]
						+ 6.0*tile[ty+3][NTHREADS2D+1] + 12.0*tile[ty+3][NTHREADS2D+2] + (-12.0)*tile[ty+3][NTHREADS2D+4] + (-6.0)*tile[ty+3][NTHREADS2D+5]
						+ 4.0*tile[ty+4][NTHREADS2D+1] +  8.0*tile[ty+4][NTHREADS2D+2] +  (-8.0)*tile[ty+4][NTHREADS2D+4] + (-4.0)*tile[ty+4][NTHREADS2D+5]
						+ 1.0*tile[ty+5][NTHREADS2D+1] +  2.0*tile[ty+5][NTHREADS2D+2] +  (-2.0)*tile[ty+5][NTHREADS2D+4] + (-1.0)*tile[ty+5][NTHREADS2D+5]);


					Gy = 
						 ((-1.0)*tile[ty+1][NTHREADS2D+1] + (-4.0)*tile[ty+1][NTHREADS2D+2] +  (-6.0)*tile[ty+1][NTHREADS2D+3] + (-4.0)*tile[ty+1][NTHREADS2D+4] + 
						(-1.0)*tile[ty+1][NTHREADS2D+5]
						+ (-2.0)*tile[ty+2][NTHREADS2D+1] + (-8.0)*tile[ty+2][NTHREADS2D+2] + (-12.0)*tile[ty+2][NTHREADS2D+3] + (-8.0)*tile[ty+2][NTHREADS2D+4] + 
						(-2.0)*tile[ty+2][NTHREADS2D+5]
						+    2.0*tile[ty+4][NTHREADS2D+1] +    8.0*tile[ty+4][NTHREADS2D+2] +    12.0*tile[ty+4][NTHREADS2D+3] +    8.0*tile[ty+4][NTHREADS2D+4] +    
						2.0*tile[ty+4][NTHREADS2D+5]
						+    1.0*tile[ty+5][NTHREADS2D+1] +    4.0*tile[ty+5][NTHREADS2D+2] +     6.0*tile[ty+5][NTHREADS2D+3] +    4.0*tile[ty+5][NTHREADS2D+4] +    
						1.0*tile[ty+5][NTHREADS2D+5]);


					G_tile[ty+1][NTHREADS2D+1] = sqrtf( (Gx*Gx) + (Gy*Gy) );	

				}
				else if(d==1){

					Gx = 
						 (1.0*tile[NTHREADS2D+1][tx+1] +  2.0*tile[NTHREADS2D+1][tx+2] +  (-2.0)*tile[NTHREADS2D+1][tx+4] + (-1.0)*tile[NTHREADS2D+1][tx+5]
						+ 4.0*tile[NTHREADS2D+2][tx+1] +  8.0*tile[NTHREADS2D+2][tx+2] +  (-8.0)*tile[NTHREADS2D+2][tx+4] + (-4.0)*tile[NTHREADS2D+2][tx+5]
						+ 6.0*tile[NTHREADS2D+3][tx+1] + 12.0*tile[NTHREADS2D+3][tx+2] + (-12.0)*tile[NTHREADS2D+3][tx+4] + (-6.0)*tile[NTHREADS2D+3][tx+5]
						+ 4.0*tile[NTHREADS2D+4][tx+1] +  8.0*tile[NTHREADS2D+4][tx+2] +  (-8.0)*tile[NTHREADS2D+4][tx+4] + (-4.0)*tile[NTHREADS2D+4][tx+5]
						+ 1.0*tile[NTHREADS2D+5][tx+1] +  2.0*tile[NTHREADS2D+5][tx+2] +  (-2.0)*tile[NTHREADS2D+5][tx+4] + (-1.0)*tile[NTHREADS2D+5][tx+5]);


					Gy = 
						 ((-1.0)*tile[NTHREADS2D+1][tx+1] + (-4.0)*tile[NTHREADS2D+1][tx+2] +  (-6.0)*tile[NTHREADS2D+1][tx+3] + (-4.0)*tile[NTHREADS2D+1][tx+4] + 
						(-1.0)*tile[NTHREADS2D+1][tx+5]
						+ (-2.0)*tile[NTHREADS2D+2][tx+1] + (-8.0)*tile[NTHREADS2D+2][tx+2] + (-12.0)*tile[NTHREADS2D+2][tx+3] + (-8.0)*tile[NTHREADS2D+2][tx+4] + 
						(-2.0)*tile[NTHREADS2D+2][tx+5]
						+    2.0*tile[NTHREADS2D+4][tx+1] +    8.0*tile[NTHREADS2D+4][tx+2] +    12.0*tile[NTHREADS2D+4][tx+3] +    8.0*tile[NTHREADS2D+4][tx+4] +    
						2.0*tile[NTHREADS2D+4][tx+5]
						+    1.0*tile[NTHREADS2D+5][tx+1] +    4.0*tile[NTHREADS2D+5][tx+2] +     6.0*tile[NTHREADS2D+5][tx+3] +    4.0*tile[NTHREADS2D+5][tx+4] +    
						1.0*tile[NTHREADS2D+5][tx+5]);

		
					G_tile[NTHREADS2D+1][tx+1] = sqrtf( (Gx*Gx) + (Gy*Gy) );

				}

				__syncthreads();

				

				if(phi == 0){
					if(G_tile[ty+1][tx+1]>G_tile[ty+1][tx+2] && G_tile[ty+1][tx+1]>G_tile[ty+1][tx]) //edge is in N-S
						pedge = 1;

				} else if(phi == 45) {
					if(G_tile[ty+1][tx+1]>G_tile[ty+2][tx+2] && G_tile[ty+1][tx+1]>G_tile[ty][tx]) // edge is in NW-SE
						pedge = 1;

				} else if(phi == 90) {
					if(G_tile[ty+1][tx+1]>G_tile[ty+2][tx+1] && G_tile[ty+1][tx+1]>G_tile[ty][tx+1]) //edge is in E-W
						pedge = 1;

				} else if(phi == 135) {
					if(G_tile[ty+1][tx+1]>G_tile[ty+2][tx] && G_tile[ty+1][tx+1]>G_tile[ty][tx+2]) // edge is in NE-SW
						pedge = 1;
				}
							
				
				if(G_tile[ty+1][tx+1]>hithres && pedge)
					image_out[i*width+j] = 255;
				else if(pedge && G_tile[ty+1][tx+1]>=lowthres && G_tile[ty+1][tx+1]<hithres){
					// check neighbours 3x3
					goOut = 0;
					for (ii=0;goOut==0 && ii<=2 ; ii++)
						for (jj=0;goOut==0 && jj<=2 ; jj++)
							if (G_tile[ty+ii][tx+jj]>hithres){
								image_out[i*width+j] = 255;
								goOut = 1;						
							}
												
				}
						
			
		}
	
}//Function


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
	float *IM_IN ,*IM_OUT,*NR;
	
	/* Mallocs GPU */
	cudaMalloc((void**)&IM_IN, sizeof(float)*height*width);
	cudaMalloc((void**)&IM_OUT, sizeof(float)*height*width);
	cudaMalloc((void**)&NR, sizeof(float)*height*width);

	
	/* CPU->GPU */
	cudaMemcpy(IM_IN, im, sizeof(float)*height*width, cudaMemcpyHostToDevice);

	dim3 dimBlock(NTHREADS2D,NTHREADS2D);

	int blocksX,blocksY;
	blocksX = ( height % NTHREADS2D == 0 ) ? height/NTHREADS2D : height/NTHREADS2D + 1 ;
	blocksY = ( width % NTHREADS2D == 0 ) ? width/NTHREADS2D : width/NTHREADS2D + 1 ;
	dim3 dimGrid(blocksY,blocksX,1);

	printf("BlocksX: %i \n",blocksX);
	printf("BlocksY: %i \n",blocksY);
	
	double t0 = get_time2();
	noiseReduction<<<dimGrid,dimBlock>>>(IM_IN,NR, height, width,blocksY,blocksX);
	cudaThreadSynchronize();
	double t1 = get_time2();
	printf("NR=%f .s\n", t1-t0);

	t0 = get_time2();

	imageGradientHysteresisThresholding<<<dimGrid,dimBlock>>>(NR, IM_OUT,level/2, 2*(level), height, width,blocksY,blocksX);
	cudaThreadSynchronize();
	t1 = get_time2();
	printf("IG/HYS=%f .s\n", t1-t0);	
		
	/* GPU->CPU */
	cudaMemcpy(image_out, IM_OUT, sizeof(float)*height*width, cudaMemcpyDeviceToHost);
	

	// Free device memory
	cudaFree(IM_IN);
	cudaFree(IM_OUT);
	cudaFree(NR);
}
