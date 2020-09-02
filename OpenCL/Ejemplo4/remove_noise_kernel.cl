// reduce_noise_kernel.cl
// Kernel source file for reduce_noise of a image
//#define WORK_SIZE 32
#define WINDOW_SIZE 3

__kernel
void reduce_noise(__global float * output,__global float * input,const uint height,
		const uint width,const float thredshold)
{
	
	int i = get_global_id(1);
	int j = get_global_id(0);
	
	float window[WINDOW_SIZE*WINDOW_SIZE];
	float tmp,median,currentPixel;
	int ii,jj;
	
	currentPixel = output[i*width+j] = input[i*width+j];

	if( i>0 && j>0 && i < height-1 && j < width-1 ){

		
		
		for (ii =-1; ii<=1; ii++){
				for (jj =-1; jj<=1; jj++){
					window[(ii+1)*WINDOW_SIZE+jj+1] = input[(i+ii)*width+(j+jj)];
				}
		}


		for (ii=1; ii<WINDOW_SIZE*WINDOW_SIZE; ii++){
			for (jj=0 ; jj<WINDOW_SIZE*WINDOW_SIZE-ii; jj++){
				if (window[jj] > window[jj+1]){
					tmp = window[jj];
					window[jj] = window[jj+1];
					window[jj+1] = tmp;
				}
			}
		}
		

		median = window[(WINDOW_SIZE*WINDOW_SIZE-1)/2];
		tmp = fabs( (median-currentPixel)/median );
		if( tmp > thredshold)
			output[i*width + j] = median;
		
	}

}
