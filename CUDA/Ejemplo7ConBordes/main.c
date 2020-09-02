#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "kernel.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

void cannyCPU(float *im, float *image_out, 
	float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge,
	float level,
	int height, int width);

double get_time(){
	static struct timeval 	tv0;
	double time_, mytime;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	mytime = time_/1000000;
	return(mytime);
}

unsigned char *readBMP(char *file_name, char header[54], int *w, int *h)
{
	//Se abre el fichero en modo binario para lectura
	FILE *f=fopen(file_name, "rb");
	if (!f){
		perror(file_name); exit(1);
	}

	// Cabecera archivo imagen
	//***********************************
	// Devuelve cantidad de bytes leidos
	int n=fread(header, 1, 54, f);

	//Si no lee 54 bytes es que la imagen de entrada es demasiado pequeña
	if (n!=54)
		fprintf(stderr, "Entrada muy pequeña (%d bytes)\n", n), exit(1);

	//Si los dos primeros bytes no corresponden con los caracteres BM no es un fichero BMP
	if (header[0]!='B'|| header[1]!='M')
		fprintf(stderr, "No BMP\n"), exit(1);

	//El tamaño de la imagen es el valor de la posicion 2 de la cabecera menos 54 bytes que ocupa esa cabecera
	int imagesize=*(int*)(header+2)-54;

	//Si la imagen tiene tamaño negativo o es superior a 48MB la imagen se rechaza
	if (imagesize<=0|| imagesize > 0x3000000)
		fprintf(stderr, "Imagen muy grande: %d bytes\n", imagesize), exit(1);

	//Si la cabecera no tiene el tamaño de 54 o el numero de bits por pixel es distinto de 24 la imagen se rechaza
	if (*(int*)(header+10)!=54|| *(short*)(header+28)!=24)
		fprintf(stderr, "No color 24-bit\n"), exit(1);
	
	//Cuando la posicion 30 del header no es 0, es que existe compresion por lo que la imagen no es valida
	if (*(int*)(header+30)!=0)
		fprintf(stderr, "Compresion no suportada\n"), exit(1);
	
	//Se recupera la altura y anchura de la cabecera
	int width=*(int*)(header+18);
	int height=*(int*)(header+22);
	//**************************************


	// Lectura de la imagen
	//*************************************
	unsigned char *image = (unsigned char*)malloc(imagesize+256+width*6); //Se reservan "imagesize+256+width*6" bytes y se devuelve un puntero a estos datos

	unsigned char *tmp;
	image+=128+width*3;
	if ((n=fread(image, 1, imagesize+1, f))!=imagesize)
		fprintf(stderr, "File size incorrect: %d bytes read insted of %d\n", n, imagesize), exit(1);

	fclose(f);
	printf("Image read correctly (width=%i height=%i, imagesize=%i).\n", width, height, imagesize);

	/* Output variables */
	*w = width;
	*h = height;

	return(image);
}

void writeBMP(float *imageFLOAT, char *file_name, char header[54], int width, int height)
{

	FILE *f;
	int i, n;

	int imagesize=*(int*)(header+2)-54;

	unsigned char *image = (unsigned char*)malloc(3*sizeof(unsigned char)*width*height);

	for (i=width*height-1;i>=0;i--){
		image[3*i+2] = imageFLOAT[i]; //B
		image[3*i+1] = imageFLOAT[i]; //G
		image[3*i]   = imageFLOAT[i]; //R 
	}
	

	f=fopen(file_name, "wb");		//Se abre el fichero en modo binario de escritura
	if (!f){
		perror(file_name); 
		exit(1);
	}

	n=fwrite(header, 1, 54, f);		//Primeramente se escribe la cabecera de la imagen
	n+=fwrite(image, 1, imagesize, f);	//Y despues se escribe el resto de la imagen
	if (n!=54+imagesize)			//Si se han escrito diferente cantidad de bytes que la suma de la cabecera y el tamaño de la imagen. Ha habido error
		fprintf(stderr, "Escritos %d de %d bytes\n", n, imagesize+54);
	fclose(f);

	free(image);

}


float *RGB2BW(unsigned char *imageUCHAR, int width, int height)
{
	int i, j;
	float *imageBW = (float *)malloc(sizeof(float)*width*height);

	unsigned char R, B, G;

	for (i=0; i<height; i++)
		for (j=0; j<width; j++)
		{
			R = imageUCHAR[3*(i*width+j)];
			G = imageUCHAR[3*(i*width+j)+1];
			B = imageUCHAR[3*(i*width+j)+2];

			imageBW[i*width+j] = 0.2989 * R + 0.5870 * G + 0.1140 * B;
		}

	return(imageBW);
}

void cannyCPU(float *im, float *image_out, 
	float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge,
	float level,
	int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}


	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}

	// Edge
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);

	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
		}
}

void freeMemory(unsigned char *imageUCHAR, float *imageBW, float *NR, float *G, float *phi, float *Gx, float *Gy, int *pedge, float *imageOUT)
{
	//free(imageUCHAR);
	free(imageBW);
	free(imageOUT);
	free(NR);
	free(G);
	free(phi);
	free(Gx);
	free(Gy);
	free(pedge);
}	


int main(int argc, char **argv) {

	int width, height;
	unsigned char *imageUCHAR;
	float *imageBW;

	char header[54];


	//Variables para calcular el tiempo
	double t0, t1;
	double cpu_time_used = 0.0;

	//Tener menos de 3 argumentos es incorrecto
	if (argc < 4) {
		fprintf(stderr, "Uso incorrecto de los parametros. exe  input.bmp output.bmp [cg]\n");
		exit(1);
	}


	// READ IMAGE & Convert image
	imageUCHAR = readBMP(argv[1], header, &width, &height);
	imageBW = RGB2BW(imageUCHAR, width, height);


	// Aux. memory
	float *NR       = (float *)malloc(sizeof(float)*width*height);
	float *G        = (float *)malloc(sizeof(float)*width*height);
	float *phi      = (float *)malloc(sizeof(float)*width*height);
	float *Gx       = (float *)malloc(sizeof(float)*width*height);
	float *Gy       = (float *)malloc(sizeof(float)*width*height);
	int *pedge      = (int *)malloc(sizeof(int)*width*height);
	float *imageOUT = (float *)malloc(sizeof(float)*width*height);

	////////////////
	// CANNY      //
	////////////////
	switch (argv[3][0]) {
		case 'c':
			t0 = get_time();
			cannyCPU(imageBW, imageOUT, NR, G, phi, Gx, Gy, pedge,
				1000.0, height, width);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
			t0 = get_time();
			cannyGPU(imageBW, imageOUT,
				1000.0, height, width);
			t1 = get_time();
			printf("GPU Exection time %f ms.\n", t1-t0);
			break;
		default:
			printf("Not Implemented yet!!\n");


	}

	// WRITE IMAGE
	writeBMP(imageOUT, argv[2], header, width, height);

	freeMemory(imageUCHAR, imageBW, NR, G, phi, Gx, Gy, pedge, imageOUT);	

	return(0);
}

