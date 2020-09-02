#include "my_ocl.h"

#define WORK_SIZE 32

/* From common.c */
extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);


int remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	
	// OpenCL host variables
	cl_int err;
	cl_uint numPlatforms;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[2];
	size_t local[2];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	//General use variables
	int i;
	float *d_image_out,*d_im;

	// read the kernel
	fp = fopen("remove_noise_kernel.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = (char*)malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	close(fp);

	if(readlen!= filelen){
		printf("error reading file\n");
		return 1;
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';

	// Set up platform and GPU device
	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0){
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0){
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Secure a device
	for (i = 0; i < numPlatforms; i++){
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)break;
	}

	if (device_id == NULL){
		printf("Error: Failed to create a device group!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	err = output_device_info(device_id);
	if (err != CL_SUCCESS ){
		printf("Failed while printing the device info\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context){
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// Create a command queue step one
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands){
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// create command queue step two
	command_queue = clCreateCommandQueue(context,device_id, 0, &err);
	if (err != CL_SUCCESS){	
		printf("Unable to create command queue. Error Code=%d\n",err);
		return 1;
	}

	// create program object from source. 
	// kernel_src contains source read from file earlier, line 38
	program = clCreateProgramWithSource(context, 1 ,(const char **)&kernel_src, NULL, &err);
	if (err != CL_SUCCESS){	
		printf("Unable to create program object. Error Code=%d\n",err);
		return 1;
	}     

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS){
        	printf("Build failed. Error Code=%s\n", err_code(err));

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		return 1;
	}
	
	kernel = clCreateKernel(program, "reduce_noise", &err);
	if (err != CL_SUCCESS){	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		return 1;
	}

	// create buffer objects to input and output args of kernel function
	d_im = (float*)clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * height*width, NULL, NULL);

	d_image_out = (float*)clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * height*width, NULL, NULL);

	
	if(!d_im || !d_image_out){
	        printf("Error: Failed to allocate device memory!\n");
		return 1;
	}
	
	//fill the d_im variable with the image data
	err = clEnqueueWriteBuffer(commands, d_im, CL_TRUE, 0, sizeof(float) * height*width, im, 0, NULL, NULL);
	if (err != CL_SUCCESS){
		printf("Error: Failed to write array1D to source array!\n%s\n", err_code(err));
		return 1;
	}

	// Set the arguments to our compute kernel
	//window_size=(window_size-1)>>1;
	err  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_image_out);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_im);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &height);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_float), &thredshold);
	if (err != CL_SUCCESS){
		printf("Error: Failed to set kernel arguments!\n%s\n", err_code(err));
		return 1;
	}

	// set the global work dimension size
	global[0] = height;
	global[1] = width;
	//local[1] = local[0] = WORK_SIZE;
	
	// Enqueue the kernel object with 
	// Dimension size = 2, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	//Launch kernel!!
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,global, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS){	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		return 1;
	}

	// wait for the command to finish
	clFinish(command_queue);
	
	// read the output back to host memory
	err = clEnqueueReadBuffer( commands, d_image_out, CL_TRUE, 0, sizeof(float) * height*width, image_out, 0, NULL, NULL );
	if (err != CL_SUCCESS){	
		printf("Error enqueuing read buffer command. Error Code=%s\n",err_code(err));
		return 1;
	}

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	return 0;
}
