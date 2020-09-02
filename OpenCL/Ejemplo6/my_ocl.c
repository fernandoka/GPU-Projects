#include <stdio.h>
#include "my_ocl.h"


int nbodiesOCL(float *m, float*x, float *y, float *z,
	float *vx,float *vy, float *vz ,int nBodies,float dt,const int nIters)
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
	float *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;

	// read the kernel
	fp = fopen("nbodies.cl","r");
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
        	printf("Build failed. Error Code=%d\n", err);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		return 1;
	}

	for (int iter = 1; iter <= nIters; iter++) {
	

		kernel = clCreateKernel(program, "bodyForce", &err);
		if (err != CL_SUCCESS){	
			printf("Unable to create kernel object. Error Code=%d\n",err);
			return 1;
		}

		// create buffer objects to input and output args of kernel function
		d_m = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		
		d_x = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		d_y = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		d_z = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		
		d_vx = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		d_vy = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		d_vz = (float*)clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * nBodies, NULL, NULL);
		
		if(!d_m || !d_x || !d_y || !d_z || !d_vx || !d_vy || !d_vz){
		    printf("Error: Failed to allocate device memory!\n");
			return 1;
		}

		//fill the d_im variable with the image data
		
		err = clEnqueueWriteBuffer(commands, d_m, CL_TRUE, 0, sizeof(float) * nBodies, m, 0, NULL, NULL);

		err |= clEnqueueWriteBuffer(commands, d_x, CL_TRUE, 0, sizeof(float) * nBodies, x, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(commands, d_y, CL_TRUE, 0, sizeof(float) * nBodies, y, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(commands, d_z, CL_TRUE, 0, sizeof(float) * nBodies, z, 0, NULL, NULL);

		err |= clEnqueueWriteBuffer(commands, d_vx, CL_TRUE, 0, sizeof(float) * nBodies, vx, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(commands, d_vy, CL_TRUE, 0, sizeof(float) * nBodies, vy, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(commands, d_vz, CL_TRUE, 0, sizeof(float) * nBodies, vz, 0, NULL, NULL);


		if (err != CL_SUCCESS){
			printf("Error: Failed to write array1D to source array!\n%s\n", err_code(err));
			return 1;
		}


		// Set the arguments to our compute kernel
		err  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_m);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_z);
		err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_vx);
		err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_vy);
		err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_vz);
		err |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &nBodies);		
		err |= clSetKernelArg(kernel, 8, sizeof(cl_float), &dt);			
		if (err != CL_SUCCESS){
			printf("Error: Failed to set kernel arguments!\n%s\n", err_code(err));
			return 1;
		}

		// set the global work dimension size
		global[0] = global[1] = nBodies;
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
		/*err = clEnqueueReadBuffer( commands, d_m, CL_TRUE, 0, sizeof(float) * nBodies, m, 0, NULL, NULL );

		err |= clEnqueueReadBuffer( commands, d_x, CL_TRUE, 0, sizeof(float) * nBodies, x, 0, NULL, NULL );
		err |= clEnqueueReadBuffer( commands, d_y, CL_TRUE, 0, sizeof(float) * nBodies, y, 0, NULL, NULL );
		err |= clEnqueueReadBuffer( commands, d_z, CL_TRUE, 0, sizeof(float) * nBodies, z, 0, NULL, NULL );
		
		err |= clEnqueueReadBuffer( commands, d_x, CL_TRUE, 0, sizeof(float) * nBodies, vx, 0, NULL, NULL );	
		err |= clEnqueueReadBuffer( commands, d_y, CL_TRUE, 0, sizeof(float) * nBodies, vy, 0, NULL, NULL );
		err |= clEnqueueReadBuffer( commands, d_z, CL_TRUE, 0, sizeof(float) * nBodies, vz, 0, NULL, NULL );
		*/
		if (err != CL_SUCCESS){	
			printf("Error enqueuing read buffer command. Error Code=%s\n",err_code(err));
			return 1;
		}

		kernel = clCreateKernel(program, "integrate", &err);
		if (err != CL_SUCCESS){	
			printf("Unable to create kernel object. Error Code=%d\n",err);
			return 1;
		}


			// Set the arguments to our compute kernel
		err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_vx);
		err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_vy);
		err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_vz);
		err |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &nBodies);		
		err |= clSetKernelArg(kernel, 7, sizeof(cl_float), &dt);			
		if (err != CL_SUCCESS){
			printf("Error: Failed to set kernel arguments!\n%s\n", err_code(err));
			return 1;
		}

		// set the global work dimension size
		global[0] = nBodies;
		//local[1] = local[0] = WORK_SIZE;
		
		// Enqueue the kernel object with 
		// Dimension size = 2, 
		// global worksize = global, 
		// local worksize = NULL - let OpenCL runtime determine
		// No event wait list
		//Launch kernel!!
		err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,global, NULL, 0, NULL, NULL);
		if (err != CL_SUCCESS){	
			printf("Unable to enqueue kernel command. Error Code=%d\n",err);
			return 1;
		}

		// wait for the command to finish
		clFinish(command_queue);

	}//For nIters

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	
	return 0;
}
