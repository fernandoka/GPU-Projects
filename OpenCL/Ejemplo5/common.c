#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <sys/resource.h>

char *err_code (cl_int err_in)
{
    switch (err_in) {

        case CL_SUCCESS :
            return (char*)" CL_SUCCESS ";
        case CL_DEVICE_NOT_FOUND :
            return (char*)" CL_DEVICE_NOT_FOUND ";
        case CL_DEVICE_NOT_AVAILABLE :
            return (char*)" CL_DEVICE_NOT_AVAILABLE ";
        case CL_COMPILER_NOT_AVAILABLE :
            return (char*)" CL_COMPILER_NOT_AVAILABLE ";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE :
            return (char*)" CL_MEM_OBJECT_ALLOCATION_FAILURE ";
        case CL_OUT_OF_RESOURCES :
            return (char*)" CL_OUT_OF_RESOURCES ";
        case CL_OUT_OF_HOST_MEMORY :
            return (char*)" CL_OUT_OF_HOST_MEMORY ";
        case CL_PROFILING_INFO_NOT_AVAILABLE :
            return (char*)" CL_PROFILING_INFO_NOT_AVAILABLE ";
        case CL_MEM_COPY_OVERLAP :
            return (char*)" CL_MEM_COPY_OVERLAP ";
        case CL_IMAGE_FORMAT_MISMATCH :
            return (char*)" CL_IMAGE_FORMAT_MISMATCH ";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED :
            return (char*)" CL_IMAGE_FORMAT_NOT_SUPPORTED ";
        case CL_BUILD_PROGRAM_FAILURE :
            return (char*)" CL_BUILD_PROGRAM_FAILURE ";
        case CL_MAP_FAILURE :
            return (char*)" CL_MAP_FAILURE ";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET :
            return (char*)" CL_MISALIGNED_SUB_BUFFER_OFFSET ";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
            return (char*)" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ";
        case CL_INVALID_VALUE :
            return (char*)" CL_INVALID_VALUE ";
        case CL_INVALID_DEVICE_TYPE :
            return (char*)" CL_INVALID_DEVICE_TYPE ";
        case CL_INVALID_PLATFORM :
            return (char*)" CL_INVALID_PLATFORM ";
        case CL_INVALID_DEVICE :
            return (char*)" CL_INVALID_DEVICE ";
        case CL_INVALID_CONTEXT :
            return (char*)" CL_INVALID_CONTEXT ";
        case CL_INVALID_QUEUE_PROPERTIES :
            return (char*)" CL_INVALID_QUEUE_PROPERTIES ";
        case CL_INVALID_COMMAND_QUEUE :
            return (char*)" CL_INVALID_COMMAND_QUEUE ";
        case CL_INVALID_HOST_PTR :
            return (char*)" CL_INVALID_HOST_PTR ";
        case CL_INVALID_MEM_OBJECT :
            return (char*)" CL_INVALID_MEM_OBJECT ";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
            return (char*)" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
        case CL_INVALID_IMAGE_SIZE :
            return (char*)" CL_INVALID_IMAGE_SIZE ";
        case CL_INVALID_SAMPLER :
            return (char*)" CL_INVALID_SAMPLER ";
        case CL_INVALID_BINARY :
            return (char*)" CL_INVALID_BINARY ";
        case CL_INVALID_BUILD_OPTIONS :
            return (char*)" CL_INVALID_BUILD_OPTIONS ";
        case CL_INVALID_PROGRAM :
            return (char*)" CL_INVALID_PROGRAM ";
        case CL_INVALID_PROGRAM_EXECUTABLE :
            return (char*)" CL_INVALID_PROGRAM_EXECUTABLE ";
        case CL_INVALID_KERNEL_NAME :
            return (char*)" CL_INVALID_KERNEL_NAME ";
        case CL_INVALID_KERNEL_DEFINITION :
            return (char*)" CL_INVALID_KERNEL_DEFINITION ";
        case CL_INVALID_KERNEL :
            return (char*)" CL_INVALID_KERNEL ";
        case CL_INVALID_ARG_INDEX :
            return (char*)" CL_INVALID_ARG_INDEX ";
        case CL_INVALID_ARG_VALUE :
            return (char*)" CL_INVALID_ARG_VALUE ";
        case CL_INVALID_ARG_SIZE :
            return (char*)" CL_INVALID_ARG_SIZE ";
        case CL_INVALID_KERNEL_ARGS :
            return (char*)" CL_INVALID_KERNEL_ARGS ";
        case CL_INVALID_WORK_DIMENSION :
            return (char*)" CL_INVALID_WORK_DIMENSION ";
        case CL_INVALID_WORK_GROUP_SIZE :
            return (char*)" CL_INVALID_WORK_GROUP_SIZE ";
        case CL_INVALID_WORK_ITEM_SIZE :
            return (char*)" CL_INVALID_WORK_ITEM_SIZE ";
        case CL_INVALID_GLOBAL_OFFSET :
            return (char*)" CL_INVALID_GLOBAL_OFFSET ";
        case CL_INVALID_EVENT_WAIT_LIST :
            return (char*)" CL_INVALID_EVENT_WAIT_LIST ";
        case CL_INVALID_EVENT :
            return (char*)" CL_INVALID_EVENT ";
        case CL_INVALID_OPERATION :
            return (char*)" CL_INVALID_OPERATION ";
        case CL_INVALID_GL_OBJECT :
            return (char*)" CL_INVALID_GL_OBJECT ";
        case CL_INVALID_BUFFER_SIZE :
            return (char*)" CL_INVALID_BUFFER_SIZE ";
        case CL_INVALID_MIP_LEVEL :
            return (char*)" CL_INVALID_MIP_LEVEL ";
        case CL_INVALID_GLOBAL_WORK_SIZE :
            return (char*)" CL_INVALID_GLOBAL_WORK_SIZE ";
        case CL_INVALID_PROPERTY :
            return (char*)" CL_INVALID_PROPERTY ";
        default:
            return (char*)"UNKNOWN ERROR";

    }
}

int output_device_info(cl_device_id device_id)

{

    int err;                            // error code returned from OpenCL calls

    cl_device_type device_type;         // Parameter defining the type of the compute device
    cl_uint comp_units;                 // the max number of compute units on a device
    cl_char vendor_name[1024] = {0};    // string to hold vendor name for compute device
    cl_char device_name[1024] = {0};    // string to hold name of compute device

#ifdef VERBOSE
    cl_uint          max_work_itm_dims;
    size_t           max_wrkgrp_size;
    size_t          *max_loc_size;
#endif





    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device name!\n");
        return EXIT_FAILURE;
    }
    printf(" \n Device is  %s ",device_name);


    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    if(device_type  == CL_DEVICE_TYPE_GPU)
       printf(" GPU from ");
    else if (device_type == CL_DEVICE_TYPE_CPU)
       printf("\n CPU from ");
    else 
       printf("\n non  CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device vendor name!\n");
        return EXIT_FAILURE;
    }
    printf(" %s ",vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }
    printf(" with a max of %d compute units \n",comp_units);


#ifdef VERBOSE
//
// Optionally print information about work group sizes
//
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                               &max_work_itm_dims, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!\n",
                                                                            err_code(err));
        return EXIT_FAILURE;
    }
    
    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if(max_loc_size == NULL){
       printf(" malloc failed\n");
       return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t), 
                               max_loc_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!\n",err_code(err));
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                               &max_wrkgrp_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)!\n",err_code(err));
        return EXIT_FAILURE;
    }
   printf("work group, work item information");
   printf("\n max loc dim ");
   for(int i=0; i< max_work_itm_dims; i++)
     printf(" %d ",(int)(*(max_loc_size+i)));
   printf("\n");
   printf(" Max work group size = %d\n",(int)max_wrkgrp_size);
#endif
    return CL_SUCCESS;
}




