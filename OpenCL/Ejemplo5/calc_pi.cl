// calc_pi.cl
// Kernel source file for calculate pi


__kernel void calc_pi(__global float * area,const uint n){
	
	int i = get_global_id(0);
	float x;

	x = ((i+1)+0.5)/n;
	area[i] = (4.0/(1.0 + x*x))/n;

}