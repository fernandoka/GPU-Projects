#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <sys/resource.h>

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}


typedef struct { float m, x, y, z, vx, vy, vz; } body;

void randomizeBodies(body *data, int n) {
	for (int i = 0; i < n; i++) {
		data[i].m  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].x  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].y  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].z  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

void randomizeBodies2(float *m, float*x, float *y, float *z,float *vx,float *vy, float *vz ,int n) {
	for (int i = 0; i < n; i++) {
		m[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		x[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		y[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		z[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}


void bodyForce(body *p, float dt, int n) {

	for (int i = 0; i < n; i++) { 
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = 0; j < n; j++) {
			if (i!=j) {
				float dx = p[j].x - p[i].x;
				float dy = p[j].y - p[i].y;
				float dz = p[j].z - p[i].z;
				float distSqr = dx*dx + dy*dy + dz*dz;
				float invDist = 1.0f / sqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist;

				float G = 6.674e-11;
				float g_masses = G * p[j].m * p[i].m;

				Fx += g_masses * dx * invDist3; 
				Fy += g_masses * dy * invDist3; 
				Fz += g_masses * dz * invDist3;
			}
		}

		p[i].vx += dt*Fx/p[i].m; p[i].vy += dt*Fy/p[i].m; p[i].vz += dt*Fz/p[i].m;
	}
}

void integrate(body *p, float dt, int n){
	for (int i = 0 ; i < n; i++) {
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		p[i].z += p[i].vz*dt;
	}
}

void nbodies(int nBodies)
{
	const float dt = 0.01f; // time step
	const int nIters = 100;  // simulation iterations

	body *p = (body*)malloc(nBodies*sizeof(body));

	randomizeBodies(p, nBodies); // Init pos / vel data

	double t0 = get_time();

	for (int iter = 1; iter <= nIters; iter++) {
		bodyForce(p, dt, nBodies); // compute interbody forces
		integrate(p, dt, nBodies); // integrate position
	}

	double totalTime = get_time()-t0; 
	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, 1e-6 * nBodies * nBodies / totalTime);

	free(p);
}

void main_nbodiesOCL(int nBodies)
{
	const float dt = 0.01f; // time step
	const int nIters = 100;  // simulation iterations

	float *m, *x, *y, *z, *vx, *vy, *vz;
	m=(float*) malloc(sizeof(float)*nBodies);
	
	x=(float*) malloc(sizeof(float)*nBodies);
	y=(float*) malloc(sizeof(float)*nBodies);
	z=(float*) malloc(sizeof(float)*nBodies); 
	
	vx=(float*) malloc(sizeof(float)*nBodies); 
	vy=(float*) malloc(sizeof(float)*nBodies);
	vz=(float*) malloc(sizeof(float)*nBodies);

	 // Init pos / vel data

	randomizeBodies2( m, x, y, z, vx, vy, vz, nBodies);

	double t0 = get_time();
	
	nbodiesOCL(m, x, y, z, vx, vy, vz, nBodies,dt,nIters);

	double totalTime = get_time()-t0; 
	printf("%d Bodies with %d iterations: %0.3f Millions Interactions/second\n", nBodies, nIters, 1e-6 * nBodies * nBodies / totalTime);

	free(m); free(x); free(y); 
	free(z); free(vx); free(vy); 
	free(vz);
}




int main(const int argc, const char** argv) {

	int nBodies = 1000;
	double t0, t1;

	if (argc == 3) 
		nBodies = atoi(argv[1]);
	else {

		printf("./exec nbodies [c,g] \n");
		return(-1);
	}


	switch (argv[2][0]) {
		case 'c':
			t0 = get_time();
			nbodies(nBodies);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
			t0 = get_time();
			main_nbodiesOCL(nBodies);
			t1 = get_time();
			printf("OCL Exection time %f ms.\n", t1-t0);
			break;
		default:
			printf("Not Implemented yet!!\n");


	}
	return(1);
}
