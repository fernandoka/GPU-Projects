// nbodies.cl

__kernel
void bodyForce(__global float * m,__global float * x,__global float * y,
		__global float * z,__global float * vx,__global float * vy,
		__global float * vz, const uint nbodies,const float dt)
{
	
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	int ii,jj;
	float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
	float G = 6.674e-11;
	float g_masses,dx,dy,dz,distSqr,invDist=1,invDist3;
				
	if(i < nbodies && j < nbodies){
			
			if (i!=j) {
				 dx = x[j] - x[i];
				 dy = y[j] - y[i];
				 dz = z[j] - z[i];
				 distSqr = dx*dx + dy*dy + dz*dz;
				 invDist = 1.0f / sqrt(distSqr); //Esta linea me peta en el portatil
				 invDist3 = invDist * invDist * invDist;

				 g_masses = G * m[j] * m[i];

				Fx += g_masses * dx * invDist3; 
				Fy += g_masses * dy * invDist3; 
				Fz += g_masses * dz * invDist3;
			}

		vx[i] += (dt*Fx)/m[i]; 
		vy[i] += (dt*Fy)/m[i]; 
		vz[i] += (dt*Fz)/m[i];
	}
}


__kernel
void integrate(__global float * x,__global float * y,__global float * z,
	__global float * vx,__global float * vy,__global float * vz, const uint nbodies,const float dt)
{
	int i = get_global_id(0);
	
	if(i < nbodies){
		x[i] += vx[i]*dt;
		y[i] += vy[i]*dt;
		z[i] += vz[i]*dt;
	}	
}