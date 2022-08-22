/*
GALAMOST - GPU-Accelerated Large-Scale Molecular Simulation Toolkit
COPYRIGHT
	GALAMOST Copyright (c) (2013) The group of Prof. Zhong-Yuan Lu
LICENSE
	This program is a free software: you can redistribute it and/or 
	modify it under the terms of the GNU General Public License. 
	This program is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANT ABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
	See the General Public License v3 for more details.
	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
DISCLAIMER
	The authors of GALAMOST do not guarantee that this program and its 
	derivatives are free from error. In no event shall the copyright 
	holder or contributors be liable for any indirect, incidental, 
	special, exemplary, or consequential loss or damage that results 
	from its use. We also have no responsibility for providing the 
	service of functional extension of this program to general users.
USER OBLIGATION 
	If any results obtained with GALAMOST are published in the scientific 
	literature, the users have an obligation to distribute this program 
	and acknowledge our efforts by citing the paper "Y.-L. Zhu, H. Liu, 
	Z.-W. Li, H.-J. Qian, G. Milano, and Z.-Y. Lu, J. Comput. Chem. 2013,
	34, 2197-2211" in their article.
CORRESPONDENCE
	State Key Laboratory of Polymer Physics and Chemistry,
	Changchun Institute of Applied Chemistry, Chinese Academy of Sciences, China, 
	Dr. You-Liang Zhu, 
	Email: youliangzhu@ciac.ac.cn
*/
//	Maintainer: You-Liang Zhu
#include "BdNvt.cuh"
extern "C" __global__ 
void gpu_bd_nvt_first_step_kernel(Real4* d_pos,
							 Real4* d_vel,
							 Real4* d_force,
							 int3* d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             BoxSize box,
                             Real dt,
							 Real dtsq)
    {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < group_size)
        {
        unsigned int idx = d_group_members[i];
        Real4 pos = d_pos[idx];
        
        Real px = pos.x;
        Real py = pos.y;
        Real pz = pos.z;
        Real pw = pos.w;
        
        Real4 vel = d_vel[idx];
		Real4 accel = d_force[idx];
        Real mass = vel.w;
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        Real dx = vel.x * dt + 0.5f * accel.x * dtsq;
        Real dy = vel.y * dt + 0.5f * accel.y * dtsq;
        Real dz = vel.z * dt + 0.5f * accel.z * dtsq;
        
        px += dx;
        py += dy;
        pz += dz;

        vel.x += 0.5f * accel.x * dt;
        vel.y += 0.5f * accel.y * dt;
        vel.z += 0.5f * accel.z * dt;

        int3 image = d_image[idx];
		box.wrap(px, py, pz, image);
        
        Real4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;

        d_pos[idx] = pos2;
        d_vel[idx] = vel;
        d_image[idx] = image;
        }
    }
   

cudaError_t gpu_bd_nvt_first_step(Real4* d_pos,
							 Real4* d_vel,
							 Real4* d_force,
							 int3* d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxSize& box,
                             unsigned int block_size,
                             Real dt)
    {
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

	Real dtsq = dt*dt;
    gpu_bd_nvt_first_step_kernel<<< grid, threads, block_size * sizeof(Real) >>>(d_pos,
																			 d_vel,	
																			 d_force,
																			 d_image,
                                                                             d_group_members,
                                                                             group_size,
                                                                             box,
                                                                             dt,
																			 dtsq);
    return cudaSuccess;
    }


extern "C" __global__ 
void gpu_bd_nvt_second_step_kernel(Real4* d_pos,
								Real4* d_vel,
								Real4* d_force,
								unsigned int *d_group_members,
								unsigned int group_size,
								unsigned int seed,
								Real *d_params,
								Real T,
								Real D,
								Real dt,
								Real dtInv)
 
 {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < group_size)
        {
        unsigned int idx = d_group_members[i];
        Real4 vel = d_vel[idx];
      
        int typ = __real_as_int(d_pos[idx].w);
        Real gamma = d_params[typ];
        Real coeff = sqrt_gala(Real(6.0) * gamma * T * dtInv);       

        Real3 bd_force = ToReal3(0.0, 0.0, 0.0);
        
	    SaruGPU RNG(seed, idx); 
	
        Real randomx = Rondom(-1.0, 1.0);
        Real randomy = Rondom(-1.0, 1.0);
        Real randomz = Rondom(-1.0, 1.0);
        
        bd_force.x = randomx*coeff - gamma*vel.x;
        bd_force.y = randomy*coeff - gamma*vel.y;
        if (D > Real(2.0))
        bd_force.z = randomz*coeff - gamma*vel.z;
        
        Real4 force = d_force[idx];
        Real mass = vel.w;
		
        Real minv = Real(1.0) / mass;
        force.x += bd_force.x;
        force.y += bd_force.y;
        force.z += bd_force.z;

        
        vel.x += Real(0.5) * force.x * minv * dt;
        vel.y += Real(0.5) * force.y * minv * dt;
        vel.z += Real(0.5) * force.z * minv * dt;
        
        d_vel[idx] = vel;
		d_force[idx] = force;
        }

    }

cudaError_t gpu_bd_nvt_second_step(Real4* d_pos,
								Real4* d_vel,
								Real4* d_force, 
								unsigned int *d_group_members,
								unsigned int group_size,
								unsigned int seed,
								unsigned int block_size,
								Real *d_params,
								Real T,
								Real D,
								Real dt)
    {
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
	Real dtInv;
	if(dt<1.0e-7)
		dtInv = 0.0;
	else
		dtInv = 1.0/dt;
		
    gpu_bd_nvt_second_step_kernel<<< grid, threads >>>(d_pos,
												d_vel,	
												d_force,
												d_group_members, 
												group_size,
												seed, 
												d_params,
												T,
												D,
												dt,
												dtInv);
    
    return cudaSuccess;
    }



