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
#include "Info.cuh"
#include "saruprngCUDA.h"
#ifndef __BDNVT_CUH__
#define __BDNVT_CUH__


cudaError_t gpu_bd_nvt_first_step(Real4* d_pos,
							 Real4* d_vel,
							 Real4* d_force,
							 int3* d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxSize& box,
                             unsigned int block_size,
                             Real dt);



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
								Real dt);

#endif 

