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

#include "IntegMethod.h"
#include "BdNvt.cuh"

#ifndef __BD_NVT_H__
#define __BD_NVT_H__


class BdNvt : public IntegMethod
    {
    public:

        BdNvt(boost::shared_ptr<AllInfo> all_info,
					  boost::shared_ptr<ParticleSet> group,					  
					  Real T,
                      unsigned int seed);
        virtual ~BdNvt() {};
		
		void setGamma(Real gamma);
  		
		void setGamma(const std::string &name, Real gamma);	
  

        virtual void firstStep(unsigned int timestep);

        virtual void secondStep(unsigned int timestep);

    protected:
        unsigned int m_seed;		
		boost::shared_ptr<Array<Real> > m_params;
		unsigned int m_nkinds;
    };


void export_BdNvt();

#endif 

