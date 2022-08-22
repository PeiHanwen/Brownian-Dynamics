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


#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "BdNvt.h"

BdNvt::BdNvt(boost::shared_ptr<AllInfo> all_info,
					  boost::shared_ptr<ParticleSet> group,
					  Real T,
                      unsigned int seed)
    : IntegMethod(all_info,group),m_seed(seed)
    {
	 m_T = T;
	 m_nkinds = m_basic_info->getNKinds();	
	m_block_size= 288;
	m_params = boost::shared_ptr<Array<Real> >(new Array<Real>(m_nkinds*m_nkinds, location::host));	
	Real* h_params = m_params->getArray(location::host, access::write);

    for (unsigned int i = 0; i < m_nkinds; i++)
        h_params[i] = Real(1.0);

	m_ObjectName= "BdNvt";
	if (m_perf_conf->isRoot())
		cout << "INFO : "<<m_ObjectName<<" object has been created" << endl;	
    }



void BdNvt::setGamma(const std::string &name, Real gamma)
	{
	   unsigned int typ= m_basic_info -> switchNameToIndex(name);
	if (typ >= m_nkinds)
		{
		cerr << endl << "***Error! Trying to set BdNvt params for a non existant type! " << typ << endl << endl;
		throw runtime_error("BdNvt::setGamma argument error");
		}

	Real* h_params = m_params->getArray(location::host, access::readwrite);
	h_params[typ] = gamma;
	}       

void BdNvt::setGamma(Real gamma)
	{
	Real* h_params = m_params->getArray(location::host, access::readwrite);	
	for(unsigned int i =0; i<m_nkinds;i++)
			h_params[i] = gamma;	  

	} 
	

void BdNvt::firstStep(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

	Real4* d_pos = m_basic_info->getPos()->getArray(location::device, access::readwrite);
	Real4* d_vel = m_basic_info->getVel()->getArray(location::device, access::readwrite);	
	int3* d_image = m_basic_info->getImage()->getArray(location::device, access::readwrite);	
	Real4* d_force = m_basic_info->getForce()->getArray(location::device, access::read);
	const BoxSize& box = m_basic_info->getBox();

    gpu_bd_nvt_first_step(d_pos,
					 d_vel,	
					 d_force,
					 d_image,
                     m_group->getIdxGPUArray(),
                     group_size,
                     box,
                     m_block_size,
                     m_dt);


        CHECK_CUDA_ERROR();

    }
        

void BdNvt::secondStep(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
		
	if(m_setVariantT)
		m_T = (Real) m_vT->getValue(timestep);
		
	Real4* d_pos = m_basic_info->getPos()->getArray(location::device, access::read);	
	Real4* d_vel = m_basic_info->getVel()->getArray(location::device, access::readwrite);
	Real4* d_force = m_basic_info->getForce()->getArray(location::device, access::read);
	Real* d_params = m_params->getArray(location::device, access::read);
    const Real D = Real(m_all_info->getNDimensions());	

    gpu_bd_nvt_second_step(d_pos,
					 d_vel,
					 d_force,
                     m_group->getIdxGPUArray(), 
                     group_size,
					 m_seed+timestep,
                     m_block_size,
                     d_params,
					 m_T,
					 D,
                     m_dt);
    
        CHECK_CUDA_ERROR();
  

    }

void export_BdNvt()
    {
    class_<BdNvt, boost::shared_ptr<BdNvt>, bases<IntegMethod>, boost::noncopyable>
        ("BdNvt", init< boost::shared_ptr<AllInfo>,
                          boost::shared_ptr<ParticleSet>,
						  Real,
						  unsigned int
                          >())
		.def("setGamma", static_cast< void (BdNvt::*)(Real) >(&BdNvt::setGamma)) 
		.def("setGamma", static_cast< void (BdNvt::*)(const std::string &, Real)>(&BdNvt::setGamma))
        ;
    }



