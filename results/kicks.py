# after testing here I implimented it into utils_BNS.py 
# I should find my external hard drive to save PSyLib/PSyLib


# -*- coding: utf-8 -*-
# Copyright (C) Charles Kimall and Michael Zevin (2018)
#
# This file is part of the progenitor package.
#
# progenitor is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# progenitor is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with progenitor.  If not, see <http://www.gnu.org/licenses/>.

__author__ = ['Michael Zevin <michael.zevin@ligo.org>', 'Chase Kimball <charles.kimball@ligo.org']
__credits__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['System']


import numpy as np
import pandas as pd

import astropy.units as units
import astropy.constants as constants

from scipy.integrate import ode
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import integrate
#import zams as zams

import matplotlib.pyplot as plt
class System:
    """
    Places system described by Mhe, Mcomp, Apre, epre and position r(R,galphi,galcosth) in galaxy model gal
    Applies SNkick Vkick and mass loss Mhe-Mns to obtain Apost, epost, and SN-imparted systemic velocity V
    
    """
    def __init__(self, Mcomp, Mhe, Apre, epre, Nkick=1000, Vkick=None, Mns=None, sys_flag=None, galphi=None, galcosth=None, omega=None, phi=None, costh=None,th_ma = None):
        """ 
        #Masses in Msun, Apre in Rsun, Vkick in km/s, R in kpc
        #galphi,galcosth,omega, phi, costh (position, initial velocity, and kick angles) sampled randomly, unless specified (>-1)
        #galphi, galcosth correspond to azimuthal and polar angles -- respectively --  in the galactic frame
        #phi, costh are defined in comments of SN:
        #   theta: angle between preSN He core velocity relative to Mcomp (i.e. the positive y axis) and the kick velocity
        #   phi: angle between Z axis and projection of kick onto X-Z plane
        #omega: angle between the galactic velocity corresponding to a circular orbit in the r-z plane and
        #the actual galactic velocity preSN corresponding to a circular orbit
        
        """
    
        # Convert inputs to SI


        self.sys_flag = sys_flag
        self.Nkick = Nkick

        if (Vkick is not None): self.Vkick = maxwell.rvs(loc=0, scale=Vkick, size=self.Nkick)*units.km.to(units.m)
            #self.Vkick = Vkick*units.km.to(units.m) # set a random v kick
#         else: self.Vkick = np.random.uniform(0,1000,self.Nkick)*units.km.to(units.m)
        else: self.Vkick = maxwell.rvs(loc=0, scale=265, size=self.Nkick)*units.km.to(units.m)  

        if (phi is not None): self.phi = phi
        else: self.phi = np.random.uniform(0,2*np.pi,self.Nkick)

        if (costh is not None): self.costh = costh
        else: self.costh = np.random.uniform(-1,1,self.Nkick)
        if (Mns is not None): self.Mns = Mns*units.M_sun.to(units.kg)
        else: self.Mns = np.random.uniform(3.,Mhe,self.Nkick)*units.M_sun.to(units.kg)
            
        if (th_ma is not None): self.th_ma = th_ma
        else: self.th_ma = np.random.uniform(0,2*np.pi,self.Nkick)
        self.E_ma =np.array([brentq(lambda x:ma -x + epre*np.sin(x),0,2*np.pi) for ma in self.th_ma])
        self.rpre = Apre*(1.-epre*np.cos(self.E_ma))*units.R_sun.to(units.m)
        self.Mhe = np.full((self.Nkick,), Mhe)*units.M_sun.to(units.kg)
        self.Mcomp = np.full((self.Nkick,), Mcomp)*units.M_sun.to(units.kg)
        self.Apre = np.full((self.Nkick,),Apre)*units.R_sun.to(units.m)
        self.epre = np.full((self.Nkick,),epre)
        
        # Get projection of R in the x-y plane to save later into output file

    def SN(self):
        """
        
        Mhe lies on origin moving in direction of positive y axis, Mcomp on negative X axis, Z completes right-handed coordinate system
        
        theta: angle between preSN He core velocity relative to Mcomp (i.e. the positive y axis) and the kick velocity
        phi: angle between Z axis and projection of kick onto X-Z plane
        
        Vr is velocity of preSN He core relative to Mcomp, directed along the positive y axis
        Vkick is kick velocity with components Vkx, Vky, Vkz in the above coordinate system
        V_sys is the resulting center of mass velocity of the system IN THE TRANSLATED COM FRAME, imparted by the SN
        Paper reference:
        Kalogera 1996: http://iopscience.iop.org/article/10.1086/177974/meta
            We use Eq 1, 3, 4, and 34: giving Vr, Apost, epost, and (Vsx,Vsy,Vsz) respectively
            Also see Fig 1 in that paper for coordinate system
        
        """

        self.flag=0      # set standard flag        

        G = constants.G.value
        Mhe, Mcomp, Mns, Apre, epre, rpre, Vkick, costh, phi = self.Mhe, self.Mcomp, self.Mns, self.Apre, self.epre, self.rpre, self.Vkick, self.costh, self.phi


        sinth = np.sqrt(1-(costh**2))
        #Mhe lies on origin moving in direction of positive y axis, Mcomp on negative X axis, Z completes right-handed coordinate system
        #See Fig 1 in Kalogera 1996

        # theta: angle between preSN He core velocity relative to Mcomp (i.e. the positive y axis) and the kick velocity
        # phi: angle between Z axis and projection of kick onto X-Z plane
        Vkx = Vkick*sinth*np.sin(phi)
        Vky = Vkick*costh
        Vkz = Vkick*sinth*np.cos(phi)
        
        #Eq 1, Kalogera 1996
        Vr = np.sqrt(G*(Mhe+Mcomp)*(2./rpre - 1./Apre))
        Mtot=Mns+Mcomp

        #Eqs 3 and 4, Kalogera 1996
        Apost = ((2.0/rpre) - (((Vkick**2)+(Vr**2)+(2*Vky*Vr))/(G*Mtot)))**-1
        sin_ang_ma = np.round(np.sqrt(G*(Mhe+Mcomp)*(1-epre**2)*Apre)/(rpre*Vr),5)
        self.sin_ang_ma = sin_ang_ma
        cos_ang_ma = np.sqrt(1-sin_ang_ma**2)
        self.cos_ang_ma = cos_ang_ma
        x = ((Vkz**2)+((sin_ang_ma*(Vr+Vky)-cos_ang_ma*(Vkx))**2))*(rpre**2)/(G*Mtot*Apost)
        epost = np.sqrt(1-x)
        # Eq 34, Kalogera 1996
        VSx = Mns*Vkx/Mtot
        VSy = (1.0/Mtot)*((Mns*Vky)-((Mhe-Mns)*Mcomp*Vr/(Mhe+Mcomp)))
        VSz = Mns*Vkz/Mtot
        V_sys = np.sqrt((VSx**2)+(VSy**2)+(VSz**2))
        
        
        # Peters (1964) c0_gw, eq. (5.11)
        #     def c0_gw(a0,e0):
        #         return a0*(1-e0**2)/(e0**(12./19.)*(1+121./304.*e0**2)**(870./2299.))

        
    #Merger time (in Gyr) for initial orbital separation a (Rsun) and eccentricity e.
    #This integrates de/dt and da/dt
    #Timestep is defined as dt_control*min(a/(da/dt),e/(de/dt)).
    #Evolution is terminated once the separation is equal to end_condition*max(R_1sch,R_2sch)
    #Timestep limit for eccentricity is only used while e>emin
    # will be in cm (unlike everywhere else
    
        Apost_cgs = Apost*units.m.to(units.cm)
        Mns_cgs = Mns*units.kg.to(units.g)
        Mcomp_cgs = Mcomp*units.kg.to(units.g)
        cgrav = constants.G.cgs.value
        clight = constants.c.cgs.value
        
        #integrand in eq. (5.14) of Peters (1964)
        T_merger_integrand = lambda e: e**(29./19.)*(1+121./304.*e**2)**(1181./2299.)/(1-e**2)**1.5
        #integral can be very small for small eccentricity, so fix tolerance
        #based on the ratio of c0_gw(a,e)**4/(a)**4
        tm_list = []
        for Apost_cgs_i, Mns_cgs_i, Mcomp_cgs_i, epost_i in zip(Apost_cgs, Mns_cgs, Mcomp_cgs, epost):
            
            if Apost_cgs_i < 0 :
                tm_list.append(-1)
            else:
                c0 = Apost_cgs_i*(1-epost_i**2)/(epost_i**(12./19.)*(1+121./304.*epost_i**2)**(870./2299.))
                beta_gw = 64./5.*cgrav**3/clight**5*Mns_cgs_i*Mcomp_cgs_i*(Mns_cgs_i+Mcomp_cgs_i)
                T = (12./19.)*(c0**4)/beta_gw*\
                        integrate.quadrature(T_merger_integrand,0,epost_i,tol=1e-10/(c0**4/Apost_cgs_i**4))[0]
                tmerge_gyr = T/(3.154e7*1e9)
                tm_list.append(tmerge_gyr)

        self.Apost, self.epost, self.VSx, self.VSy, self.VSz, self.V_sys, self.Vr, self.tmerger = Apost, epost, VSx, VSy, VSz, V_sys, Vr, np.array(tm_list)
        
        def SNCheck(self):
            """
            Paper References:
            Willems et al 2002: http://iopscience.iop.org/article/10.1086/429557/meta
                We use eq 21, 22, 23, 24, 25, 26 for checks of SN survival
            Kalogera and Lorimer 2000: http://iopscience.iop.org/article/10.1086/308417/meta
            
            V_He;preSN is the same variable as V_r from Kalogera 1996
            
            """
            Mhe, Mcomp, Mns, Apre, epre, rpre, Apost, epost, Vr, Vkick = self.Mhe, self.Mcomp, self.Mns, self.Apre, self.epre, self.rpre, self.Apost, self.epost, self.Vr, self.Vkick
            #Equation numbers and quotes in comments correspond to Willems et al. 2002 paper on J1655.
            Mtot_pre = Mhe + Mcomp
            Mtot_post = Mns + Mcomp

            # SNflag1: eq 21 (with typo fixed). Continuity demands Post SN orbit must pass through preSN positions.
            #from Flannery & Van Heuvel 1975                                                             

            self.oldSNflag1 = (1-epost <= Apre*(1-epre)/Apost) & (Apre*(1+epre)/Apost <= 1+epost)
            self.SNflag1 = (1-epost <= rpre/Apost) & (rpre/Apost <= 1+epost)

            # SNflag2: Equations 22 & 23. "Lower and upper limits on amount of orbital contraction or expansion that can take place                                
            #for a given amount of mass loss and a given magnitude of the kick velocity (see, e.g., Kalogera & Lorimer 2000)"                         self.Mhe, self.Mcomp,self.Apre, self.epre = Mhe, Mcomp, Apre, epre           

            self.SNflag2 = (Apre/Apost < 2-((Mtot_pre/Mtot_post)*((Vkick/Vr)-1)**2)) & (Apre/Apost > 2-((Mtot_pre/Mtot_post)*((Vkick/Vr)+1)**2))

            #SNflag3: Equations 24 and 25."The magnitude of the kick velocity imparted to the BH at birth is restricted to the
            #range determined by (Brandt & Podsiadlowski 1995; Kalogera & Lorimer 2000)
            #the first inequality expresses the requirement that the binary must remain bound after the SN explosion,
            #while the second inequality yields the minimum kick velocity required to keep the system bound if more than
            #half of the total system mass is lost in the explosion.

            self.SNflag3 = (Vkick/Vr < 1 + np.sqrt(2*Mtot_post/Mtot_pre)) & ((Mtot_post/Mtot_pre > 0.5) | (Vkick/Vr>1 - np.sqrt(2*Mtot_post/Mtot_pre)))
            
            
            
def impart_kick(data_detached, model, new_born_NS_mass=1.4):

    n_distrupted_list = []; n_wide_binary_list = []; n_merge_list = []; sn_type_list = []
    post_SN_a_mean_list = []; post_SN_a_median_list = []
    # lists for post-SN binary parameters 
    t_merger_list = []; a_list=[]; e_list=[]
    
    for i in range(len(data_detached['m1'].values)):
        a = float(data_detached.iloc[i]['a_pre']); e = 0
        m2 = float(data_detached.iloc[i]['M_2f(Msun)'])
        m1_he = float(data_detached.iloc[i]['M_1f(Msun)']); m1_ns = 1.4
        he_core = float(data_detached.iloc[i]['He_core_1(Msun)'])
        co_core = float(data_detached.iloc[i]['C_core_1(Msun)'])

        if model == 'Tauris':
            if (co_core > 1.37) and (co_core < 1.43): # >1.4
                Vkick_val = 20     # electron capture SN  
                sn_type = 1
            else:
                Vkick_val = 265    # core collapse SN
                sn_type = 0

        if model == 'Pod':
            if (he_core > 1.4) and (he_core < 2.5): 
                Vkick_val = 20     # electron capture SN  
                sn_type = 1
            else:

                Vkick_val = 265    # core collapse SN
                sn_type = 0

        Nkick_val=2000
        Mns_array = np.ones(Nkick_val)*new_born_NS_mass
        test = System(Mcomp=m2, Mhe=m1_he, Apre=a, epre=e, Mns=Mns_array, Vkick=Vkick_val, Nkick=Nkick_val,
                      galphi=None, galcosth=None, omega=None, phi=None, costh=None, th_ma = None)
        test.SN()
        
        # save distribution of post-SN kick values
        t_merger_list.append(test.tmerger.tolist())
        a_list.append(test.Apost.tolist())
        e_list.append(test.epost.tolist())

        apost = test.Apost[ test.Apost>0 ]
        post_SN_a_mean_list.append(np.mean(apost)) 
        post_SN_a_median_list.append(np.median(apost))

        n_wide_binary = np.sum(test.tmerger>13.8)
        n_distrupted = np.sum(test.tmerger<0)

        
        n_distrupted_list.append(n_distrupted) 
        n_wide_binary_list.append(n_wide_binary)
        n_merge_list.append(Nkick_val - n_wide_binary - n_distrupted)
        sn_type_list.append(sn_type)

    data_detached['n_distrupted'] = n_distrupted_list 
    data_detached['n_wide_binary'] = n_wide_binary_list
    data_detached['n_merge'] = n_merge_list
    data_detached['n_kicks'] = Nkick_val
    data_detached['sn_type'] = sn_type_list
    data_detached['post_SN_a_mean'] = post_SN_a_mean_list
    data_detached['post_SN_a_median'] = post_SN_a_median_list
    data_detached['t_merger'] = t_merger_list
    data_detached['a_post'] = a_list
    data_detached['e_post'] = e_list

    failed_BNS_outcomes = ['CE_merger','error']
    data_failed = data[ data['result'].isin(failed_BNS_outcomes)].copy()

    data_failed['n_distrupted'] = -1 
    data_failed['n_wide_binary'] = -1
    data_failed['n_merge'] = -1
    data_failed['n_kicks'] = -1

    data_tmerger_dist = pd.concat([data_detached, data_failed])

    if len(data_tmerger_dist.index)==len(data.index):
        filename_list = path_to_data.split('/')[0:-1] + [path_to_data.split('/')[-1][0:-4]+'_tmerger_dist_'+model+'_ECSN.csv']
        filename = '/'.join(filename_list)
        data_tmerger_dist.to_csv(filename)
        print(filename)
        
    return data_tmerger_dist            