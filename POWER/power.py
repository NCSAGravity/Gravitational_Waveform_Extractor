#!/usr/bin/env python

# Copyright (c) 2017 The Board of Trustees of the University of Illinois
# All rights reserved.
#
# Developed by: Daniel Johnson, E. A. Huerta, Roland Haas
#               NCSA Gravity Group
#               National Center for Supercomputing Applications
#               University of Illinois at Urbana-Champaign
#               http://gravity.ncsa.illinois.edu/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimers.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimers in the documentation
# and/or other materials provided with the distribution.
#
# Neither the names of the National Center for Supercomputing Applications,
# University of Illinois at Urbana-Champaign, nor the names of its
# contributors may be used to endorse or promote products derived from this
# Software without specific prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# WITH THE SOFTWARE.

# Based off of SimulationTools Mathematica Package
# http://www.simulationtools.org/

import numpy as np
import glob
import os
import h5py
import re
import math
import sys
import warnings
import scipy.optimize
import scipy.interpolate
import scipy.integrate
import argparse
import configparser

#-----Function Definitions-----#

def joinDsets(dsets):
    
        """joints multiple datasets which each have a
        time like first column, eg iteration number of
        time. Removes overlapping segments, keeping the
        last segment.
    
        dsets = iterable of 2d array like objects with data"""
        # joins multiple datasets of which the first column is assumed to be "time"
        if(not dsets):
            return None
        length = 0
        for d in dsets:
            length += len(d)
        newshape = list(dsets[0].shape)
        newshape[0] = length
        dset = np.empty(shape=newshape, dtype=dsets[0].dtype)
        usedlength = 0
        for d in dsets:
            insertpointidx = np.where(dset[0:usedlength,0] >= d[0,0])
            if(insertpointidx[0].size):
                insertpoint = insertpointidx[0][0]
            else:
                insertpoint = usedlength
            newlength = insertpoint+len(d)
            dset[insertpoint:newlength] = d
            usedlength = newlength
        return dset[0:usedlength]


def loadHDF5Series(nameglob, series):
        """load HDF5 timeseries data and concatenate the content of multiple files
    
        nameglob = a shell glob that matches all files to be loaded,
        files are sorted alphabetically
        series = HDF5 dataset name of dataset to load from files"""
        dsets = list()
        for fn in sorted(glob.glob(nameglob)):
            fh = h5py.File(fn, "r")
            dsets.append(fh[series])
        return joinDsets(dsets)



    
#Convert radial to tortoise coordinates
def RadialToTortoise(r, M):
    """
    Convert the radial coordinate to the tortoise coordinate

    r = radial coordinate
    M = ADMMass used to convert coordinate
    return = tortoise coordinate value
    """
    return r + 2. * M * math.log( r / (2. * M) - 1.)

#Convert modified psi4 to strain
def psi4ToStrain(mp_psi4, f0):
    """
    Convert the input mp_psi4 data to the strain of the gravitational wave
    
    mp_psi4 = Weyl scalar result from simulation
    f0 = cutoff frequency
    return = strain (h) of the gravitational wave
    """
    #TODO: Check for uniform spacing in time
    t0 = mp_psi4[:, 0]
    list_len = len(t0)
    complexPsi = np.zeros(list_len, dtype=np.complex_)
    complexPsi = mp_psi4[:, 1]+1.j*mp_psi4[:, 2]

    freq, psif = myFourierTransform(t0, complexPsi)
    dhf = ffi(freq, psif, f0)
    hf = ffi(freq, dhf, f0)

    time, h = myFourierTransformInverse(freq, hf, t0[0])
    hTable = np.column_stack((time, h))
    return hTable

#Fixed frequency integration
# See https://arxiv.org/abs/1508.07250 for method
def ffi(freq, data, f0):
    """
    Integrates the data according to the input frequency and cutoff frequency

    freq = fourier transform frequency
    data = input on which ffi is performed
    f0 = cutoff frequency
    """
    f1 = f0/(2*math.pi)
    fs = freq
    gs = data
    mask1 = (np.sign((fs/f1) - 1) + 1)/2.
    mask2 = (np.sign((-fs/f1) - 1) + 1)/2.
    mask = 1 - (1 - mask1) * (1 - mask2)
    fs2 = mask * fs + (1-mask) * f1 * np.sign(fs - np.finfo(float).eps)
    new_gs = gs/(2*math.pi*1.j*fs2)
    return new_gs

#Fourier Transform
def myFourierTransform(t0, complexPsi):
    """
    Transforms the complexPsi data to frequency space

    t0 = time data points
    complexPsi = data points of Psi to be transformed
    """
    psif = np.fft.fft(complexPsi, norm="ortho")
    l = len(complexPsi)
    n = int(math.floor(l/2.))
    newpsif = psif[l-n:]
    newpsif = np.append(newpsif, psif[:l-n])
    T = np.amin(np.diff(t0))*l
    freq = range(-n, l-n)/T
    return freq, newpsif

#Inverse Fourier Transform
def myFourierTransformInverse(freq, hf, t0):
    l = len(hf)
    n = int(math.floor(l/2.))
    newhf = hf[n:]
    newhf = np.append(newhf, hf[:n])
    amp = np.fft.ifft(newhf, norm="ortho")
    df = np.amin(np.diff(freq))
    time = t0 + range(0, l)/(df*l)
    return time, amp

def angular_momentum(x, q, m, chi1, chi2, LInitNR):
    eta = q/(1.+q)**2
    m1 = (1.+math.sqrt(1.-4.*eta))/2.
    m2 = m - m1
    S1 = m1**2. * chi1
    S2 = m2**2. * chi2
    Sl = S1+S2
    Sigmal = S2/m2 - S1/m1
    DeltaM = m1 - m2
    mu = eta
    nu = eta
    GammaE = 0.5772156649;
    e4 = -(123671./5760.)+(9037.* math.pi**2.)/1536.+(896.*GammaE)/15.+(-(498449./3456.)+(3157.*math.pi**2.)/576.)*nu+(301. * nu**2.)/1728.+(77.*nu**3.)/31104.+(1792. *math.log(2.))/15.
    e5 = -55.13
    j4 = -(5./7.)*e4+64./35.
    j5 = -(2./3.)*e5-4988./945.-656./135. * eta;
    CapitalDelta = (1.-4.*eta)**0.5

    # RH: expression was originally provided by EAH
    # TODO: get referecen for this from EAH
    l = (eta/x**(1./2.)*(
        1. +
        x*(3./2. + 1./6.*eta) + 
        x**2. *(27./8. - 19./8.*eta + 1./24.*eta**2.) + 
        x**3. *(135./16. + (-6889./144. + 41./24. * math.pi**2.)*eta + 31./24.*eta**2. + 7./1296.*eta**3.) + 
        x**4. *((2835./128.) + eta*j4 - (64.*eta*math.log(x)/3.))+ 
        x**5. *((15309./256.) + eta*j5 + ((9976./105.) + (1312.*eta/15.))*eta*math.log(x))+
        x**(3./2.)*(-(35./6.)*Sl - 5./2.*DeltaM* Sigmal) + 
        x**(5./2.)*((-(77./8.) + 427./72.*eta)*Sl + DeltaM* (-(21./8.) + 35./12.*eta)*Sigmal) + 
        x**(7./2.)*((-(405./16.) + 1101./16.*eta - 29./16.*eta**2.)*Sl + DeltaM*(-(81./16.) + 117./4.*eta - 15./16.*eta**2.)*Sigmal) + 
        (1./2. + (m1 - m2)/2. - eta)* chi1**2. * x**2. +
        (1./2. + (m2 - m1)/2. - eta)* chi2**2. * x**2. + 
        2.*eta*chi1*chi2*x**2. +
        ((13.*chi1**2.)/9. +
        (13.*CapitalDelta*chi1**2.)/9. -
        (55.*nu*chi1**2.)/9. - 
        29./9.*CapitalDelta*nu*chi1**2. + 
        (14.*nu**2. *chi1**2.)/9. +
        (7.*nu*chi1*chi2)/3. +
        17./18.* nu**2. * chi1 * chi2 + 
        (13.* chi2**2.)/9. -
        (13.*CapitalDelta*chi2**2.)/9. -
        (55.*nu*chi2**2.)/9. +
        29./9.*CapitalDelta*nu*chi2**2. +
        (14.*nu**2. * chi2**2.)/9.)
        * x**3.))
    return l - LInitNR


def getADMMassFromTwoPunctureBBH(meta_filename):
    """
    Determine cutoff frequency of simulation

    meta_filename = path to TwoPunctures.bbh
    return = initial ADM mass of system
    """

    config = configparser.ConfigParser()
    config.read(meta_filename)

    ADMmass = float(config['metadata']['initial-ADM-energy'])

    return ADMmass

def getCutoffFrequencyFromTwoPuncturesBBH(meta_filename):
    """
    Determine cutoff frequency of simulation

    meta_filename = path to TwoPunctures.bbh
    return = cutoff frequency
    """

    config = configparser.ConfigParser()
    config.read(meta_filename)

    position1x = float(config['metadata']['initial-bh-position1x'])
    position1y = float(config['metadata']['initial-bh-position1y'])
    position1z = float(config['metadata']['initial-bh-position1z'])
    position2x = float(config['metadata']['initial-bh-position2x'])
    position2y = float(config['metadata']['initial-bh-position2y'])
    position2z = float(config['metadata']['initial-bh-position2z'])
    momentum1x = float(config['metadata']['initial-bh-momentum1x'])
    momentum1y = float(config['metadata']['initial-bh-momentum1y'])
    momentum1z = float(config['metadata']['initial-bh-momentum1z'])
    momentum2x = float(config['metadata']['initial-bh-momentum2x'])
    momentum2y = float(config['metadata']['initial-bh-momentum2y'])
    momentum2z = float(config['metadata']['initial-bh-momentum2z'])
    spin1x = float(config['metadata']['initial-bh-spin1x'])
    spin1y = float(config['metadata']['initial-bh-spin1y'])
    spin1z = float(config['metadata']['initial-bh-spin1z'])
    spin2x = float(config['metadata']['initial-bh-spin2x'])
    spin2y = float(config['metadata']['initial-bh-spin2y'])
    spin2z = float(config['metadata']['initial-bh-spin2z'])
    mass1 = float(config['metadata']['initial-bh-puncture-adm-mass1'])
    mass2 = float(config['metadata']['initial-bh-puncture-adm-mass2'])

    angularmomentum1x = position1y * momentum1z - momentum1z * momentum1y
    angularmomentum1y = position1z * momentum1x - momentum1x * momentum1z
    angularmomentum1z = position1x * momentum1y - momentum1y * momentum1x

    angularmomentum2x = position2y * momentum2z - momentum2z * momentum2y
    angularmomentum2y = position2z * momentum2x - momentum2x * momentum2z
    angularmomentum2z = position2x * momentum2y - momentum2y * momentum2x

    angularmomentumx = angularmomentum1x + angularmomentum2x
    angularmomentumy = angularmomentum1y + angularmomentum2y
    angularmomentumz = angularmomentum1z + angularmomentum2z

    LInitNR = math.sqrt(angularmomentumx**2 + angularmomentumy**2 + angularmomentumz**2)
    S1 = math.sqrt(spin1x**2 + spin1y**2 + spin1z**2)
    S2 = math.sqrt(spin2x**2 + spin2y**2 + spin2z**2)

    M = mass1+mass2
    q = mass1/mass2
    chi1 = S1/mass1**2
    chi2 = S2/mass2**2
    # .014 is the initial guess for cutoff frequency
    omOrbPN = scipy.optimize.fsolve(angular_momentum, .014, (q, M, chi1, chi2, LInitNR))[0]
    omOrbPN = omOrbPN**(3./2.)
    omGWPN = 2. * omOrbPN
    omCutoff = 0.75 * omGWPN
    return omCutoff

def getModesInFile(sim_path):
    """
    Find all modes and radii in file.

    sim_path = path to simulation main directory
    return = radii, modes
    """

    fn = sorted(glob.glob(sim_path+"/output-????/*/mp_[Pp]si4.h5"))[0]
    with h5py.File(fn, "r") as fh:
        radii = set()
        modes = set()
        for dset in fh:
            # TODO: extend Multipole to save the radii as attributes and/or
            # use a group structure in the hdf5 file
            m = re.match(r'l(\d*)_m(-?\d*)_r(\d*\.\d)', dset)
            if m:
                radius = float(m.group(3))
                mode = (int(m.group(1)), int(m.group(2)))
                modes.add(mode)
                radii.add(radius)
    modes = sorted(modes)
    radii = sorted(radii)
    return radii, modes



# -----------------------------------------------------------------------------
# POWER Method
# -----------------------------------------------------------------------------
 
    
def POWER(sim_path, radii, modes):
    main_dir = sim_path
    sim = os.path.split(sim_path)[-1]
    
    simdirs = main_dir+"/output-????/%s/" % (sim)
    f0 = getCutoffFrequencyFromTwoPuncturesBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
    #Get simulation total mass
    ADMMass = getADMMassFromTwoPunctureBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
    
    #Create data directories
    main_directory = "Extrapolated_Strain"
    sim_dir = main_directory+"/"+sim
    if not os.path.exists(main_directory):
            os.makedirs(main_directory)
    if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

    # get translation table from (mode, radius) to dataset name
    # TODO: this ought to be handled differently
    dsets = {}
    fn = sorted(glob.glob(sim_path+"/output-????/*/mp_[Pp]si4.h5"))[0]
    with h5py.File(fn, "r") as fh:
        for dset in fh:
            # TODO: extend Multipole to save the radii as attributes and/or
            # use a group structure in the hdf5 file
            m = re.match(r'l(\d*)_m(-?\d*)_r(\d*\.\d)', dset)
            if m:
                radius = float(m.group(3))
                mode = (int(m.group(1)), int(m.group(2)))
                dsets[(radius, mode)] = dset

    #Get Psi4
    
    for (l,m) in modes:                      # 25 times through the loop, from (1,1) to (4,4)
            mp_psi4_vars = []
            strain = []
            phase = []
            amp = []
            for i in range(len(radii)):      # so 7 times through each mode at each of the 7 radii
                    #------------------------------------------------
                    # Read in HDF5 data
                    #------------------------------------------------
                    radius = radii[i]
                    psi4dsetname = dsets[(radius, (l,m))]
                    mp_psi4 = loadHDF5Series(simdirs+"mp_psi4.h5", psi4dsetname)
                    mp_psi4_vars.append(mp_psi4)
                    

                    #-----------------------------------------
                    # Prepare for conversion to strain
                    #-----------------------------------------
                    # retardate time by estimated travel time to each detector,
                    # convert from psi4 to r*psi4 to account for initial 1/r falloff
                    # RH: it might be even better (though harder to define) to
                    # get a retardating time by looking at the time of the
                    # maximum (found as an optimization over an interpolating
                    # function, not argmax)
                    mp_psi4_vars[i][:, 0] -= RadialToTortoise(radius, ADMMass)
                    mp_psi4_vars[i][:, 1] *= radii[i]
                    mp_psi4_vars[i][:, 2] *= radii[i]
    
                    #Check for psi4 amplitude going to zero
                    # RH: this makes very little sense since the amplitude is
                    # expected to be zero initially and very late
                    psi4_amp = np.sqrt(mp_psi4_vars[i][:, 1]**2 + mp_psi4_vars[i][:, 2]**2)
                    min_psi4_amp = np.amin(psi4_amp)
                    if(min_psi4_amp < np.finfo(float).eps and l >= 2):
                        print("The psi4 amplitude is near zero. The phase is ill-defined.")
    
                    #Fixed-frequency integration twice to get strain
                    #-----------------------------------------------------------------
                    # Strain Conversion
                    #-----------------------------------------------------------------

                    hTable = psi4ToStrain(mp_psi4_vars[i], f0)  # table of strain

                    
                    time = hTable[:, 0]
                    h = hTable[:, 1]
                    hplus = h.real
                    hcross = h.imag
                    newhTable = np.column_stack((time, hplus, hcross))
                    warnings.filterwarnings('ignore')
                    finalhTable = newhTable.astype(float)
                    np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_strain_at_"+str(radii[i])+"_l"+str(l)+"_m"+str(m)+".dat", finalhTable)
                    strain.append(finalhTable)
                    
                    
                    #-------------------------------------------------------------------
                    # Analysis of Strain
                    #-------------------------------------------------------------------
                    #Get phase and amplitude of strain
                    h_phase = np.unwrap(np.angle(h))
                    # print(len(h_phase), "h_phase length")
                    # print(len(time), "time length")
                    angleTable = np.column_stack((time, h_phase))     ### start here
                    angleTable = angleTable.astype(float)             ### b/c t is defined based on
                    phase.append(angleTable)                          ### time here
                    h_amp = np.absolute(h)
                    ampTable = np.column_stack((time, h_amp))
                    ampTable = ampTable.astype(float)
                    amp.append(ampTable)


            #print("h_amp:" , h_amp)

            #----------------------------------------------------------------------
            # Extrapolation
            #----------------------------------------------------------------------

            # get common range in times
            tmin = max([phase[i][ 0,0] for i in range(len(phase))])
            tmax = min([phase[i][-1,0] for i in range(len(phase))])

            # smallest timestep in any series
            dtmin = min([np.amin(np.diff(phase[0][:,0])) for i in range(len(phase))])

            # uniform, common time
            t = np.arange(tmin, tmax, dtmin)

            # Interpolate phase and amplitude
            interpolation_order = 9
            for i in range(len(radii)):
                    interp_function = scipy.interpolate.interp1d(amp[i][:, 0], amp[i][:, 1], kind=interpolation_order)
                    resampled_amp_vals = interp_function(t)
                    amp[i] = np.column_stack((t, resampled_amp_vals))

                    interp_function = scipy.interpolate.interp1d(phase[i][:, 0], phase[i][:, 1], kind=interpolation_order)
                    resampled_phase_vals = interp_function(t)
                    # try and keep all phases at the amplitude maximum within 2pi of each other
                    # alignment is between neighbhours just in case there actually ever is
                    # >2pi difference between the innermost and the ohtermost detector
                    if(i > 0):
                        # for some modes (post 2,2) the initial junk can be the
                        # largest amplitude contribution, so w try to skip it
                        # when looking for maxima
                        junk_time = 50.
                        post_junk_idx_p = amp[i-1][:,0] > junk_time
                        post_junk_idx = amp[i-1][:,0] > junk_time
                        maxargp = np.argmax(amp[i-1][post_junk_idx_p,1])
                        maxarg = np.argmax(amp[i][post_junk_idx,1])
                        phase_shift = round((resampled_phase_vals[post_junk_idx][maxarg] - phase[i-1][post_junk_idx_p][maxargp,1])/(2.*math.pi))*2.*math.pi
                        resampled_phase_vals -= phase_shift
                    phase[i] = np.column_stack((t, resampled_phase_vals))

            #Extrapolate
            phase_extrapolation_order = 1
            amp_extrapolation_order = 2
            radii = np.asarray(radii, dtype=float)
            # TODO: replace by np.ones (which is all it does anyway)
            A_phase = np.power(radii, 0)
            A_amp = np.power(radii, 0)

            for i in range(1, phase_extrapolation_order+1):
                    A_phase = np.column_stack((A_phase, np.power(radii, -1*i*math.pi)))
    
            for i in range(1, amp_extrapolation_order+1):
                    A_amp = np.column_stack((A_amp, np.power(radii, -1*i*math.pi)))
    
            radially_extrapolated_phase = np.empty(0)
            radially_extrapolated_amp = np.empty(0)
            for i in range(0, len(t)):
                    b_phase = np.empty(0)
                    for j in range(len(radii)):
                            b_phase = np.append(b_phase, phase[j][i, 1])
                    x_phase = np.linalg.lstsq(A_phase, b_phase)[0]
                    radially_extrapolated_phase = np.append(radially_extrapolated_phase, x_phase[0])
    
                    b_amp = np.empty(0)
                    for j in range(len(radii)):
                            b_amp = np.append(b_amp, amp[j][i, 1])
                    x_amp = np.linalg.lstsq(A_amp, b_amp)[0]
                    radially_extrapolated_amp = np.append(radially_extrapolated_amp, x_amp[0])
    
            radially_extrapolated_h_plus = np.empty(0)
            radially_extrapolated_h_cross = np.empty(0)
            for i in range(0, len(radially_extrapolated_amp)):
                    radially_extrapolated_h_plus = np.append(radially_extrapolated_h_plus, radially_extrapolated_amp[i] * math.cos(radially_extrapolated_phase[i]))
                    radially_extrapolated_h_cross = np.append(radially_extrapolated_h_cross, radially_extrapolated_amp[i] * math.sin(radially_extrapolated_phase[i]))


            np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_strain_l"+str(l)+"_m"+str(m)+".dat", np.column_stack((t, radially_extrapolated_h_plus, radially_extrapolated_h_cross)))
            np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_amplitude_l"+str(l)+"_m"+str(m)+".dat", np.column_stack((t, radially_extrapolated_amp)))
            np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_phase_l"+str(l)+"_m"+str(m)+".dat", np.column_stack((t, radially_extrapolated_phase[:])))
        

# -----------------------------------------------------------------------------
# Nakano Method
# -----------------------------------------------------------------------------

        
def eq_29(argv, args):
    
    def psi4ToStrain2(mp_psi4, f0):
        """
        Convert the input mp_psi4 data to the strain of the gravitational wave
        
        mp_psi4 = Weyl scalar result from simulation
        f0 = cutoff frequency
        return = strain (h) of the gravitational wave
        """
        #TODO: Check for uniform spacing in time
        t0 = mp_psi4[:, 0]
        list_len = len(t0)
        complexPsi = np.zeros(list_len, dtype=np.complex_)
        complexPsi = mp_psi4[:, 1]+1.j*mp_psi4[:, 2]
    
        freq, psif = myFourierTransform(t0, complexPsi)
        hf = ffi(freq, psif, f0)
    
        time, h = myFourierTransformInverse(freq, hf, t0[0])
        hTable = np.column_stack((time, h))
        return hTable
            
### ---------
            
    radii_list = [100.00 , 115.00 , 136.00 , 167.00, 214.00 , 300.00 , 500.00]
    modes = [(0, 0), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (2, 2), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3)]

    if len(argv) < 5:               ### For looping over all modes
        print("Extrapolating over all modes")
        paths = argv[2]
        main_dir = paths
        sim = os.path.split(paths)[-1]
        simdirs = main_dir+"/output-????/%s/" % (sim)
        
    ### Setting up directory for saving the file(s) at the end ###
        main_directory = "Extrapolated_Strain(Nakano_Kerr)"
        sim_dir = main_directory+"/"+sim
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

        for (l,m) in modes:
            for i in radii_list:
            
                    landm = (l,m)
                    l = int(landm[0])
                    m = int(landm[1])
                    radius = float(i)
                    if m > l:                                                           ### Fail if m > l
                        print("Error: %s is a non-physical mode" % (landm))
                        sys.exit()
                    
        
                    modes = "l%d_m%d_r%.2f" %(l,m,radius)    
                    ar = loadHDF5Series(simdirs+"mp_psi4.h5" , modes)   # loads HDF5 Series from file mp_psi4.h5, specifically the "l%d_m%d_r100.00" ones ... let's loop this over all radii
                    
                
                    t = ar[:,0]      # 1st column of ar, time data points
                    psi = ar[:,1]    # 2nd column of ar, data points for psi
                    impsi = ar[:, 2]   # 3rd column of ar, data points for imaginary psi
                    
                         
                    f0 = getCutoffFrequencyFromTwoPuncturesBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
                    length_psi = len(psi)
                    some_zeros1 = np.zeros(length_psi, dtype = np.complex_)
        
                    new_psi = np.column_stack((t, psi, some_zeros1))       
                    new_impsi = np.column_stack((t, impsi, some_zeros1))
        
                    s_in = psi4ToStrain2(new_psi, f0)
                    ims_in = psi4ToStrain2(new_impsi, f0)
                    s_in = s_in[:-2]
                    ims_in = ims_in[:-2]
                    s_in = np.column_stack((s_in, some_zeros1[:-2]))  
                    ims_in = np.column_stack((ims_in, some_zeros1[:-2]))
                            
                    d_in = psi4ToStrain2(s_in, f0)
                    imd_in = psi4ToStrain2(ims_in, f0)
                
                    
                    mass_path = sorted(glob.glob(simdirs))
                    A_val = np.loadtxt(mass_path[-1]+"quasilocalmeasures-qlm_scalars..asc")     ## For mass calculation
                    r = radius
                    M = A_val[:,58][-1]
                    a = (A_val[:,37]/A_val[:,58])[-1]
        
                       
                    modes_a = "l%d_m%d_r%.2f" %(l+1, m, radius)         # "top" modes                    
                    modes_b = "l%d_m%d_r%.2f" %(l-1, m, radius)         # "bottom" modes
                    ar_a = loadHDF5Series(simdirs+'mp_psi4.h5' , modes_a)
                    
                    
                    t_a = ar_a[:,0]
                    psi_a = ar_a[:,1]
                    impsi_a = ar_a[:,2]
                    t_b = t_a
                
        
                    if m > l-1:                                     # if m is greater than the bottom mode...
                        psi_b = np.zeros(len(psi_a))                # ...fill psi_b and impsi_b arrays with zeros (mode is unphysical)
                        impsi_b = np.zeros(len(impsi_a))
        
                    else:
                        ar_b = loadHDF5Series(simdirs+'mp_psi4.h5' , modes_b)
                        psi_b = ar_b[:,1]      
                        impsi_b = ar_b[:,2]
        
        
                    A = 1-(2*M/r)
                    a_1 = r
                    a_2 = ((l-1)*(l+2))/(2*r)
                    a_3 = ((l-1)*(l+2)*(l**2 + l -4))/(8*r*r)
                    B = ((0+a*2j)/((l+1)**2))*((((l+3)*(l-1)*(l+m+1)*(l-m+1))/((2*l+1)*(2*l+3)))**(1/2))
                    b_1 = r
                    b_2 = l*(l+3)
                    C = ((0+a*2j)/((l)**2))*((((l+2)*(l-2)*(l+m)*(l-m))/((2*l-1)*(2*l+1)))**(1/2))
                    c_1 = r
                    c_2 = (l-2)*(l+1)
                    
                  
                    ans = A*(a_1*psi[2:] - a_2*s_in[:,2] + a_3*d_in[:,1]) + B*(b_1*np.gradient(psi_a, t_a)[2:] - b_2*psi_a[2:]) - C*(c_1*np.gradient(psi_b, t_b)[2:] - c_2*psi_b[2:])
                    imans = A*(a_1*impsi[2:] - a_2*ims_in[:,1] + a_3*imd_in[:,1]) + B*(b_1*np.gradient(impsi_a, t_a)[2:] - b_2*impsi_a[2:]) - C*(c_1*np.gradient(impsi_b, t_b)[2:] - c_2*impsi_b[2:])
                 
                    
                    f0 = getCutoffFrequencyFromTwoPuncturesBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
                    length = len(ans)
                    some_zeros = np.zeros(length, dtype = np.complex_)
                    mp_psi4 = np.column_stack((t[:-2], ans, some_zeros))
                    immp_psi4 = np.column_stack((t[:-2], imans, some_zeros))
                    
                    f1 = psi4ToStrain(mp_psi4, f0)
                    imf1 = psi4ToStrain(immp_psi4, f0)
                    
                    f3_cmp = f1 + imf1*.1j
                    # f3_cmp = f2 + imf2*1j
                    imf3 = f3_cmp.imag
                    f3 = f3_cmp.real
                
                    complex_psi = f3 + 1j*imf3
        
                       
                    np.savetxt("./Extrapolated_Strain(Nakano_Kerr)/"+sim+"/"+sim+"_f2_l%d_m%d_r%.2f_Looped.dat" %(l, m, radius) , np.column_stack((t[4:] , complex_psi.real[:-2] , complex_psi.imag[:-2])))   


    else:                               ### if they specify a certain mode
        mode = argv[2]   
        print("Extrapolating for %s mode" %mode)    
        paths = argv[4]
        main_dir = paths
        sim = os.path.split(paths)[-1]
        simdirs = main_dir+"/output-????/%s/" % (sim)
        ### Setting up directory for saving the file(s) at the end ###
    
        main_directory = "Extrapolated_Strain(Nakano_Kerr)"
        sim_dir = main_directory+"/"+sim
        if not os.path.exists(main_directory):
                os.makedirs(main_directory)
        if not os.path.exists(sim_dir):
                os.makedirs(sim_dir)
            
        for i in radii_list:
            landm = argv[2]
            landm = landm.split(',')
            l = int(landm[0])
            m = int(landm[1])
            radius = float(i)
            if m > l:                                                           ### Fail if m > l
                print("Error: %s is a non-physical mode" % (landm))
                sys.exit()
            

            modes = "l%d_m%d_r%.2f" %(l,m,radius)    
            ar = loadHDF5Series(simdirs+"mp_psi4.h5" , modes)   # loads HDF5 Series from file mp_psi4.h5, specifically the "l%d_m%d_r100.00" ones ... let's loop this over all radii
            
        
            t = ar[:,0]      # 1st column of ar, time data points
            psi = ar[:,1]    # 2nd column of ar, data points for psi
            impsi = ar[:, 2]   # 3rd column of ar, data points for imaginary psi
            
                 
            f0 = getCutoffFrequencyFromTwoPuncturesBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
            length_psi = len(psi)
            some_zeros1 = np.zeros(length_psi, dtype = np.complex_)

            new_psi = np.column_stack((t, psi, some_zeros1))
            new_impsi = np.column_stack((t, impsi, some_zeros1))

            s_in = psi4ToStrain2(new_psi, f0)
            ims_in = psi4ToStrain2(new_impsi, f0)
            s_in = s_in[:-2]
            ims_in = ims_in[:-2]
            s_in = np.column_stack((s_in, some_zeros1[:-2]))  
            ims_in = np.column_stack((ims_in, some_zeros1[:-2]))

            d_in = psi4ToStrain2(s_in, f0)
            imd_in = psi4ToStrain2(ims_in, f0)
        
            
            mass_path = sorted(glob.glob(simdirs))
            A_val = np.loadtxt(mass_path[-1]+"quasilocalmeasures-qlm_scalars..asc")     ## For mass calculation
            r = radius
            M = A_val[-1,58]
            a = A_val[-1,37]/A_val[-1,58]

               
            modes_a = "l%d_m%d_r%.2f" %(l+1, m, radius)         # "top" modes            
            modes_b = "l%d_m%d_r%.2f" %(l-1, m, radius)         # "bottom" modes


            ar_a = loadHDF5Series(simdirs+'mp_psi4.h5' , modes_a)

            t_a = ar_a[:,0]
            psi_a = ar_a[:,1]
            impsi_a = ar_a[:,2]
            t_b = t_a
        

            if m > l-1:                                     # if m is greater than the bottom mode...
                psi_b = np.zeros(len(psi_a))                # ...fill psi_b and impsi_b arrays with zeros (mode is unphysical)
                impsi_b = np.zeros(len(impsi_a))

            else:
                ar_b = loadHDF5Series(simdirs+'mp_psi4.h5' , modes_b)
                psi_b = ar_b[:,1]      
                impsi_b = ar_b[:,2]


            A = 1-(2*M/r)
            a_1 = r
            a_2 = ((l-1)*(l+2))/(2*r)
            a_3 = ((l-1)*(l+2)*(l**2 + l -4))/(8*r*r)
            B = ((0+a*2j)/((l+1)**2))*((((l+3)*(l-1)*(l+m+1)*(l-m+1))/((2*l+1)*(2*l+3)))**(1/2))
            b_1 = r
            b_2 = l*(l+3)
            C = ((0+a*2j)/((l)**2))*((((l+2)*(l-2)*(l+m)*(l-m))/((2*l-1)*(2*l+1)))**(1/2))
            c_1 = r
            c_2 = (l-2)*(l+1)
        

            ans = A*(a_1*psi[2:] - a_2*s_in[:,2] + a_3*d_in[:,1]) + B*(b_1*np.gradient(psi_a, t_a)[2:] - b_2*psi_a[2:]) - C*(c_1*np.gradient(psi_b, t_b)[2:] - c_2*psi_b[2:])
            imans = A*(a_1*impsi[2:] - a_2*ims_in[:,1] + a_3*imd_in[:,1]) + B*(b_1*np.gradient(impsi_a, t_a)[2:] - b_2*impsi_a[2:]) - C*(c_1*np.gradient(impsi_b, t_b)[2:] - c_2*impsi_b[2:])
            
                        
            f0 = getCutoffFrequencyFromTwoPuncturesBBH(main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim))
            length = len(ans)
            some_zeros = np.zeros(length, dtype = np.complex_)
            mp_psi4 = np.column_stack((t[:-2], ans, some_zeros))
            immp_psi4 = np.column_stack((t[:-2], imans, some_zeros))
            
            f1 = psi4ToStrain(mp_psi4, f0)
            imf1 = psi4ToStrain(immp_psi4, f0)
            
            f3_cmp = f1 + imf1*.1j
            # f3_cmp = f2 + imf2*1j
            imf3 = f3_cmp.imag
            f3 = f3_cmp.real
            
            complex_psi = f3 + 1j*imf3

            
            np.savetxt("./Extrapolated_Strain(Nakano_Kerr)/"+sim+"/"+sim+"_f2_l%d_m%d_r%.2f_spec.dat" %(l, m, radius) , np.column_stack((t[4:] , complex_psi.real[:-2] , complex_psi.imag[:-2])))   


# -----------------------------------------------------------------------------
        
    
### argparse machinery:

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        print("Not a directory: %s" %(string))
        # raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Choose extrapolation method')
parser.add_argument("method" , choices=["POWER" , "Nakano"] , help="Extrapolation method to be used here")
parser.add_argument('-r', "--radii" , type=int , help="For POWER method; Number of radii to be used", default=7)
parser.add_argument('-m' , "--modes" , type=str , help="For Nakano method; modes to use, l,m. Leave blank to extrapolate over all available modes")
parser.add_argument("path" , type=dir_path , help="Simulation to be used here")
args = parser.parse_args()

if args.method == "POWER":        
    print("Extrapolating with POWER method...")
    radii, modes = getModesInFile(args.path)
    radii = radii[0:args.radii]

    POWER(args.path, radii, modes)
    
    

elif args.method == "Nakano":
        print("Extrapolating with Nakano method...")
        eq_29(sys.argv, args)    
  


            
