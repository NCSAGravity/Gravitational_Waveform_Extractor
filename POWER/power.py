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
import string
import math
import sys
import warnings
import scipy.optimize
import scipy.interpolate

#-----Function Definitions-----#

#Function used in getting psi4 from simulation
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

#Function used in getting psi4 from simulation
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
	a1 = -2.18522;
	a2 = 1.05185;
	a3 = -2.43395;
	a4 = 0.400665;
	a5 = -5.9991;
	CapitalDelta = (1.-4.*eta)**0.5

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

#Get cutoff frequency
def getCutoffFrequency(sim_name):
	"""
	Determine cutoff frequency of simulation

	sim_name = string of simulation
	return = cutoff frequency
	"""
	filename = main_dir+"/output-0000/%s.par" % (sim_name)
	with open(filename) as file:
		contents = file.readlines()
		for line in contents:
			line_elems = line.split(" ")
			if(line_elems[0] == "TwoPunctures::par_b"):
				par_b = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::center_offset[0]"):
				center_offset = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::par_P_plus[1]"):
				pyp = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::par_P_minus[1]"):
				pym = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::target_M_plus"):
				m1 = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::target_M_minus"):
				m2 = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::par_S_plus[2]"):
				S1 = float(line_elems[-1])
			if(line_elems[0] == "TwoPunctures::par_S_minus[2]"):
				S2 = float(line_elems[-1])

	xp = par_b + center_offset
	xm = -1*par_b + center_offset
	LInitNR = xp*pyp + xm*pym
	M = m1+m2
	q = m1/m2
	chi1 = S1/m1**2
	chi2 = S2/m2**2
	# .014 is the initial guess for cutoff frequency
	omOrbPN = scipy.optimize.fsolve(angular_momentum, .014, (q, M, chi1, chi2, LInitNR))[0]
	omOrbPN = omOrbPN**(3./2.)
	omGWPN = 2. * omOrbPN
	omCutoff = 0.75 * omGWPN
	return omCutoff

#Get Energy
def get_energy(sim):
	"""
	Save the energy radiated energy
	sim = string of simulation
	"""
	python_strain = np.loadtxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_strain_l2_m2.dat")
	val = np.zeros(len(python_strain))
	val = val.astype(np.complex_)
	cur_max_time = python_strain[0][0]
	cur_max_amp = abs(pow(python_strain[0][1], 2))
	# TODO: rewrite as array operations (use numpy.argmax)
	for i in python_strain[:]:
		cur_time = i[0]
		cur_amp = abs(pow(i[1], 2))
		if(cur_amp>cur_max_amp):
			cur_max_amp = cur_amp
			cur_max_time = cur_time

	max_idx = 0
	for i in range(0, len(python_strain[:])):
		if(python_strain[i][1] > python_strain[max_idx][1]):
			max_idx = i

	paths = glob.glob("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_strain_l[2-4]_m*.dat")
	for path in paths:
		python_strain = np.loadtxt(path)

		t = python_strain[:, 0]
		t = t.astype(np.complex_)
		h = python_strain[:, 1] + 1j * python_strain[:, 2]
		dh = np.zeros(len(t), dtype=np.complex_) 
		for i in range(0, len(t)-1):
			dh[i] = ((h[i+1] - h[i])/(t[i+1] - t[i]))
		dh[len(t)-1] = dh[len(t)-2]

		dh_conj = np.conj(dh)
		prod = np.multiply(dh, dh_conj)
		local_val = np.zeros(len(t))
		local_val = local_val.astype(np.complex_)
                # TODO: rewrite as array notation using numpy.cumtrapz
		for i in range(0, len(t)):
			local_val[i] = np.trapz(prod[:i], x=(t[:i]))
		val += local_val
		
	val *= 1/(16 * math.pi)
	np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_energy.dat", val)

#Get angular momentum
def get_angular_momentum(python_strain):
	"""
	Save the energy radiated angular momentum
	sim = string of simulation
	"""
	python_strain = np.loadtxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_strain_l2_m2.dat")
	val = np.zeros(len(python_strain))
	val = val.astype(np.complex_)
	cur_max_time = python_strain[0][0]
	cur_max_amp = abs(pow(python_strain[0][1], 2))
	# TODO: rewrite as array operations (use numpy.argmax)
	for i in python_strain[:]:
		cur_time = i[0]
		cur_amp = abs(pow(i[1], 2))
		if(cur_amp>cur_max_amp):
			cur_max_amp = cur_amp
			cur_max_time = cur_time

	max_idx = 0
	for i in range(0, len(python_strain[:])):
		if(python_strain[i][1] > python_strain[max_idx][1]):
			max_idx = i

	paths = glob.glob("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_strain_l[2-4]_m*.dat")
	for path in paths:
		python_strain = np.loadtxt(path)

		t = python_strain[:, 0]
		t = t.astype(np.complex_)
		h = python_strain[:, 1] + 1j * python_strain[:, 2]
		dh = np.zeros(len(t), dtype=np.complex_) 
                # TODO: rewrite using array notation
		for i in range(0, len(t)-1):
			dh[i] = ((h[i+1] - h[i])/(t[i+1] - t[i]))
		dh[len(t)-1] = dh[len(t)-2]

		dh_conj = np.conj(dh)
		prod = np.multiply(h, dh_conj)
		local_val = np.zeros(len(t))
		local_val = local_val.astype(np.complex_)
                # TODO: rewrite as array notation using numpy.cumtrapz. Move atoi call out of inner loop.
		for i in range(0, len(t)):
			local_val[i] = np.trapz(prod[:i], x=(t[:i])) * int(((path.split("_")[-1]).split("m")[-1]).split(".")[0])
		val += local_val
		
	val *= 1/(16 * math.pi)
	np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_radially_extrapolated_angular_momentum.dat", val)

#-----Main-----#

if __name__ == "__main__":
    #Initialize simulation data
    if(len(sys.argv) < 2):
            print("Pass in the number n of the n innermost detector radii to be used in the extrapolation (optional, default=all) and the simulation folders (e.g., ./power.py 6 ./simulations/J0040_N40 /path/to/my_simulation_folder).")
            sys.exit()
    elif(os.path.isdir(sys.argv[1])):
            radiiUsedForExtrapolation = 7	#use the first n radii available
            paths = sys.argv[1:]
    elif(not os.path.isdir(sys.argv[1])):
            radiiUsedForExtrapolation = int(sys.argv[1])	#use the first n radii available
            if(radiiUsedForExtrapolation < 1 or radiiUsedForExtrapolation > 7):
                    print("Invalid specified radii number")
                    sys.exit()
            paths = sys.argv[2:]

    for sim_path in paths:
            main_dir = sim_path
            sim = os.path.split(sim_path)[-1]

            simdirs = main_dir+"/output-????/%s/" % (sim)
            f0 = getCutoffFrequency(sim)

            #Check if necessary files exist
            par_file = main_dir+"/output-0000/%s.par" % (sim)
            two_punctures_file = main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim)
            if(not os.path.isfile(par_file) or not os.path.isfile(two_punctures_file)):
                    continue

            #Create data directories
            main_directory = "Extrapolated_Strain"
            sim_dir = main_directory+"/"+sim
            if not os.path.exists(main_directory):
                    os.makedirs(main_directory)
            if not os.path.exists(sim_dir):
                    os.makedirs(sim_dir)

            #Get ADMMass
            ADMMass = -1
            filename = main_dir+"/output-0000/%s/TwoPunctures.bbh" % (sim)
            with open(filename) as file:
                    contents = file.readlines()
                    for line in contents:
                            line_elems = line.split(" ")
                            if(line_elems[0] == "initial-ADM-energy"):
                                    ADMMass = float(line_elems[-1])

            # TODO: fix this. It will fail if output-0000 does not contain any mp
            # output and also will open the output files multiple times
            fn = sorted(glob.glob(simdirs+"mp_psi4.h5"))[0]
            with h5py.File(fn, "r") as fh:
                    # get all radii
                    radii = set()
                    modes = set()
                    dsets = dict()
                    for dset in fh:
                            # TODO: extend Multipole to save the radii as attributes and/or
                            # use a group structure in the hdf5 file
                            m = re.match(r'l(\d*)_m(-?\d*)_r(\d*\.\d)', dset)
                            if m:
                                    radius = float(m.group(3))
                                    mode = (int(m.group(1)), int(m.group(2)))
                                    modes.add(mode)
                                    radii.add(radius)
                                    dsets[(radius, mode)] = dset
                    modes = sorted(modes)
                    radii = sorted(radii)

            #Get Psi4
            for (l,m) in modes:
                    mp_psi4_vars = []
                    for radius in radii:
                            psi4dsetname = dsets[(radius, (l,m))]
                            mp_psi4 = loadHDF5Series(simdirs+"mp_psi4.h5", psi4dsetname)
                            mp_psi4_vars.append(mp_psi4)

                    #Get Tortoise Coordinate
                    tortoise = []
                    for radius in radii:
                            tortoise.append(-RadialToTortoise(radius, ADMMass))

                    strain = []
                    phase = []
                    amp = []
                    for i in range(len(radii)):
                            #Get modified Psi4 (Multiply real and imaginary psi4 columns by radii and add the tortoise coordinate to the time colum)
                            mp_psi4_vars[i][:, 0] += tortoise[i]
                            mp_psi4_vars[i][:, 1] *= radii[i]
                            mp_psi4_vars[i][:, 2] *= radii[i]

                            #Check for psi4 amplitude going to zero
                            cur_psi4_amp = np.sqrt(mp_psi4_vars[i][0, 1]**2 + mp_psi4_vars[i][0, 2]**2)
                            min_psi4_amp = cur_psi4_amp
                            # TODO: use array notatino for this since it finds the minimum amplitude
                            for j in range(0, len(mp_psi4_vars[i][:, 0])):
                                    cur_psi4_amp = np.sqrt(mp_psi4_vars[i][j, 1]**2 + mp_psi4_vars[i][j, 2]**2)
                                    if(cur_psi4_amp < min_psi4_amp):
                                            min_psi4_amp = cur_psi4_amp
                            if(min_psi4_amp < np.finfo(float).eps and l >= 2):
                                    print("The psi4 amplitude is near zero. The phase is ill-defined.")

                            #Fixed-frequency integration twice to get strain
                            hTable = psi4ToStrain(mp_psi4_vars[i], f0)
                            time = hTable[:, 0]
                            h = hTable[:, 1]
                            hplus = h.real
                            hcross = h.imag
                            newhTable = np.column_stack((time, hplus, hcross))
                            warnings.filterwarnings('ignore')
                            finalhTable = newhTable.astype(float)
                            np.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_strain_at_"+str(radii[i])+"_l"+str(l)+"_m"+str(m)+".dat", finalhTable)
                            strain.append(finalhTable)

                            #Get phase and amplitude of strain
                            h_phase = np.unwrap(np.angle(h))
                            angleTable = np.column_stack((time, h_phase))
                            angleTable = angleTable.astype(float)
                            phase.append(angleTable)
                            h_amp = np.absolute(h)
                            ampTable = np.column_stack((time, h_amp))
                            ampTable = ampTable.astype(float)
                            amp.append(ampTable)

                    #Interpolate phase and amplitude
                    t = phase[0][:, 0]
                    last_t = phase[radiiUsedForExtrapolation - 1][-1, 0]
                    last_index = 0;
                    # TODO: use array notation for this (this is a boolean
                    # plus a first_of or so)
                    for i in range(0, len(phase[0][:, 0])):
                            if(t[i] > last_t):
                                    last_index = i
                                    break
                    last_index = last_index-1
                    t = phase[0][0:last_index, 0]
                    dts = t[1:] - t[:-1]
                    dt = float(np.amin(dts))
                    t = np.arange(phase[0][0, 0], phase[0][last_index, 0], dt)
                    interpolation_order = 9
                    for i in range(0, radiiUsedForExtrapolation):
                            interp_function = scipy.interpolate.interp1d(phase[i][:, 0], phase[i][:, 1], kind=interpolation_order)
                            resampled_phase_vals = interp_function(t)
                            # try and keep all initial phases within 2pi of each other
                            if(i > 0):
                                phase_shift = round((resampled_phase_vals[0] - phase[0][0,1])/(2.*math.pi))*2.*math.pi
                                resampled_phase_vals -= phase_shift
                            phase[i] = np.column_stack((t, resampled_phase_vals))
                            interp_function = scipy.interpolate.interp1d(amp[i][:, 0], amp[i][:, 1], kind=interpolation_order)
                            resampled_amp_vals = interp_function(t)
                            amp[i] = np.column_stack((t, resampled_amp_vals))

                    #Extrapolate
                    phase_extrapolation_order = 1
                    amp_extrapolation_order = 2
                    radii = np.asarray(radii, dtype=float)
                    radii = radii[0:radiiUsedForExtrapolation]
                    # TODO: replace by np.ones (which is all it does anyway)
                    A_phase = np.power(radii, 0)
                    A_amp = np.power(radii, 0)
                    for i in range(1, phase_extrapolation_order+1):
                            A_phase = np.column_stack((A_phase, np.power(radii, -1*i)))

                    for i in range(1, amp_extrapolation_order+1):
                            A_amp = np.column_stack((A_amp, np.power(radii, -1*i)))

                    radially_extrapolated_phase = np.empty(0)
                    radially_extrapolated_amp = np.empty(0)
                    for i in range(0, len(t)):
                            b_phase = np.empty(0)
                            for j in range(0, radiiUsedForExtrapolation):
                                    b_phase = np.append(b_phase, phase[j][i, 1])
                            x_phase = np.linalg.lstsq(A_phase, b_phase)[0]
                            radially_extrapolated_phase = np.append(radially_extrapolated_phase, x_phase[0])

                            b_amp = np.empty(0)
                            for j in range(0, radiiUsedForExtrapolation):
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

            get_energy(sim)
            get_angular_momentum(sim)
