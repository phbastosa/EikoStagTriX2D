import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

parameters = sys.argv[1]

SPS_path = pyf.catch_parameter(parameters, "SPS")
RPS_path = pyf.catch_parameter(parameters, "RPS")
XPS_path = pyf.catch_parameter(parameters, "XPS")

SPS = np.loadtxt(SPS_path, delimiter = ",", dtype = np.float32) 
RPS = np.loadtxt(RPS_path, delimiter = ",", dtype = np.float32) 

nt = int(pyf.catch_parameter(parameters, "time_samples"))
dt = float(pyf.catch_parameter(parameters, "time_spacing"))

ns = len(SPS)
nr = len(RPS)

time = np.arange(nt)*dt

for sId in range(ns):

    data_file = f"../outputs/seismograms/triclinic_ssg_Ps_nStations{nr}_nSamples{nt}_shot_{sId+1}.bin"

    data = pyf.read_binary_matrix(nt, nr, data_file)

    data *= 1.0 / np.max(np.abs(data))

    travel_time = np.sqrt((SPS[sId,0] - RPS[:,0])**2 + 
                          (SPS[sId,1] - RPS[:,1])**2) / 1500

    tmute = np.array((travel_time + 0.15) / dt, dtype = int)

    for recId in range(nr):
        data[:tmute[recId], recId] *= 0.0        

    data[:300] *= 0.0

    output_path = f"../outputs/seismograms/seismogram_input_shot_{sId+1}.bin"
    
    data.flatten("F").astype(np.float32, order = "F").tofile(output_path)
