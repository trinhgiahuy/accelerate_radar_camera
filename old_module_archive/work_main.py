# ===========================================================================
# Copyright (C) 2021 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import argparse
import matplotlib.pyplot as plt
import numpy as np

from ifxRadarSDK import *
from scipy import signal
from radar_cfg import *
from collections import namedtuple

from datetime import datetime
import json
import os.path
from os import path
import pickle




# -----------------------------------------------------------
# Arguments
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Displays range doppler map")

parser.add_argument('-n', '--nframes', type=int, default=10000, help="number of frames, default 25")
parser.add_argument('-f', '--frate', type=int, default=5, help="frame rate in Hz, default 5")

args = parser.parse_args()

# -----------------------------------------------------------
# Device configuration
# -----------------------------------------------------------
paramtype = namedtuple('paramtype',['range_resolution_m',
                                    'num_samples_per_chirp',
                                    'max_speed_m_s',
                                    'sample_rate_Hz',
                                    'frame_repetition_time_s'])

#----------------------------------------------------------------
#--------------------- Radar paramaters -------------------------
#----------------------------------------------------------------
range_resolution_m=0.17 #4.99 meter
num_samples_per_chirp=128
params = paramtype(range_resolution_m = 0.17,
                   num_samples_per_chirp = 128,
                   max_speed_m_s = 3,
                   sample_rate_Hz = 1000000,
                   frame_repetition_time_s = 1/args.frate)


# configure and open device
device, metrics = common_radar_device_config(params)


# create frame
frame = device.create_frame_from_device_handle()

# number of virtual active receiving antennas
num_rx = frame.get_num_rx()

numchirps = metrics.num_chirps_per_frame
chirpsamples = metrics.num_samples_per_chirp

#----------------------------------------------------------------
#----------------------- Main Routine ---------------------------
#----------------------------------------------------------------
# A loop for fetching and processing a finite number of frames

#_RadarName = "Montreal-PT62-Stand-Back-Tilted
_RadarName = "MUHC-PT54-Tilt-under-the-bed-45cm"
now = datetime.now()

storage=datetime.today().strftime('%Y-%m-%d')+"_"+_RadarName

if(path.exists(storage)!=True):
    os.mkdir(storage)
    print(storage)

encoding = 'utf-8'
while True:
    try:
        device.get_next_frame(frame)
    except RadarSDKFifoOverflowError:
        print("Fifo Overflow")
        #exit(1)

    frame0={}
    frame1={}
    frame2={}
    for iAnt in range(0,3): 

        mat = frame.get_mat_from_antenna(iAnt)
        
        lists = mat.tolist()
        json_str = json.dumps(lists)
        
        if(iAnt==0):
            rx1=lists
        if(iAnt==1):
            rx2=lists
        if(iAnt==2):
            rx3=lists
       
    
    msg={
        "RX1":rx1,
        "RX2":rx2,
        "RX3":rx3,
        "numchirps":numchirps,
        "chirpsamples":chirpsamples     
    }
    
    now = datetime.now()
    now1=str(now).replace(':', '-')
    if(str(now.hour)==str(6)):
        print("shutdown")
        os.system("sudo shutdown -h now")
    frameName=_RadarName+"_"+str(now1)+".json"
    print(frameName)
        
    geeky_file = open("ahmad.pickle", 'wb')
    pickle.dump(msg, geeky_file)    
    geeky_file.close()
