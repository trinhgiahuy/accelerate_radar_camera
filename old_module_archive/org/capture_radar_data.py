import sys
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

import bz2

def store_as_pickle_bz2(_path, payload,file_type):
    now1 = datetime.now()
    now=str(now1).replace(':', '-')
    payload["radar_config"]["date"]=str(now)
    ofile = bz2.BZ2File(_path+file_type, 'wb')  # FILE_FORMAT=pickle.bz2
    pickle.dump(payload, ofile)
    ofile.close()
def extract_bz2(_path):
    ifile = bz2.BZ2File(_path,'rb')
    pickle_data = pickle.load(ifile)
    ifile.close()
    return pickle_data
def store_as_json(_path,payload,file_type):
    f = open(storage+"/"+frameName+file_type, "a")
    f.write(json.dumps(payload))
    f.close()
# -----------------------------------------------------------
# Arguments
# -----------------------------------------------------------
parser = argparse.ArgumentParser(description="Displays range doppler map")

parser.add_argument('-n', '--nframes', type=int, default=10000, help="number of frames, default 25")
parser.add_argument('-f', '--frate', type=int, default=10, help="frame rate in Hz, default 5")

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
range_resolution_m=0.15 #4.99 meter
num_samples_per_chirp=64
max_speed_m_s = 3
max_range_m = range_resolution_m * num_samples_per_chirp * 0.25
params = paramtype(range_resolution_m = 0.15,#0.05
                   num_samples_per_chirp = 64,#64
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
_RadarName = "Distracted_Driver_ Arunav"
now = datetime.now()

storage=_RadarName+"_"+datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

if(path.exists(storage)!=True):
    os.mkdir(storage)
    print('=============================================================>',storage)

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

    
    now = datetime.now()
    now1=str(now).replace(':', '-')   
    
    msg={
        "RX1":rx1,
        "RX2":rx2,
        "RX3":rx3,
        "radar_config":{
            "numchirps":numchirps,
            "chirpsamples":chirpsamples,
            "date":now1,
            "max_speed_m_s": max_speed_m_s,
            "max_range_m": max_range_m,
            "range_resolution_m": range_resolution_m
        }
    }
    
    if(str(now.hour)==str(6)):
        print("shutdown")
        os.system("sudo shutdown -h now")
    frameName=_RadarName+"_"+str(now1)
    #print(frameName)
    # if(now.hour>=7 or now.hour<6):
    #     print(frameName)
    #     f = open(storage+"/"+frameName, "a")
    #     f.write(json.dumps(msg))
    #     f.close()
    store_as_pickle_bz2(storage+"/"+frameName,msg,file_type=".pkl.bz2")
    #store_as_json(storage+"/"+frameName,msg,file_type='.json')


