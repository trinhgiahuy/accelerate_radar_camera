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


def store_as_pickle_bz2(_path, payload, file_type):
    ofile = bz2.BZ2File(_path + file_type, 'wb')  # FILE_FORMAT=pickle.bz2
    pickle.dump(payload, ofile)
    ofile.close()


def extract_bz2(_path):
    ifile = bz2.BZ2File(_path, 'rb')
    pickle_data = pickle.load(ifile)
    ifile.close()
    return pickle_data


def store_as_json(_path, payload, file_type):
    f = open(storage + "/" + frameName + file_type, "a")
    f.write(json.dumps(payload))
    f.close()


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
paramtype = namedtuple('paramtype', ['range_resolution_m',
                                     'num_samples_per_chirp',
                                     'max_speed_m_s',
                                     'sample_rate_Hz',
                                     'frame_repetition_time_s'])

# ----------------------------------------------------------------
# --------------------- Radar paramaters -------------------------
# ----------------------------------------------------------------
range_resolution_m = 0.1  # 4.99 meter
max_speed_m_s =3

params = paramtype(range_resolution_m=0.075,
                   num_samples_per_chirp=64,
                   max_speed_m_s=3,
                   sample_rate_Hz=1000000,
                   frame_repetition_time_s=1 / args.frate)

# configure and open device
device, metrics = common_radar_device_config(params)

# create frame
frame = device.create_frame_from_device_handle()

# number of virtual active receiving antennas
num_rx = frame.get_num_rx()

numchirps = metrics.num_chirps_per_frame
chirpsamples = metrics.num_samples_per_chirp
first_run=True

# ----------------------------------------------------------------
# ----------------------- Main Routine ---------------------------
# ----------------------------------------------------------------
# A loop for fetching and processing a finite number of frames

# _RadarName = "Montreal-PT62-Stand-Back-Tilted
_RadarName = "Distracted_Driver_ Arunav"
now = datetime.now()

storage = _RadarName + "_" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

if (path.exists(storage) != True):
    os.mkdir(storage)
    #print(storage)

encoding = 'utf-8'
while True:
    try:
        device.get_next_frame(frame)
    except RadarSDKFifoOverflowError:
        print("Fifo Overflow")
        # exit(1)

    frame0 = {}
    frame1 = {}
    frame2 = {}

    if first_run == True:
        first_run = False
        #max_speed_m_s =6
        max_range_m = metrics.max_range_m
        #range_resolution_m =0.07
        #print(numchirps)
        #print(chirpsamples)
        #print(max_range_m)
        #print(range_resolution_m)

        # ----------------------------------------------------------------
        # ------------------- One Time Processing ------------------------
        # ----------------------------------------------------------------
        fall_vel_threshold = 0.7
        moving_avg_alpha = 0.6
        mti_alpha = 1.0
        count_vel = 0
        det_count = 0
        f_count = 0
        abnorm_count = 0
        absence_count = 0
        Present_count = 0
        check_count = 0
        Pos_count = 0
        vel_point = np.linspace(-max_speed_m_s, max_speed_m_s, numchirps * 2)
        vel_sum = np.zeros((numchirps * 2, 1), dtype=complex)
        vel_old = np.zeros((numchirps * 2, 1), dtype=complex)
        vel_itemindex = np.where(np.abs(np.array(vel_point)) > fall_vel_threshold)
        vel_filt_old = vel_old[vel_itemindex]

        Range_Dopp_prof = []
        anglelen = numchirps * 2
        philen = numchirps * 2
        theta_vec = np.linspace(-np.pi / 2, np.pi / 2, anglelen)
        phi_vec = np.linspace(-np.pi / 2, np.pi / 2, philen)
        y_spec = np.zeros(int(len(theta_vec)), dtype=complex)
        z_spec = np.zeros(int(len(phi_vec)), dtype=complex)
        dim = 2
        num_rx = 3

        RngAzMat = np.zeros((int(chirpsamples / 2), int(len(theta_vec))), dtype=complex)
        RngElMat = np.zeros((int(chirpsamples / 2), int(len(phi_vec))), dtype=complex)
        Rang_Prof_Az = np.zeros((int(chirpsamples / 2), int(len(phi_vec)), int(dim)), dtype=complex)
        Rang_Prof_Elve = np.zeros((int(chirpsamples / 2), int(len(phi_vec)), int(dim)), dtype=complex)
        RngElMat = np.zeros((int(chirpsamples / 2), int(len(phi_vec))), dtype=complex)

        RngAzMat_sum = np.zeros((int(chirpsamples / 2), int(len(theta_vec))), dtype=complex)
        RngElMat_sum = np.zeros((int(chirpsamples / 2), int(len(theta_vec))), dtype=complex)
        RngDopp_sum = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)
        RngAzpoint1 = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)
        RngAzpoint2 = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)
        RngElpoint1 = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)
        RngElpoint2 = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)

        dist_points = np.linspace(0, max_range_m / 2, int(chirpsamples / 2))
        a_thetta = np.exp(-1j * np.pi * np.sin(theta_vec))
        # ----------------------------------------------------------------
        # ------------------- One Time Processing ------------------------
        # ----------------------------------------------------------------
        # compute Blackman-Harris Window matrix over chirp samples(range)
        range_window = signal.blackmanharris(chirpsamples).reshape(1, chirpsamples)
        # compute Blackman-Harris Window matrix over number of chirps(velocity)
        doppler_window = signal.blackmanharris(numchirps).reshape(1, numchirps)

        # initialize doppler averages for all antennae
        dopp_avg = np.zeros((chirpsamples // 2, numchirps * 2, num_rx), dtype=complex)

        dopp_appnd = []
        Rng_appnd = []
        Az_appnd = []
        Elv_appnd = []
        stft_const = []
        velocity = []
        rng_cent = []
        az_cent = []
        cnt_cent = []
        cnt_az_cent = []
        position_rng = []
        position_az = []

        SFD = False

        rng_old = np.zeros((1, chirpsamples), dtype=complex)
        RngDopp_sum = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)
    fft_2 = np.zeros((int(chirpsamples / 2), numchirps * 2), dtype=complex)



    for iAnt in range(0, 3):

        mat = frame.get_mat_from_antenna(iAnt)
        #mat = data[ele]

        # ----------------------------------------------------------------
        # ----------------------- Main Routine ---------------------------
        # ----------------------------------------------------------------
        # A loop for fetching and processing a finite number of frames
        avgs = np.average(mat, 1).reshape(numchirps, 1)

        # de-bias values
        mat = mat - avgs

        # -------------------------------------------------
        # Step 2 - Windowing the Data
        # -------------------------------------------------
        mat = np.multiply(mat, range_window)

        # -------------------------------------------------
        # Step 3 - add zero padding here
        # -------------------------------------------------
        zp1 = np.pad(
            mat, ((0, 0), (0, chirpsamples)), 'constant')


        # -------------------------------------------------
        # Step 4 - Compute FFT for distance information
        # -------------------------------------------------
        range_fft = np.fft.fft(zp1) / chirpsamples
        del zp1
        # ignore the redundant info in negative spectrum
        # compensate energy by doubling magnitude
        range_fft = 2 * range_fft[:,
                        range(int(chirpsamples / 2))]

        # # prepare for dopplerfft
        # ------------------------------------------------
        # Transpose
        # Distance is now indicated on y axis
        # ------------------------------------------------
        fft1d = range_fft + 0
        # fft1d[0:skip] = 0
        fft1d = np.transpose(fft1d)

        # -------------------------------------------------
        # Step 7 - Windowing the Data in doppler
        # -------------------------------------------------
        fft1d = np.multiply(fft1d, doppler_window)

        zp2 = np.pad(
            fft1d, ((0, 0), (0, numchirps)), 'constant')

        fft2d = np.fft.fft(zp2) / numchirps
        # update moving average
        dopp_avg[:, :, iAnt] = (
                                       fft2d * moving_avg_alpha) + (
                                       dopp_avg[:, :, iAnt] * (1 - moving_avg_alpha))
        # MTI processing
        # needed to remove static objects
        # step 1 moving average
        # multiply history by (mti_alpha)
        # mti_alpha=0
        fft2d_mti = fft2d - (dopp_avg[:, :, iAnt] * mti_alpha)

        # re-arrange fft result for zero speed at centre
        dopplerfft = np.fft.fftshift(fft2d_mti, (1,))
        Range_Dopp_prof.append(dopplerfft)  # appending to range profile for azimuth and elevation
        fft_2 = fft_2 + dopplerfft  # integration over channels
        # print(fft_2)
        # ---------------Plot Range Doppler Map-----------------------
        #
        # plot3 = plt.figure("RngDoppMat")
        # plt.imshow(abs(fft_2), cmap='hot',
        #            extent=(-max_speed_m_s, max_speed_m_s,
        #                    0, max_range_m / 2),
        #            origin='lower')
        # plt.xlabel("velocity (m/s)")
        # plt.ylabel("distance (m)")
        # plt.title(data["radar_config"]['date'])
        # plt.draw()
        # plt.pause(1e-2)
        # plt.cla()
        # ---------------Capon Beamformer-----------------------
        # # for azimuth
    Rang_Prof_Az[:, :, 0] = Range_Dopp_prof[0]
    Rang_Prof_Az[:, :, 1] = Range_Dopp_prof[2]
    RangeMatrix_her = 1 / numchirps * np.conjugate(Rang_Prof_Az.transpose([0, 2, 1]))

    for rr in range(0, int(chirpsamples / 2)):
        #
        inv_R_hat = np.linalg.inv(RangeMatrix_her[rr, ...] @ Rang_Prof_Az[rr, ...])  # for azimuth

        for jj in range(len(theta_vec)):
            a_hat = np.array([1, a_thetta[jj]])
            y_spec[jj] = 1 / (a_hat.conjugate().transpose() @ inv_R_hat @ a_hat)

        RngAzMat[rr, :] = y_spec  # range-Azimuth map

    # ---------------Plot Range Doppler Map-----------------------
    plt.figure(1)
    plt.clf()
    plt.cla()
    plt.subplot(211)
    ## plot3 = plt.figure("RngDoppMat")
    plt.imshow(abs(fft_2), cmap='hot',
                extent=(-max_speed_m_s, max_speed_m_s,
                        0, max_range_m / 2),
                origin='lower')
    # cbar = plt.colorbar(h)
    plt.xlabel("velocity (m/s)")
    plt.ylabel("distance (m)")
    #plt.title(data["radar_config"]['date'])
    plt.subplot(212)
    plt.pcolormesh(theta_vec * 180 / (np.pi), dist_points, np.abs(RngAzMat), alpha=None,
                    norm=None,
                    cmap=None, shading=None, antialiased=None)

    plt.xlabel("Azimuth Angle [deg]")
    plt.ylabel("Range [m]")
    # plt.title(data["radar_config"]['date'])
    # plt.draw()
    plt.pause(1e-2)
    plt.draw()
    Range_Dopp_prof = []

    # lists = mat.tolist()
    # json_str = json.dumps(lists)

    # if (iAnt == 0):
    #     rx1 = lists
    # if (iAnt == 1):
    #     rx2 = lists
    # if (iAnt == 2):
    #     rx3 = lists

    # msg = {
    #     "RX1": rx1,
    #     "RX2": rx2,
    #     "RX3": rx3,
    #     "numchirps": numchirps,
    #     "chirpsamples": chirpsamples
    # }

    # now = datetime.now()
    # now1 = str(now).replace(':', '-')
    # if (str(now.hour) == str(6)):
    #     print("shutdown")
    #     os.system("sudo shutdown -h now")
    # frameName = _RadarName + "_" + str(now1)
    # print(frameName)
    # if(now.hour>=7 or now.hour<6):
    #     print(frameName)
    #     f = open(storage+"/"+frameName, "a")
    #     f.write(json.dumps(msg))
    #     f.close()
    # store_as_pickle_bz2(storage+"/"+frameName,msg,file_type=".pkl.bz2")
    # store_as_json(storage + "/" + frameName, msg, file_type='.json')


