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

from ifxRadarSDK import *
from collections import namedtuple

def common_radar_device_config(params):
    range_resolution_m = params.range_resolution_m
    num_samples_per_chirp = params.num_samples_per_chirp
    max_speed_m_s = params.max_speed_m_s
    frame_repetition_time_s = params.frame_repetition_time_s
    sample_rate_Hz = params.sample_rate_Hz

    # ----------------------------------------------------------
    # Constants
    # ----------------------------------------------------------

    const_num_chirps_per_frame = 128
    const_speedlight_mps = 299792458
    const_samplingdelay_s = 0.000005
    const_center_frequency_Hz = 60750000000

    print("Samples per chirp = " + format(num_samples_per_chirp))
    print("Chirps per frame = " + format(const_num_chirps_per_frame))
    max_range_m = range_resolution_m*num_samples_per_chirp/2
    print ("Derived Maximum Range = " + format(max_range_m) + " meters")

    # Derive chirp frequency
    samplingtime_s = num_samples_per_chirp/sample_rate_Hz
    # increase the actual chirp time so the sampled frequency sweep corresponds to range resolution
    chirptime_s = samplingtime_s + const_samplingdelay_s
    increase_factor = chirptime_s/samplingtime_s

    bandwidth_Hz = (const_speedlight_mps/(2*range_resolution_m))*(increase_factor)

    lower_frequency_Hz = const_center_frequency_Hz-bandwidth_Hz/2
    upper_frequency_Hz = const_center_frequency_Hz+bandwidth_Hz/2
    #print ("Chirp Freq Band = " + format(lo_freq_khz) + " kHz Till " + format(hi_freq_khz) + " kHz")

    #Derive Pulse repetition time (PRT)
    chirp_repetition_time_s = const_speedlight_mps/(4*const_center_frequency_Hz*max_speed_m_s)
    # open device
    device = Device()

    # set device config
    device.set_config(
        sample_rate_Hz = sample_rate_Hz,
        rx_mask = 7,
        tx_mask = 1,
        tx_power_level = 31,
        if_gain_dB = 33,
        lower_frequency_Hz = lower_frequency_Hz,
        upper_frequency_Hz = upper_frequency_Hz,
        num_samples_per_chirp = num_samples_per_chirp,
        num_chirps_per_frame = const_num_chirps_per_frame,
        chirp_repetition_time_s = chirp_repetition_time_s,
        frame_repetition_time_s = frame_repetition_time_s,
        mimo_mode = "off")

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------
    metrictype = namedtuple('metrictype',['max_range_m',
                                    'num_samples_per_chirp',
                                    'num_chirps_per_frame'])
    metrics = metrictype(max_range_m = max_range_m,
                         num_samples_per_chirp = num_samples_per_chirp,
                         num_chirps_per_frame = const_num_chirps_per_frame)
    return device, metrics
