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

"""Python wrapper for Infineon Radar SDK

The package expects the library (radar_sdk.dll on Windows, libradar_sdk.so on
Linux) either in the same directory as this file (ifxRadarSDK.py) or in a
subdirectory ../../libs/ARCH/ relative to this file where ARCH is depending on
the platform either win32_x86, win32_x64, raspi, or linux_x64.
"""

from ctypes import *
import platform
import platform, os, sys
import numpy as np

# by default,
#   from ifxRadarSDK import *
# would import all objects, including the ones from ctypes. To avoid name space
# pollution, we list what symbols should be exported.
__all__ = ["sdk_version", "sdk_min_version",
           "error_get", "error_clear", "Frame", "Device", "RadarSDKError",
           "get_version", "get_version_full"]

# version of sdk, will be initialized in initialize_module()
sdk_version = None

# minimum version of SDK required
sdk_min_version = "2.0.0"

# mapping of error code to the respective exception
error_mapping = {}

def check_version(version):
    """Check that version is at least py_sdk_min_ver"""
    major,minor,patch = version.split(".")
    min_major,min_minor,min_patch = sdk_min_version.split(".")

    if major > min_major:
        return True
    elif major < min_major:
        return False

    if minor > min_minor:
        return True
    elif minor < min_minor:
        return False

    if patch >= min_patch:
        return True
    elif patch < min_patch:
        return False

def find_library():
    """Find path to dll/shared object"""
    system = None
    libname = None
    if platform.system() == "Windows":
        libname = "radar_sdk.dll"
        is64bit = bool(sys.maxsize > 2**32)
        if is64bit:
            system = "win32_x64"
        else:
            system = "win32_x86"
    elif platform.system() == "Linux":
        libname = "libradar_sdk.so"
        machine = os.uname()[4]
        if machine == "x86_64":
            system = "linux_x64"
        elif machine == "armv7l":
            system = "raspi"
        elif machine == "aarch64":
            system = "linux_aarch64"

    if system == None or libname == None:
        raise RuntimeError("System not supported")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for reldir in (".", os.path.join("../../../libs/", system)):
        libpath = os.path.join(script_dir, reldir, libname)
        if os.path.isfile(libpath):
            return libpath

    raise RuntimeError("Cannot find " + libname)

# structs
class DeviceConfigStruct(Structure):
    """Wrapper for structure ifx_Device_Config_t"""
    _fields_ = (("sample_rate_Hz", c_uint32),
                ("rx_mask", c_uint32),
                ("tx_mask", c_uint32),
                ("tx_power_level", c_uint32),
                ("if_gain_dB", c_uint32),
                ("lower_frequency_Hz", c_uint64),
                ("upper_frequency_Hz", c_uint64),
                ("num_samples_per_chirp", c_uint32),
                ("num_chirps_per_frame", c_uint32),
                ("chirp_repetition_time_s", c_float),
                ("frame_repetition_time_s", c_float),
                ("mimo_mode", c_int))

class DeviceListEntry(Structure):
    """Wrapper for structure ifx_Device_Config_t"""
    _fields_ = (("board_type", c_int),
                ("shield_uuid", c_char*64))


class MatrixRStruct(Structure):
    _fields_ = (('d', POINTER(c_float)),
                ('rows', c_uint32),
                ('cols', c_uint32),
                ('lda', c_uint32, 31),
                ('owns_d', c_uint8, 1))


class FrameStruct(Structure):
    _fields_ = (('num_rx', c_uint8),
                ('rx_data', POINTER(POINTER(MatrixRStruct))))

FrameStructPointer = POINTER(FrameStruct)
MatrixRStructPointer = POINTER(MatrixRStruct)
DeviceConfigStructPointer = POINTER(DeviceConfigStruct)

def initialize_module():
    """Initialize the module and return ctypes handle"""
    dll = CDLL(find_library())

    dll.ifx_sdk_get_version_string.restype = c_char_p
    dll.ifx_sdk_get_version_string.argtypes = None

    dll.ifx_sdk_get_version_string_full.restype = c_char_p
    dll.ifx_sdk_get_version_string_full.argtypes = None

    sdk_version = dll.ifx_sdk_get_version_string().decode("ascii")
    if not check_version(sdk_version):
        # exception about non matching dll
        raise RuntimeError("radar SDK is version %s, but required at least %s" % (sdk_version, sdk_min_version))

    # error
    dll.ifx_error_to_string.restype = c_char_p
    dll.ifx_error_to_string.argtypes = [c_int]

    dll.ifx_error_clear.restype = None
    dll.ifx_error_clear.argtypes = None

    dll.ifx_error_get.restype = c_int
    dll.ifx_error_get.argtypes = None

    # device
    dll.ifx_device_create.restype = c_void_p
    dll.ifx_device_create.argtypes = None

    dll.ifx_device_create_by_port.restype = c_void_p
    dll.ifx_device_create_by_port.argtypes = [c_char_p]

    dll.ifx_device_get_list.restype = c_void_p
    dll.ifx_device_get_list.argtypes = None

    dll.ifx_device_get_list_by_shield_type.restype = c_void_p
    dll.ifx_device_get_list_by_shield_type.argtypes = [c_int]

    dll.ifx_device_create_by_uuid.restype = c_void_p
    dll.ifx_device_create_by_uuid.argtypes = [c_char_p]

    dll.ifx_device_get_shield_uuid.restype = c_char_p
    dll.ifx_device_get_shield_uuid.argtypes = [c_void_p]

    dll.ifx_device_set_config.restype = None
    dll.ifx_device_set_config.argtypes = [c_void_p, DeviceConfigStructPointer]

    dll.ifx_device_get_config.restype = None
    dll.ifx_device_get_config.argtypes = [c_void_p, DeviceConfigStructPointer]

    dll.ifx_device_destroy.restype = None
    dll.ifx_device_destroy.argtypes = [c_void_p]

    dll.ifx_device_create_frame_from_device_handle.restype = FrameStructPointer
    dll.ifx_device_create_frame_from_device_handle.argtypes = [c_void_p]

    dll.ifx_device_get_next_frame.restype = c_int
    dll.ifx_device_get_next_frame.argtypes = [c_void_p , FrameStructPointer]

    dll.ifx_device_get_next_frame_timeout.restype = c_int
    dll.ifx_device_get_next_frame_timeout.argtypes = [c_void_p , FrameStructPointer, c_uint16]

    dll.ifx_device_get_temperature.restype = None
    dll.ifx_device_get_temperature.argtypes = [c_void_p , POINTER(c_float)]

    # frame
    dll.ifx_frame_create_r.restype = FrameStructPointer
    dll.ifx_frame_create_r.argtypes = [c_uint8, c_uint32, c_uint32]

    dll.ifx_frame_destroy_r.restype = None
    dll.ifx_frame_destroy_r.argtypes = [FrameStructPointer]

    dll.ifx_frame_get_mat_from_antenna_r.restype = MatrixRStructPointer
    dll.ifx_frame_get_mat_from_antenna_r.argtypes = [FrameStructPointer, c_uint8]

    # list
    dll.ifx_list_destroy.restype = None
    dll.ifx_list_destroy.argtypes = [c_void_p]

    dll.ifx_list_size.restype = c_size_t
    dll.ifx_list_size.argtypes = [c_void_p]

    dll.ifx_list_get.restype = c_void_p
    dll.ifx_list_get.argtypes = [c_void_p, c_size_t]

    error_api_base = 0x00010000
    error_dev_base = 0x00011000
    error_app_base = 0x00020000

    # list of all
    errors = {
        error_api_base+0x01: "RadarSDKArgumentNullError",
        error_api_base+0x02: "RadarSDKArgumentInvalidError",
        error_api_base+0x03: "RadarSDKArgumentOutOfBoundsError",
        error_api_base+0x04: "RadarSDKArgumentInvalidExpectedRealError",
        error_api_base+0x05: "RadarSDKArgumentInvalidExpectedComplexError",
        error_api_base+0x06: "RadarSDKIndexOutOfBoundsError",
        error_api_base+0x07: "RadarSDKDimensionMismatchError",
        error_api_base+0x08: "RadarSDKMemoryAllocationFailedError",
        error_api_base+0x09: "RadarSDKInplaceCalculationNotSupportedError",
        error_api_base+0x0A: "RadarSDKMatrixSingularError",
        error_api_base+0x0B: "RadarSDKMatrixNotPositiveDefinitieError",
        error_api_base+0x0C: "RadarSDKNotSupportedError",
        # device related errors
        error_dev_base+0x00: "RadarSDKNoDeviceError",
        error_dev_base+0x01: "RadarSDKDeviceBusyError",
        error_dev_base+0x02: "RadarSDKCommunicationError",
        error_dev_base+0x03: "RadarSDKNumSamplesOutOfRangeError",
        error_dev_base+0x04: "RadarSDKRxAntennaCombinationNotAllowedError",
        error_dev_base+0x05: "RadarSDKIfGainOutOfRangeError",
        error_dev_base+0x06: "RadarSDKSamplerateOutOfRangeError",
        error_dev_base+0x07: "RadarSDKRfOutOfRangeError",
        error_dev_base+0x08: "RadarSDKTxPowerOutOfRangeError",
        error_dev_base+0x09: "RadarSDKChirpRateOutOfRangeError",
        error_dev_base+0x0a: "RadarSDKFrameRateOutOfRangeError",
        error_dev_base+0x0b: "RadarSDKNumChirpsNotAllowedError",
        error_dev_base+0x0C: "RadarSDKFrameSizeNotSupportedError",
        error_dev_base+0x0D: "RadarSDKTimeoutError",
        error_dev_base+0x0E: "RadarSDKFifoOverflowError",
        error_dev_base+0x0F: "RadarSDKTxAntennaModeNotAllowedError",
        error_dev_base+0x10: "RadarSDKFirmwareVersionNotSupported",
        error_dev_base+0x10: "RadarSDKDeviceNotSupported"
    }

    for errcode, name in errors.items():
        descr = dll.ifx_error_to_string(errcode).decode("ascii")

        # dynamically generate the class in the modules global scope
        pycode = """
global %s
class %s(RadarSDKError):
    '''%s'''
    def __init__(self):
        super().__init__(%d)
        """ % (name,name,descr,errcode)
        exec(pycode)

        # export the error class
        __all__.append(name)

        # add the class to the list of exceptions
        error_mapping[errcode] = eval(name)

    return dll

class RadarSDKError(Exception):
    def __init__(self, error):
        """Create new RadarSDKException with error code given by error"""
        self.error = error
        self.message = dll.ifx_error_to_string(error).decode("ascii")

    def __str__(self):
        """Exception message"""
        return self.message

dll = initialize_module()


def error_get():
    """Get last SDK error"""
    return dll.ifx_error_get()

def error_clear():
    """Clear SDK error"""
    dll.ifx_error_clear()


def get_version():
    """Return SDK version string (excluding git tag from which it was build)"""
    return dll.ifx_sdk_get_version_string().decode("ascii")

def get_version_full():
    """Return full SDK version string including git tag from which it was build"""
    return dll.ifx_sdk_get_version_string_full().decode("ascii")


def check_rc(error_code=None):
    """Raise an exception if error_code is not IFX_OK (0)"""
    if error_code == None:
        error_code = dll.ifx_error_get()
        error_clear()
    if error_code:
        if error_code in error_mapping:
            raise error_mapping[error_code]()
        else:
            raise RadarSDKError(error_code)


class Frame():
    def __init__(self, num_antennas, num_chirps_per_frame, num_samples_per_chirp):
        """Create frame for time domain data acquisition

        This function initializes a data structure that can hold a time domain
        data frame according to the dimensions provided as parameters.

        If a device is connected then the method Device.create_frame_from_device_handle
        can be used instead of this function, as that function reads the
        dimensions from configured the device handle.

        Parameters:
            num_antennas            Number of virtual active Rx antennas configured in the device
            num_chirps_per_frame    Number of chirps configured in a frame
            num_samples_per_chirp   Number of chirps configured in a frame
        """
        self.handle = dll.ifx_frame_create_r(num_antennas, num_chirps_per_frame, num_samples_per_chirp)
        check_rc()

    @classmethod
    def create_from_pointer(cls, framepointer):
        """Create Frame from FramePointer"""
        self = cls.__new__(cls)
        self.handle = framepointer
        return self

    def __del__(self):
        """Destroy frame handle"""
        if hasattr(self, "handle"):
            dll.ifx_frame_destroy_r(self.handle)

    def get_num_rx(self):
        """Return the number of virtual active Rx antennas in the radar device"""
        return self.handle.contents.num_rx

    def get_mat_from_antenna(self, antenna, copy=True):
        """Get matrix from antenna

        If copy is True, a copy of the original matrix is returned. If copy is
        False, the matrix is not copied and the matrix must *not* be used after
        the frame object has been destroyed.

        Parameters:
            antenna     number of antenna
            copy        if True a copy of the matrix will be returned
        """
        # we don't have to free mat because the matrix is saved in the frame
        # handle.
        # matrices are in C order (row major order)
        mat = dll.ifx_frame_get_mat_from_antenna_r(self.handle, antenna)
        d = mat.contents.d
        shape = (mat.contents.rows, mat.contents.cols)
        print(shape)
        return np.array(np.ctypeslib.as_array(d, shape), order="C", copy=copy)


class Device():
    @staticmethod
    def get_list(board_type="any"):
        """Return a list of com ports

        The function returns a list of unique ids (uuids) that correspond to
        radar devices. The Shield type can be optionally specified.
		The entries of the returned list can be used to open the
        radar device. Currently supported board types are:
			#. "bgt60tr13"
			#. "atr24"

        **Examples**
            for uuid in Device.get_list(): #scans all types of radar devices
                dev = Device(uuid)
                # ...
			for uuid in Device.get_list("bgt60tr13"): #scans all devices with specified shield attached

        """
        if board_type.lower() == "bgt60tr13":
            shield_type = 0x200
        elif board_type.lower() == "atr24":
            shield_type = 0x201
        else:
            shield_type = 0xffff # this corresponds to any shield type
        uuids = []

        ifx_list = dll.ifx_device_get_list_by_shield_type(shield_type)
        size = dll.ifx_list_size(ifx_list)
        for i in range(size):
            p = dll.ifx_list_get(ifx_list, i)
            entry = cast(p, POINTER(DeviceListEntry))
            uuids.append(entry.contents.shield_uuid.decode("ascii"))
        dll.ifx_list_destroy(ifx_list)

        return uuids

    def __init__(self, uuid=None, port=None):
        """Create new device

        Search for a Infineon radar sensor device connected to the host machine
        and connects to the first found sensor device.

        The device is automatically closed by the destructor. If you want to
        close the device yourself, you can use the keyword del:
            device = Device()
            # do something with device
            ...
            # close device
            del device

        If port is given, the specific port is opened. If uuid is given and
        port is not given, the radar device with the given uuid is opened. If
        no parameters are given, the first found radar device will be opened.

        Examples:
          - Open first found radar device:
            dev = Device()
          - Open radar device on COM5:
            dev = Device(port="COM5")
          - Open radar device with uuid 0123456789abcdef0123456789abcdef
            dev = Device(uuid="0123456789abcdef0123456789abcdef")

        Optional parameters:
            port:       opens the given port
            uuid:       open the radar device with unique id given by uuid
                        the uuid is represented as a 32 character string of
                        hexadecimal characters. In addition, the uuid may
                        contain dash characters (-) which will be ignored.
                        Both examples are valid and correspond to the same
                        uuid:
                            0123456789abcdef0123456789abcdef
                            01234567-89ab-cdef-0123-456789abcdef
        """
        if uuid:
            self.handle = dll.ifx_device_create_by_uuid(uuid.encode("ascii"))
        elif port:
            self.handle = dll.ifx_device_create_by_port(port.encode("ascii"))
        else:
            self.handle = dll.ifx_device_create()

        # check return code
        check_rc()


    def set_config(self,
               sample_rate_Hz = 1e6,
               rx_mask = 7,
               tx_mask = 1,
               tx_power_level = 31,
               if_gain_dB = 33,
               lower_frequency_Hz = 58e9,
               upper_frequency_Hz = 63e9,
               num_samples_per_chirp = 128,
               num_chirps_per_frame = 32,
               chirp_repetition_time_s = 5e-4,
               frame_repetition_time_s = 0.1,
               mimo_mode = "off"):
        """Configure device and start acquisition of time domain data

        The board is configured according to the parameters provided
        through config and acquisition of time domain data is started.

        Parameters:
            sample_rate_Hz:
                Sampling rate of the ADC used to acquire the samples during a
                chirp. The duration of a single chirp depends on the number of
                samples and the sampling rate.

            rx_mask:
                Bitmask where each bit represents one RX antenna of the radar
                device. If a bit is set the according RX antenna is enabled
                during the chirps and the signal received through that antenna
                is captured. The least significant bit corresponds to antenna
                1.

            tx_mask:
                Bitmask where each bit represents one TX antenna. Analogous to
                rx_mask.

            tx_power_level:
                This value controls the power of the transmitted RX signal.
                This is an abstract value between 0 and 31 without any physical
                meaning.

            if_gain_dB:
                Amplification factor that is applied to the IF signal coming
                from the RF mixer before it is fed into the ADC.

            lower_frequency_Hz:
                Lower frequency (start frequency) of the FMCW chirp.

            upper_frequency_Hz:
                Upper frequency (stop frequency) of the FMCW chirp.

            num_samples_per_chirp:
                This is the number of samples acquired during each chirp of a
                frame. The duration of a single chirp depends on the number of
                samples and the sampling rate.

            num_chirps_per_frame:
                This is the number of chirps a single data frame consists of.

            chirp_repetition_time_s:
                This is the time period that elapses between the beginnings of
                two consecutive chirps in a frame. (Also commonly referred to as
                pulse repetition time or chirp-to-chirp time.)

            frame_repetition_time_s:
                This is the time period that elapses between the beginnings of
                two consecutive frames. The reciprocal of this parameter is the
                frame rate. (Also commonly referred to as frame time or frame
                period.)

            mimo_mode:
                Mode of MIMO. Allowed values are "tdm" for
                time-domain-multiplexed MIMO or "off" for MIMO deactivated.
        """
        if mimo_mode.lower() == "tdm":
            mimo_mode = 1
        else:
            mimo_mode = 0

        config = DeviceConfigStruct(int(sample_rate_Hz),
                                    rx_mask,
                                    tx_mask,
                                    tx_power_level,
                                    if_gain_dB,
                                    int(lower_frequency_Hz),
                                    int(upper_frequency_Hz),
                                    num_samples_per_chirp,
                                    num_chirps_per_frame,
                                    chirp_repetition_time_s,
                                    frame_repetition_time_s,
                                    mimo_mode)
        dll.ifx_device_set_config(self.handle, byref(config))
        check_rc()

    def get_config(self):
        """Get the configuration from the device"""
        config = DeviceConfigStruct()
        dll.ifx_device_get_config(self.handle, byref(config))
        check_rc()
        # return struct as dictionary
        return dict((field, getattr(config, field)) for field, _ in config._fields_)

    def get_next_frame(self, frame, timeout_ms=None):
        """Retrieve next frame of time domain data from device

        Retrieve the next complete frame of time domain data from the connected
        device. The samples from all chirps and all enabled RX antennas will be
        copied to the provided data structure frame.

        If timeout_ms is given, an IFX_ERROR_TIMEOUT exception is thrown if a
        complete frame is not given within timeout_ms miliseconds.
        """
        if timeout_ms:
            ret = dll.ifx_device_get_next_frame_timeout(self.handle, frame.handle, timeout_ms)
        else:
            ret = dll.ifx_device_get_next_frame(self.handle, frame.handle)
        check_rc(ret)

    def create_frame_from_device_handle(self):
        """Create frame for time domain data acquisition

        This method checks the current configuration of the specified sensor
        device and initializes a data structure that can hold a time domain
        data frame according acquired through that device.
        """
        frame_p = dll.ifx_device_create_frame_from_device_handle(self.handle)
        check_rc()
        return Frame.create_from_pointer(frame_p)

    def get_shield_uuid(self):
        """Get the unique id for the radar shield"""
        c_uuid = dll.ifx_device_get_shield_uuid(self.handle)
        check_rc()
        return c_uuid.decode("utf-8")

    def get_temperature(self):
        """Get the temperature of the device in degrees Celsius"""
        temperature = c_float(0)
        dll.ifx_device_get_temperature(self.handle, pointer(temperature))
        check_rc()
        return temperature.value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        """Destroy device handle"""
        if hasattr(self, "handle") and self.handle:
            dll.ifx_device_destroy(self.handle)
            self.handle = None
