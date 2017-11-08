import platform
import ctypes

import six


class CUDADeviceProperties(ctypes.Structure):
    # See $CUDA_HOME/include/cuda_runtime_api.h for the definition of
    # the cudaDeviceProp structypes.
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture1DMipmap", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture2DMipmap", ctypes.c_int * 2),
        ("maxTexture2DLinear", ctypes.c_int * 3),
        ("maxTexture2DGather", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("maxTexture3DAlt", ctypes.c_int * 3),
        ("maxTextureCubemap", ctypes.c_int),
        ("maxTexture1DLayered", ctypes.c_int * 2),
        ("maxTexture2DLayered", ctypes.c_int * 3),
        ("maxTextureCubemapLayered", ctypes.c_int * 2),
        ("maxSurface1D", ctypes.c_int),
        ("maxSurface2D", ctypes.c_int * 2),
        ("maxSurface3D", ctypes.c_int * 3),
        ("maxSurface1DLayered", ctypes.c_int * 2),
        ("maxSurface2DLayered", ctypes.c_int * 3),
        ("maxSurfaceCubemap", ctypes.c_int),
        ("maxSurfaceCubemapLayered", ctypes.c_int * 2),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("streamPrioritiesSupported", ctypes.c_int),
        ("globalL1CacheSupported", ctypes.c_int),
        ("localL1CacheSupported", ctypes.c_int),
        ("sharedMemPerMultiprocessor", ctypes.c_size_t),
        ("regsPerMultiprocessor", ctypes.c_int),
        ("managedMemSupported", ctypes.c_int),
        ("isMultiGpuBoard", ctypes.c_int),
        ("multiGpuBoardGroupID", ctypes.c_int),
        # Pad with extra space to avoid dereference crashes if future
        # versions of CUDA extend the size of this structypes.
        ("__future_buffer", ctypes.c_char * 4096)
    ]


def get_installed_devices():
    try:
        libcudart = get_libcudart()

        device_count = ctypes.c_int()
        libcudart.cudaGetDeviceCount(ctypes.byref(device_count))

        return device_count.value
    except Exception:
        return 0


def get_memory(device):
    try:
        libcudart = get_libcudart()

        libcudart.cudaSetDevice(device)

        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        rc = libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0:
            return None

        return free.value, total.value
    except Exception:
        return None


def get_ordered_devices():
    """
    Default CUDA_DEVICE_ORDER is not compatible with nvidia-docker.
    Nvidia-Docker is using CUDA_DEVICE_ORDER=PCI_BUS_ID.

    https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation
    """

    libcudart = get_libcudart()

    devices = {}
    for i in range(0, get_installed_devices()):
        gpu = get_device_properties(i)

        pciBusId = ctypes.create_string_buffer(64)
        libcudart.cudaDeviceGetPCIBusId(ctypes.byref(pciBusId), 64, i)
        full_id = pciBusId.value.decode('utf-8')
        gpu['fullId'] = full_id
        gpu['id'] = i
        devices[full_id] = gpu

    ordered = []

    for key in sorted(devices):
        ordered.append(devices[key])

    return ordered


def get_device_properties(device, all=False):
    try:
        libcudart = get_libcudart()

        properties = CUDADeviceProperties()
        rc = libcudart.cudaGetDeviceProperties(ctypes.byref(properties), device)
        if rc != 0:
            return None
    except Exception:
        return None

    if all:
        values = {}
        for field in properties._fields_:
            values[field[0]] = getattr(properties, field[0])

            if isinstance(values[field[0]], six.binary_type):
                values[field[0]] = values[field[0]].decode('utf-8')

            if '_Array_' in type(values[field[0]]).__name__:
                values[field[0]] = [x for x in values[field[0]]]

        return values

    return {
        'device': device,
        'name': properties.name,
        'clock': properties.clockRate,
        'memory': properties.totalGlobalMem,
        'pciDomainID': properties.pciDomainID,
        'pciDeviceID': properties.pciDeviceID,
        'pciBusID': properties.pciBusID,
    }


def get_version():
    libcudart = get_libcudart()
    version = ctypes.c_int()
    libcudart.cudaRuntimeGetVersion(ctypes.byref(version))

    return version.value

def get_libcudart():
    system = platform.system()
    if system == "Linux":
        libcudart = ctypes.cdll.LoadLibrary("libcudart.so")
    elif system == "Darwin":
        libcudart = ctypes.cdll.LoadLibrary("libcudart.dylib")
    elif system == "Windows":
        libcudart = ctypes.windll.LoadLibrary("libcudart.dll")
    else:
        raise NotImplementedError("Cannot identify system.")

    version = ctypes.c_int()
    rc = libcudart.cudaRuntimeGetVersion(ctypes.byref(version))
    if rc != 0:
        raise ValueError("Could not get version")
    if version.value < 6050:
        raise NotImplementedError("CUDA version must be >= 6.5")

    return libcudart
