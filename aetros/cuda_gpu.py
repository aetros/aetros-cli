import platform
import ctypes


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
    except:
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
    except:
        return None


def get_device_properties(device):
    try:
        libcudart = get_libcudart()

        properties = CUDADeviceProperties()
        rc = libcudart.cudaGetDeviceProperties(ctypes.byref(properties), device)
        if rc != 0:
            return None
    except:
        return None

    return {
        'device': device,
        'name': properties.name,
        'clock': properties.clockRate,
        'memory': properties.totalGlobalMem,
        'pciDeviceID': properties.pciDeviceID,
        'pciBusID': properties.pciBusID,
    }


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
        raise NotImplementedError("CUDA version must be between >= 6.5")

    return libcudart
