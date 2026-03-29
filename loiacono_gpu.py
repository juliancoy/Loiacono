import os
import sys
import time
from scipy.signal import get_window

gpuhere = os.path.dirname(os.path.abspath(__file__))

try:
    import pkg_resources
    if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
        sys.path = [os.path.join(gpuhere, "..", "vulkanese")] + sys.path
except ImportError:
    sys.path = [os.path.join(gpuhere, "..", "vulkanese")] + sys.path

import vulkanese as ve
from loiacono import *
import vulkan as vk

loiacono_home = os.path.dirname(os.path.abspath(__file__))

# Create a compute shader
class Loiacono_GPU(ve.shader.Shader):
    def __init__(
        self,
        device,
        fprime,
        multiple,
        signalLength=2**15,
        constantsDict={},
        DEBUG=False,
        buffType="float",
        memProperties=0
        | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    ):

        constantsDict["multiple"] = multiple
        constantsDict["SIGNAL_LENGTH"] = signalLength
        constantsDict["PROCTYPE"] = buffType
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        constantsDict["windowed"] = 0
        self.signalLength = signalLength

        self.instance = device.instance
        self.device = device
        self.constantsDict = constantsDict
        self.spectrum = np.zeros((len(fprime)))

        # Buffers: x (input signal), L (output spectrum), f (frequencies), offset
        buffers = [
            ve.buffer.StorageBuffer(
                device=self.device,
                name="x",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[signalLength],
                memProperties=memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="L",
                memtype=buffType,
                qualifier="writeonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="f",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="offset",
                memtype="uint",
                qualifier="readonly",
                dimensionVals=[16],
                memProperties=memProperties,
            ),
        ]

        # One workgroup per frequency bin
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                loiacono_home, "shaders/loiacono_generic.comp.template"
            ),
            constantsDict=self.constantsDict,
            device=self.device,
            name="loiacono",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=DEBUG,
            workgroupCount=[len(fprime), 1, 1],
            useFence=True
        )
        self.finalize()

        self.gpuBuffers.f.set(fprime)
        self.gpuBuffers.offset.zeroInitialize()
        self.offset = 0

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        self.spectrum = self.gpuBuffers.L
        print("vlen " + str(vlen))

    def feed(self, newData, blocking=True):
        self.gpuBuffers.x.setByIndexStart(self.offset, newData)
        self.offset = (self.offset + len(newData)) % self.signalLength
        self.gpuBuffers.offset.setByIndex(index=0, data=[self.offset])
        self.run(blocking)

    def getSpectrum(self):
        self.spectrum = self.gpuBuffers.L.get()
        return self.spectrum


if __name__ == "__main__":

    sr = 48000
    A4 = 440
    z = np.sin(np.arange(2**15)*2*np.pi*A4/sr)
    z += np.sin(2*np.arange(2**15)*2*np.pi*A4/sr)
    z += np.sin(3*np.arange(2**15)*2*np.pi*A4/sr)
    z += np.sin(4*np.arange(2**15)*2*np.pi*A4/sr)

    multiple = 40
    normalizedStep = 5.0/sr
    fprime = np.arange(100/sr, 3000/sr, normalizedStep)

    linst = Loiacono(
        fprime=fprime,
        multiple=multiple,
        dtftlen=2**15
    )
    linst.debugRun(z)

    instance = ve.instance.Instance(verbose=True)
    device = instance.getDevice(0)
    linst_gpu = Loiacono_GPU(
        device=device,
        fprime=fprime,
        multiple=linst.multiple,
    )
    linst_gpu.gpuBuffers.x.set(z)
    for i in range(10):
        linst_gpu.debugRun()

    readstart = time.time()
    linst_gpu.spectrum = linst_gpu.gpuBuffers.L.get()
    print("Readtime " + str(time.time() - readstart))

    import matplotlib.pyplot as plt
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.fprime*sr, linst_gpu.spectrum)
    ax1.set_title("GPU Result")
    ax2.plot(linst.fprime*sr, linst.spectrum)
    ax2.set_title("CPU Result")

    plt.show()

    instance.release()
