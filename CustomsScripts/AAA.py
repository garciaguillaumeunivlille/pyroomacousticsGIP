"""
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
"""

import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, fromText=""):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use  .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{fromText} Total elapsed time : {elapsed_time:0.4f} seconds")

    def show(self, fromText=""):
        """print time elapsed without stopping timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        print(f"{fromText} elapsed time : {elapsed_time:0.4f} seconds")


def exportIRtoWav(rawIR, filename, mono=False, norm=False):
    """
    CUSTOM
    """
    from scipy.io import wavfile

    # print(self.signals)
    # print('-------------')
    # print(self.M)
    # print('-------------')
    # print(self.signals.T)

    float_types = [float, float, np.float32, np.float64]
    bitdepth = float_types[0]

    if mono is True:
        formattedIR = np.asarray(rawIR, dtype=bitdepth)
        signal = formattedIR[1 // 2]
    else:
        signal = rawIR

    if norm:
        from utilities import normalize
        signal = normalize(signal, bits=bits)

    signal = np.array(signal, dtype=bitdepth)

    wavfile.write(filename, fs, signal)


try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

# fs, anechoicAudioSource = wavfile.read("../examples/samples/guitar_16k.wav")
fs, anechoicAudioSource = wavfile.read("../examples/CustomSamples/Basic-808-Clap.wav")

stl_path = Path("../PyroomMeshes/cubeRoomNormalsOUT.stl")
# stl_path = Path("../PyroomMeshes/manyObjectsCube.stl")
# stl_path = Path("../PyroomMeshes/cubeRoomNormalsIN.stl")
# stl_path = Path("../PyroomMeshes/TheatreLOW.stl")
the_mesh = mesh.Mesh.from_file(stl_path)
ntriang, nvec, npts = the_mesh.vectors.shape
size_reduc_factor = 1  # to get a realistic room size (not 3km)
material = pra.Material(energy_absorption=0.2, scattering=0.1)

# create one wall per triangle
walls = []
for w in range(ntriang):
    walls.append(
        pra.wall_factory(
            the_mesh.vectors[w].T / size_reduc_factor,
            material.energy_absorption["coeffs"],
            material.scattering["coeffs"],
        )
    )

cubeRoom = pra.Room(
    walls=walls,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# source and mic locations
cubeRoom.add_source([0, 0, 1], signal=anechoicAudioSource)
cubeRoom.add_microphone_array(np.c_[[1, 2, 3]])
# cubeRoom.add_microphone_array(pra.MicrophoneArray(np.array([[1, 2, 3]]).T, cubeRoom.fs))

plt.figure()


# compute the rir
t = Timer()
t.start()

# This function will generate all the images sources up to the order required and use them to generate the RIRs, which will be stored in the rir attribute of room. 
# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
cubeRoom.image_source_model()
t.show("ImageSource")

cubeRoom.ray_tracing()
t.show("RayTracing")

cubeRoom.compute_rir()
t.show("RIR")

# plot the RIR between mic 1 and source 0
# plt.plot(cubeRoom.rir[1][0])

# By calling simulate(), a convolution of the signal of each source (if not None) will be performed with the corresponding room impulse response. 
# The output from the convolutions will be summed up at the microphones. 
# The result is stored in the signals attribute of room.mic_array with each row corresponding to one microphone.
# cubeRoom.simulate()
# plot signal at microphone 1
# plt.plot(room.mic_array.signals[1,:])

cubeRoom.plot_rir()
t.show("Plot-RIR")


# this plots the RIR between the 1st source and the 1st microphone
# rir_time = np.arange(len(cubeRoom.rir[0][0])) / cubeRoom.fs
# plt.plot(rir_time, cubeRoom.rir[0][0])


# ---------------Normal usecase, export what the mic recorded in the room (the source+applied IR)---------------
# But it's not the IR itself

# for x in [np.int8,np.int16,np.int32,np.int64] :
cubeRoom.mic_array.to_wav("CC.wav", mono=True, norm=True)
# float, float, np.float32, np.float64

# ---------------Custom call to the same to_wav function---------------
# to_wav(
#     self=cubeRoom.mic_array,
#     filename="BB.wav",
#     mono=False,
#     norm=False,
#     bitdepth=np.int16,
# )

# ---------------Custom call with rawIR instead of mic signals---------------

# exportIRtoWav(
#     rawIR=cubeRoom.rir,
#     filename="IR.wav",
#     mono=False,
#     norm=True,
# )

t.show("Wav Export")
t.stop()
plt.show()
