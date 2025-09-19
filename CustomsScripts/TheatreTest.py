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

# stl_path = Path("../PyroomMeshes/manyObjectsCube.stl")
stl_path = Path("../PyroomMeshes/TheatreLOW.stl")
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

theatreRoom = pra.Room(
    walls=walls,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# theatreRoom.add_source([0, 0, 1], signal=anechoicAudioSource)
# theatreRoom.add_microphone_array(np.c_[[1, 2, 3], [3, 2, 1]])
# theatreRoom.add_microphone_array(pra.MicrophoneArray(np.array([[1, 2, 3]]).T, fs))
# theatreRoom.add_source([0, 8, 5])
# theatreRoom.add_microphone_array(np.c_[[0, -8, 5]])
# theatreRoom.add_microphone_array(np.c_[[-5,-7,2],[5,-7,2]])
# theatreRoom.add_microphone_array(np.c_[[3.8000, -3.7500, 1.3000],[2.4077, -7.3239, 1.3000],[5.0000, -2.0000, 3.5000],[3.5000, -8.2000, 3.5000],[5.1000, -2.0000, 5.8000],[4.0000, -8.5000, 5.8000],[5.1000, -2.0000, 8.2000],[4.0000, -8.5000, 8.2000],[-3.8000, -3.7500, 1.3000],[-2.4077, -7.3239, 1.3000],[-5.0000, -2.0000, 3.5000],[-3.5000, -8.2000, 3.5000],[-5.1000, -2.0000, 5.8000],[-4.0000, -8.5000, 5.8000],[-5.1000, -2.0000, 8.2000],[-4.0000, -8.5000, 8.2000]])

# source and mic locations
theatreRoom.add_source([0, 0, 1], signal=anechoicAudioSource)
theatreRoom.add_microphone_array(np.c_[[1, 2, 3]])
# theatreRoom.add_microphone_array(pra.MicrophoneArray(np.array([[1, 2, 3]]).T, theatreRoom.fs))

plt.figure()
# run ism
theatreRoom.simulate()

# compute the rir
t = Timer()
t.start()

theatreRoom.image_source_model()
t.show("ImageSource")
theatreRoom.ray_tracing()
t.show("RayTracing")
theatreRoom.compute_rir()
t.show("RIR")
theatreRoom.plot_rir()
t.show("Plot-RIR")


# this plots the RIR between the 1st source and the 1st microphone
# rir_time = np.arange(len(theatreRoom.rir[0][0])) / theatreRoom.fs
# plt.plot(rir_time, theatreRoom.rir[0][0])


# ---------------Normal usecase, export what the mic recorded in the room (the source+applied IR)---------------
# But it's not the IR itself
# theatreRoom.mic_array.to_wav("MicAudio.wav", mono=True, norm=True, bitdepth=np.int16)

# ---------------Custom call with rawIR instead of mic signals---------------

exportIRtoWav(
    rawIR=theatreRoom.rir,
    filename="IR.wav",
    mono=False,
    norm=True,
)

t.show("Wav Export")
t.stop()
plt.show()
