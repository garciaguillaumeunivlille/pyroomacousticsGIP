# MatCubeTest


"""
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
"""
import json
import os
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


x = 1.0
mat = pra.Material(energy_absorption=x, scattering=0.0)
# folderpath = f"./IR{x}"
# os.mkdir(folderpath)


def exportIRsToWav(rawIRs, filename, norm=False):
    """
    CUSTOM
    """
    from scipy.io import wavfile

    float_types = [float, float, np.float32, np.float64]
    bitdepth = float_types[0]

    # signal = rawIRs
    print(type(rawIRs))
    formattedIRs = np.asarray(rawIRs, dtype="object")
    print(type(formattedIRs))
    for i in range(0, len(formattedIRs)):

        signal = formattedIRs[i][0]  # [micro][source]
 
        if norm:
            from utilities import normalize
            signal = normalize(signal, bits=np.int8)

        signal = np.array(signal, dtype=bitdepth)

        # create .wav file
        name = f"{filename}-{i+1}"
        path = f"IR{x}/{name}.wav"
        wavfile.write(path, fs, signal)

        # store json data
        pickUpDataFromSignal(signal, name, i)

    formatAndWriteJSON(filename)

JSONData = {}
def pickUpDataFromSignal(signal, name, i):

    # Get volume RootMinSquare sqr(avg(all [i]²))
    volume = pra.rms(signal)
    # maxAmp = np.max(pra.doa.detect_peaks(signal))

    # Get Source-Mic distance
    m = microphoneMap.get(i + 1)
    s = cubeRoom.sources[0].position
    dist = np.linalg.norm(s - m)
 
    # detection of first ""peak"" > 0.000000
    firstPeakIndex = pra.doa.detect_peaks(signal, mph=0.000001, show=True)[0]
    peakDelay = firstPeakIndex / fs
 
    irData = {
        "distance": float(dist),
        "volume": float(volume),
        "peakDelay": float(peakDelay),
    }
    JSONData.update({name: irData})

 
def formatAndWriteJSON(allAmps, allDists, filename):
 
    for i in range(0, len(allAmps)):
        # Format
        irData = {
            "distance": float(allDists[i]),
            "volume": float(allAmps[i]),
        }
        name = f"{filename}-{i+1}"
        JSONData.update({name: irData})
    writeJsonFile()


def writeJsonFile():

    #  encode dict as JSON
    data = json.dumps(JSONData, indent=1, ensure_ascii=True)
    #  set output path and file name (set your own)

    #  write JSON file
    filepath = f"{folderpath}/IR{x}.json"
    with open(filepath, "w") as outfile:
        outfile.write(data + "\n")


def remapInRange(oldValue, oldMin, oldMax, newMin, newMax):

    oldRange = oldMax - oldMin
    if oldRange == 0:
        return newMin
    else:
        newRange = newMax - newMin
        return (((oldValue - oldMin) * newRange) / oldRange) + newMin


def remapAmplitudesToDistance(amplitudes, distances):
    res = []
    # min à 0
    minA = min(amplitudes)
    # max à
    maxA = max(amplitudes)
    minD = min(distances)
    maxD = max(distances)

    for amp in amplitudes:
        remapedAmplitude = remapInRange(amp, minA, maxA, minD, maxD)
        res.append(remapedAmplitude)

    return res


try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

# fs is samplerate as integer
# audio source is numpy.ndarray
# fs, anechoicAudioSource = wavfile.read("../examples/samples/guitar_16k.wav")
fs, anechoicAudioSource = wavfile.read("../examples/CustomSamples/Basic-808-Clap.wav")

size_reduc_factor = 1  # to get a realistic room size (not 3km)

# Comment appliquer les bon matériaux
# Exporter le théatre en objets découpés par matériaux
# Un stl par objet
# un matériau par objet
# condition dans walls.append et loop sur tous les objets pour remplir walls
# avant de créer la room avec pra.room


allMeshesGeometry = []
# mat = pra.Material(energy_absorption=1, scattering=0.2)
# mat = pra.Material(energy_absorption=1, scattering=1.0)
materials = [
    mat,
    mat,
    # pra.Material(energy_absorption=0.3, scattering=0.3),
    # pra.Material(energy_absorption=0.4, scattering=0.4),
]

# import des fichiers stl séparés par matériaux distincts, avec un nombre et ordre prédéfini
for i in range(0, 2):
    # import stl
    # 4 stl_path = Path(f"../PyroomMeshes/MatCube.00{i}.stl")
    stl_path = Path(f"../PyroomMeshes/TheatreLowPyroomMat.00{i}.stl")
    the_mesh = mesh.Mesh.from_file(stl_path)
    ntriang, nvec, npts = the_mesh.vectors.shape

    # create one wall per triangle
    for w in range(ntriang):
        # appliquer les matériaux indexés dans materials
        allMeshesGeometry.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                materials[i].energy_absorption["coeffs"],
                materials[i].scattering["coeffs"],
            )
        )

cubeRoom = pra.Room(
    walls=allMeshesGeometry,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# source and mic locations

# TetsCube
# cubeRoom.add_source([0, 0, 1], signal=anechoicAudioSource)
# microphoneMap = {
#     1: [0, 0.4, 0.5],
#     2: [0, -0.4, 0.5],
# }

# Theatre
cubeRoom.add_source([0, 5, 4], signal=anechoicAudioSource)
microphoneMap = {
    1: [-3.8, -3.7500, 1.3],
    2: [3.8, -3.7500, 1.3],
    3: [2.4077, -7.3239, 1.3],
    4: [-2.4077, -7.3239, 1.3],
    5: [-5.0, -2.0, 3.5],
    6: [5.0, -2.0, 3.5],
    7: [3.5, -8.2, 3.5],
    8: [-3.5, -8.2, 3.5],
    9: [-5.1, -2.0, 5.8],
    10: [5.1, -2.0, 5.8],
    11: [4.0, -8.5, 5.8],
    12: [-4.0, -8.5, 5.8],
    13: [-5.1, -2.0, 8.2],
    14: [5.1, -2.0, 8.2],
    15: [4.0, -8.5, 8.2],
    16: [-4.0, -8.5, 8.2],
}
# ################################### recheck mic positions
# display axes


# regen files

# Making pyroom mic array from the map
microphoneArray = []
for i in range(1, len(microphoneMap) + 1):
    microphoneArray.append(microphoneMap[i])

cubeRoom.add_microphone_array(
    pra.MicrophoneArray(np.array(microphoneArray).T, cubeRoom.fs)
)

cubeRoom.plot()

# compute the rir
t = Timer()
t.start()

# This function will generate all the images sources up to the order required and use them to generate the RIRs, which will be stored in the rir attribute of room.
# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
cubeRoom.image_source_model()
t.show("ImageSource")

# cubeRoom.ray_tracing()
# # t.show("RayTracing")

# cubeRoom.compute_rir()
# t.show("RIR")

# plot the RIR between mic 1 and source 0
# plt.plot(cubeRoom.rir[1][0])

# By calling simulate(), a convolution of the signal of each source (if not None) will be performed with the corresponding room impulse response.
# The output from the convolutions will be summed up at the microphones.
# The result is stored in the signals attribute of room.mic_array with each row corresponding to one microphone.
# cubeRoom.simulate()
# plot signal at microphone 1
# plt.plot(room.mic_array.signals[1,:])

# plot rir does compute_rir() if not done
cubeRoom.plot_rir()
t.show("Plot-RIR")


# this plots the RIR between the 1st source and the 1st microphone
# rir_time = np.arange(len(cubeRoom.rir[0][0])) / cubeRoom.fs
# plt.plot(rir_time, cubeRoom.rir[0][0])


# ---------------Normal usecase, export what the mic recorded in the room (the source+applied IR)---------------
# But it's not the IR itself

# for x in [np.int8,np.int16,np.int32,np.int64] :
# cubeRoom.mic_array.to_wav("AAA.wav", mono=True, norm=True)
# float, float, np.float32, np.float64

# ---------------Custom call to the same to_wav function---------------
# to_wav(
#     self=cubeRoom.mic_array,
#     filename="BB.wav",
#     mono=False,
#     norm=False,
#     bitdepth=np.int16,
# )

# ---------------Custom call with rawIRs instead of mic signals---------------

exportIRsToWav(
    rawIRs=cubeRoom.rir,
    filename="IR",
    norm=False,
)

# t.show("Wav Export")
# t.stop()
# plt.show()
