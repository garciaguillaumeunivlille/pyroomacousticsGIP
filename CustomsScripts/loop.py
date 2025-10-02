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


def exportIRToWav(formattedIRs, norm, fileName, iMic):

    path = f"{folderpath}/{fileName}.wav"

    signal = formattedIRs[iMic][0]  # [micro][source]

    if norm:
        from utilities import normalize

        signal = normalize(signal, bits=np.int8)

    float_types = [float, float, np.float32, np.float64]
    bitdepth = float_types[0]
    signal = np.array(signal, dtype=bitdepth)

    # create .wav file
    wavfile.write(path, fs, signal)
    return signal


JSONData = {}


def makeJsonData(signal, name, i, showGraph):

    # Get volume RootMinSquare sqr(avg(all [i]²))
    volume = pra.rms(signal)
    # maxAmp = np.max(pra.doa.detect_peaks(signal))

    # Get Source-Mic distance
    m = microphonesMap.get(i + 1)
    s = TheatreRoom.sources[0].position
    dist = np.linalg.norm(s - m)

    # detection of first ""peak"" > 0.000000
    p = pra.doa.detect_peaks(signal, mph=0.000001, show=showGraph)
    firstPeakIndex = p[0] if len(p) else 0
    peakDelay = firstPeakIndex / fs

    irData = {
        "distance": float(dist),
        "volume": float(volume),
        "peakDelay": float(peakDelay),
    }
    JSONData.update({name: irData})


def writeJsonFile():

    #  encode dict as JSON
    data = json.dumps(JSONData, indent=1, ensure_ascii=True)
    #  set output path and file name (set your own)

    #  write JSON file
    filepath = f"{folderpath}/IR.json"
    with open(filepath, "w") as outfile:
        outfile.write(data + "\n")


# Mesh Import
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
allMeshesGeometry = []
mat = pra.Material(energy_absorption=1.0, scattering=0.0)
materials = [
    mat,
    mat,
    # pra.Material(energy_absorption=0.3, scattering=0.3),
    # pra.Material(energy_absorption=0.4, scattering=0.4),
]

# import des fichiers stl séparés par matériaux distincts, avec un nombre et ordre prédéfini
for i in range(0, 2):
    # import stl
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

TheatreRoom = pra.Room(
    walls=allMeshesGeometry,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# sources/mic locations
sourcesMap = {
    "A": [-1.7500, 9.1500, 3.3572],
    "B": [1.7500, 9.1500, 3.3572],
    "C": [3.0000, 2.0000, 3.3572],
    "D": [-3.0000, 2.0000, 3.3572],
    "E": [-3.3500, -1.0000, 1.4000],
    "F": [0.0000, -1.0000, 1.4000],
    "G": [3.3500, -1.0000, 1.4000],
}
microphonesMap = {
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


# Making pyroom mic array from the map, to add them in the room
microphoneArray = []
for i in range(1, len(microphonesMap) + 1):
    microphoneArray.append(microphonesMap[i])

# microphoneArray = [(k, sum(v)) for k, v in microphonesMap.items()]

TheatreRoom.add_microphone_array(
    pra.MicrophoneArray(np.array(microphoneArray).T, TheatreRoom.fs)
)

folderpath = f"./IRtest"
if not os.path.exists(folderpath):
    os.makedirs(folderpath)


t = Timer()
t.start()


# Sources Loop
for sourceLabel, pos in sourcesMap.items():

    if TheatreRoom.sources:
        TheatreRoom.sources.clear()
    TheatreRoom.add_source(pos, signal=anechoicAudioSource)

    if TheatreRoom.rir:
        TheatreRoom.rir.clear()

    TheatreRoom.plot()

    # This function will generate all the images sources up to the order required and use them to generate the RIRs, which will be stored in the rir attribute of room.
    TheatreRoom.image_source_model()
    t.show("ImageSource")

    # TheatreRoom.plot_rir()
    TheatreRoom.simulate()
    # t.show("Plot-RIR")
    t.show('simulate')
    # The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
    allRawIRs = TheatreRoom.rir
    # required conversion from list to nparray
    formattedIRs = np.asarray(allRawIRs, dtype="object")

    # mic loop through each IR
    if len(microphoneArray) == len(formattedIRs):
        for i in range(0, len(formattedIRs)):

            fileName = f"{sourceLabel}-{i+1}"

            signal = exportIRToWav(
                formattedIRs=formattedIRs, norm=False, fileName=fileName, iMic=i
            )

            # store json data
            makeJsonData(signal, fileName, i, showGraph=False)

    plt.title(sourceLabel)
    plt.savefig(f"{folderpath}/{sourceLabel}.png")

writeJsonFile()


t.show("Wav Export")
t.stop()
plt.show()
