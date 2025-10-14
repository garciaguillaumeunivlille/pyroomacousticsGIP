import json
import os, sys
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


def exportIRToWav(computedIRs, norm, fileName, micIndex):

    path = f"{folderpath}/{fileName}"

    signal = computedIRs[micIndex][0]  # [micro][source]

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


def makeJsonData(signal, name, wavName, sLabel, micID, sourcePos, micPos, showGraph):

    # Get volume RootMinSquare sqr(avg(all [i]²))
    volume = pra.rms(signal)
    # maxAmp = np.max(pra.doa.detect_peaks(signal))

    # Get Source-Mic distance
    dist = np.linalg.norm(np.array(sourcePos) - np.array(micPos))

    # detection of first ""peak"" > 0.000000
    p = pra.doa.detect_peaks(signal, mph=0.000001, show=showGraph)
    if len(p):
        firstPeakIndex = max(p)
    else:
        firstPeakIndex = 0

    peakDelay = firstPeakIndex / fs

    compareVolume = compareLawfulVolume(dist, volume)

    irData = {
        "IRPath": wavName,
        "distance": truncate(dist, 6),
        "volume": truncate(volume, 6),
        "lawfulVolume": truncate(compareVolume[0], 5),
        "volumeGap": truncate(compareVolume[1], 5),
        "peakDelay": truncate(peakDelay, 5),
        "sourceID": sLabel,
        "sourceX": sourcePos[0],
        "sourceY": sourcePos[1],
        "sourceZ": sourcePos[2],
        "micID": str(micID),
        "micX": micPos[0],
        "micY": micPos[1],
        "micZ": micPos[2],
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


def compareLawfulVolume(distance, volume):
    supposedVolume = 1 / pow(distance, 2)
    gap = abs(supposedVolume - volume)
    return [supposedVolume, gap]


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "{}".format(f)
    if "e" in s or "E" in s:
        return "{0:.{1}f}".format(f, n)
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])


def writeRawIRs(jdata, filename):

    #  encode dict as JSON
    data = json.dumps(jdata, indent=1, ensure_ascii=True)
    #  set output path and file name (set your own)

    #  write JSON file
    filepath = f"{filename}.txt"
    with open(filepath, "w") as outfile:
        outfile.write(data + "\n")


t = Timer()
t.start()

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
fs, anechoicAudioSource = wavfile.read(
    "../examples/CustomSamples/IR-Dirac-44100-20hz-22050hz-1s.wav"
)

size_reduc_factor = 1  # to get a realistic room size (not 3km)

# Le rendu en cours c'est le theatre joined en un seul STL avec (energy_absorption=1.0, scattering=0.0)
# Le code en cours est pour préparer le rendu avec plusieurs matériaux,
# il reste à trouver les bonnes valeurs

meshMatMap = {
    "roomOUT": {
        "stlPath": "roomOccIN.stl",
        "material": pra.Material(energy_absorption=1.0, scattering=0.0),
    },
}

# import des fichiers stl séparés par matériaux distincts, avec un nombre et ordre prédéfini
allMeshesGeometry = []
for k, v in meshMatMap.items():

    # import des fichiers stl
    path = v["stlPath"]
    the_mesh = mesh.Mesh.from_file(Path(f"../PyroomMeshes/MetricMeshes/{path}"))
    ntriang, nvec, npts = the_mesh.vectors.shape

    # create one wall per triangle
    for w in range(ntriang):
        # appliquer les matériaux indexés dans materials
        allMeshesGeometry.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                v["material"].energy_absorption["coeffs"],
                v["material"].scattering["coeffs"],
            )
        )
# t.show("Done STL imports")

metricRoom = pra.Room(
    walls=allMeshesGeometry,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# t.show("Room created")

# rayon de captation des rayons (plus grand = plus rapide mais - précis)
metricRoom.set_ray_tracing(receiver_radius=0.5)  # default =0.5


# sources/mic locations
sourcesMap = {"A": [-3, 3, 5], "B": [-3, 3, 1]}
microphonesMap = {
    1: [3, -3, 1],
    2: [3, 3, 4],
    # 3: [-2, -2, 6]
}

# Making pyroom mic array from the map, to add them in the room
microphoneArray = list(microphonesMap.values())  # [[mx,my,mz],[mx,my,mz]...]
metricRoom.add_microphone_array(
    pra.MicrophoneArray(
        R=np.array(object=microphoneArray).T,
        fs=metricRoom.fs,
        directivity=(
            None if metricRoom.ray_tracing else pra.directivities.Omnidirectional()
        ),
    )
)
# t.show("setup microphones")

# for m in metricRoom.mic_array:
#     print(m.receiver_radius)

# Render folder
folderpath = f"./metric"
if not os.path.exists(folderpath):
    os.makedirs(folderpath)


# t.show("start main loop")
# Main Loop
for sourceLabel, sourcePos in sourcesMap.items():

    if metricRoom.sources:
        metricRoom.sources.clear()
    metricRoom.add_source(sourcePos, signal=anechoicAudioSource)

    if metricRoom.rir:
        metricRoom.rir.clear()

    metricRoom.plot()
    # check refelxion order parameters

    # plt.show()

    # t.show("begin image source")
    # This function will generate all the images sources up to the order required and use them to generate the RIRs, which will be stored in the rir attribute of room.
    metricRoom.image_source_model()
    # t.show(f"--ImageSource for source {sourceLabel}--")

    if metricRoom.ray_tracing:
        metricRoom.ray_tracing()
        # t.show(f"--RayTracing for source {sourceLabel}--")

    # metricRoom.plot_rir()
    metricRoom.simulate()
    # t.show("Plot-RIR")
    # t.show(f"--Simulate sound with source {sourceLabel}--")

    # The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
    computedIRs = metricRoom.rir

    # Create a plot
    plt.figure()

    # mic loop through each computedIR : [[micro][source]]
    if len(computedIRs) == len(microphoneArray):
        # needed since we iterate from a map/dict with arbitrary int IDs
        micLoopIndex = 0
        for micID, micPos in microphonesMap.items():

            fileName = f"{sourceLabel}{micID}"
            wavFileName = f"{fileName}.wav"

            signal = exportIRToWav(
                computedIRs=computedIRs,
                norm=False,
                fileName=wavFileName,
                micIndex=(micLoopIndex),
            )

            printIndex = micLoopIndex + 1
            # t.show(f"Export {wavFileName} {printIndex}/{len(computedIRs)}")

            # store json data
            makeJsonData(
                signal,
                fileName,
                wavFileName,
                sourceLabel,
                micID,
                sourcePos,
                micPos,
                showGraph=False,
            )

            # plot signal at microphone 1
            plt.subplot(len(microphoneArray), 1, printIndex)
            plt.plot(metricRoom.mic_array.signals[micLoopIndex])
            plt.title(f"Microphone {printIndex} signal")
            plt.xlabel("Time [s]")
            micLoopIndex += 1
    else:
        raise Exception(f"IR data is missing some Microphones indexes")

    plt.savefig(f"{folderpath}/{sourceLabel}.png")

writeJsonFile()
# t.show("Json Export")
t.stop()
