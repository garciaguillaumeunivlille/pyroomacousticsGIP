import json
import os, sys
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from timer import Timer
from scipy.io import wavfile
from pathlib import Path

RenderARGS = {
    "meshPath" : "Joined-TheatreEnveloppeOUT.stl",
    "exportPath" : "Generated-IRs/20-10",
    "material" : pra.Material(energy_absorption=0.001, scattering=0.0),
    "fs" : 44100,
    "IMS_Order" : 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays" : 5000,
    "sourcePos": [],
    "micPos": [],
}

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
    wavfile.write(path, RenderARGS["fs"], signal)
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

    peakDelay = firstPeakIndex / RenderARGS["fs"]

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
anechoicAudioSource = wavfile.read(
    # "CustomSamples/Basic-808-Clap.wav"
    "CustomSamples/IR-Dirac-44100-20hz-22050hz-1s.wav"
)

 

meshMatMap = {
    # "Fabric": {
    #     "stlPath": "TheatreJoined.001.stl",
    #     "material": RenderARGS["material"],
    # },
    "Stone": {
        "stlPath": "TheatreJoined.002.stl",
        "material": RenderARGS["material"],
    },
    # "Dry Wall": {
    #     "stlPath": "TheatreJoined.003.stl",
    #     "material": RenderARGS["material"],
    # },
    "Worked Wood": {
        "stlPath": "TheatreJoined.004.stl",
        "material": RenderARGS["material"],
    },
    "Smooth Wood": {
        "stlPath": "TheatreJoined.005.stl",
        "material": RenderARGS["material"],
    },
}

JoinedMesh = {
    "Joined-TheatreEnveloppeOUT": {
        "stlPath": RenderARGS["meshPath"],
        "material": RenderARGS["material"],
    }
}

# import des fichiers stl séparés par matériaux distincts, avec un nombre et ordre prédéfini
allMeshesGeometry = []
for k, v in JoinedMesh.items():

    # import des fichiers stl
    path = v["stlPath"]
    # the_mesh = mesh.Mesh.from_file(Path(f"PyroomMeshes/{path}"))
    the_mesh = mesh.Mesh.from_file(Path(f"PyroomMeshes/{path}"))
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 1  # to get a realistic room size (not 3km)

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
t.show("Done STL imports")
 

TheatreRoom = pra.Room(
    walls=allMeshesGeometry,
    fs=RenderARGS["fs"],
    max_order=RenderARGS["IMS_Order"],
    ray_tracing= RenderARGS["useRayTracing"],
    air_absorption=True,
)
# max_order=1 = 10s
# max_order=2 = 1m40s
# max_order=3 = 15m
t.show("Room created")

if RenderARGS["useRayTracing"]:
    # rayon de captation des rayons (plus grand = plus rapide mais - précis)
    TheatreRoom.set_ray_tracing(receiver_radius=RenderARGS["RT_receiver_radius"], n_rays=RenderARGS["RT_n_rays"])  # default =0.5

# sources/mic locations
sourcesMap = {
    # "A": [-1.75, 9.15, 3.3572],
    # "B": [1.75, 9.15, 3.3572],
    # "C": [3.0, 2.0, 3.3572],
    # "D": [-3.0, 2.0, 3.3572],
    # "E": [-3.35, -1.0, 1.4],
    "F": [0.0, -1.0, 1.4],
    # "G": [3.35, -1.0, 1.4],
}
microphonesMap = {
    # 1: [-3.8, -3.75, 1.3],
    # 2: [3.8, -3.75, 1.3],
    # 3: [2.4077, -7.3239, 1.3],
    4: [-2.4077, -7.3239, 1.3],
    5: [-5.0, -2.0, 3.5],
    6: [5.0, -2.0, 3.5],
    # 7: [3.5, -8.2, 3.5],
    # 8: [-3.5, -8.2, 3.5],
    # 9: [-5.1, -2.0, 5.8],
    # 10: [5.1, -2.0, 5.8],
    # 11: [4.0, -8.5, 5.8],
    # 12: [-4.0, -8.5, 5.8],
    # 13: [-5.1, -2.0, 8.2],
    # 14: [5.1, -2.0, 8.2],
    # 15: [4.0, -8.5, 8.2],
    # 16: [-4.0, -8.5, 8.2],
}

# Making pyroom mic array from the map, to add them in the room
microphoneArray = list(microphonesMap.values())  # [[mx,my,mz],[mx,my,mz]...]
TheatreRoom.add_microphone_array(
    pra.MicrophoneArray(
        R=np.array(object=microphoneArray).T,
        fs=RenderARGS["fs"],
        directivity=(None if RenderARGS["useRayTracing"] else pra.directivities.Omnidirectional()),
    )
)
t.show(">Setup microphones")

# for m in TheatreRoom.mic_array:
#     print(m.receiver_radius)

# Render folder
folderpath = RenderARGS["exportPath"]
if not os.path.exists(folderpath):
    os.makedirs(folderpath)


t.show(">Start main loop")
# Main Loop
for sourceLabel, sourcePos in sourcesMap.items():

    if TheatreRoom.sources:
        TheatreRoom.sources.clear()
    TheatreRoom.add_source(sourcePos, signal=anechoicAudioSource)

    if TheatreRoom.rir:
        TheatreRoom.rir.clear()

    TheatreRoom.plot()

    # t.show(">Begin image source")
    # # This function will generate all the images sources up to the order required and use them to generate the RIRs, which will be stored in the rir attribute of room.
    # TheatreRoom.image_source_model()
    # t.show(f"--ImageSource for source {sourceLabel}--")

    # if RenderARGS["useRayTracing"]:
    #     TheatreRoom.ray_tracing()
    #     nRays = TheatreRoom.rt_args["n_rays"]
    #     print(f"number of rays : {nRays}")
    #     t.show(f"--RayTracing for source {sourceLabel}--")

    # TheatreRoom.plot_rir()
    TheatreRoom.compute_rir()  # TRACK 1/5
    t.show(f"--Simulate sound with source {sourceLabel}--")

    # The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
    computedIRs = TheatreRoom.rir

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
            t.show(f">Export {wavFileName} {printIndex}/{len(computedIRs)}")

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
            # plt.subplot(len(microphoneArray), 1, printIndex)
            # plt.plot(TheatreRoom.mic_array.signals[micLoopIndex])
            # plt.title(f"Microphone {printIndex} signal")
            # plt.xlabel("Time [s]")
            micLoopIndex += 1
    else:
        raise Exception(f"IR data is missing some Microphones indexes")

    plt.savefig(f"{folderpath}/{sourceLabel}.png")

writeJsonFile()
t.show(">Json Export")
t.stop()
