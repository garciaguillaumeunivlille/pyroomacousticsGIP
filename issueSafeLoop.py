import os
import json
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from stl import mesh
from timer import Timer
from pathlib import Path
from scipy.io import wavfile

# Timer to log elapsed time
t = Timer()
t.start()

# IR computing parameters, except [Material, Mesh, Source-Micros locations]
RenderARGS = {
    "exportPath": "Generated-IRs/EXPORT",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
}

JSONData = {}

# add renderArgs to the json
JSONData.update({"RenderARGS": RenderARGS})


def makeJsonData(signal, name, path, sLabel, micID, sourcePos, micPos, showGraph):

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
        "IRPath": path,
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
    filepath = f"{RenderARGS['exportPath']}/IR.json"
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
    truncated = ".".join([i, (d + "0" * n)[:n]])
    return float(truncated)


def writeRawIRs(jdata, filename):

    #  encode dict as JSON
    data = json.dumps(jdata, indent=1, ensure_ascii=True)
    #  set output path and file name (set your own)

    #  write JSON file
    filepath = f"{filename}.txt"
    with open(filepath, "w") as outfile:
        outfile.write(data + "\n")


def customExportIRToWav(computedIRs, fileName):
    signal = computedIRs[0][0]  # [micro][source]
    signal = np.array(signal, dtype=np.float32)
    # map to 16-bit
    max_16bit = 2**15
    signal = signal * max_16bit
    # change the data type to PCM 16 bits
    signal = signal.astype(np.int16)
    # create .wav file
    wavfile.write(fileName, RenderARGS["fs"], signal)
    return signal


# map indexing path to file and material properties for the mesh multiple mesh parts that were split by materials
meshMatMap = {
    "Theatre_Wood_Parquet": {
        "stlFileName": "Theatre_Wood_Parquet.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.07,
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Walls": {
        "stlFileName": "Theatre_Wood_Walls.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.06,
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Deco": {
        "stlFileName": "Theatre_Wood_Deco.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.04,
            scattering=0.0,
        ),
    },
    "Theatre_Limestone": {
        "stlFileName": "Theatre_Limestone.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.02,
            scattering=0.0,
        ),
    },
    "Theatre_Plaster": {
        "stlFileName": "Theatre_Plaster.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.01,
            scattering=0.0,
        ),
    },
    "Theatre_Fibre": {
        "stlFileName": "Theatre_Fibre.Flip.stl",
        "material": pra.Material(
            energy_absorption=0.2,
            scattering=0.0,
        ),
    },
}

sourcesMap = {
    "A": [-1.75, 9.15, 3.3572],
    "B": [1.75, 9.15, 3.3572],
    "C": [3.0, 2.0, 3.3572],
    "D": [-3.0, 2.0, 3.3572],
    "E": [-3.35, -1.0, 1.4],
    "F": [0.0, -1.0, 1.4],
    "G": [3.35, -1.0, 1.4],
}

microphonesMap = {
    1: [-3.8, -3.75, 1.3],
    2: [3.8, -3.75, 1.3],
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

# ok B13    13: [-4.3, -2.5, 8.2],

# Build room from geometry
walls = []
for k, v in meshMatMap.items():

    # import des fichiers stl
    stlFileName = v["stlFileName"]
    the_mesh = mesh.Mesh.from_file(Path(f"PyroomMeshes/ReworkedMeshes/{stlFileName}"))
    ntriang, nvec, npts = the_mesh.vectors.shape
    # size_reduc_factor = 1 # unused because stl file is up to scale

    # create one wall per triangle
    for w in range(ntriang):
        # inverse normals because subject stl room has inward normals
        inverse_triangle = np.array(
            [the_mesh.vectors[w][2], the_mesh.vectors[w][1], the_mesh.vectors[w][0]]
        )
        walls.append(
            pra.wall_factory(
                inverse_triangle.T,
                v["material"].energy_absorption["coeffs"],
                v["material"].scattering["coeffs"],
            )
        )
t.show("Done STL imports")

try:

    # for times in range(0, 2):
    for sourceLabel, sourcePos in sourcesMap.items():
        micIndex = 0
        for micID, micPos in microphonesMap.items():

            # Instanciating room with geometry and some render parameters
            room = pra.Room(
                walls,
                fs=RenderARGS["fs"],
                max_order=RenderARGS["IMS_Order"],
                ray_tracing=RenderARGS["useRayTracing"],
                air_absorption=True,
            ).add_microphone_array(
                np.c_[micPos],
            )

            # attempting to add source in room externally to catch a common unresolved persistent error
            addSrcAttempts = 0
            while addSrcAttempts < 3:
                try:
                    room.add_source(sourcePos)
                    break
                except ValueError:
                    addSrcAttempts += 1
                    t.show(f">>>failed adding source {addSrcAttempts}/3 attempts")
            t.show(f"added source OK")

            if RenderARGS["useRayTracing"]:
                room.set_ray_tracing(
                    n_rays=RenderARGS["RT_n_rays"],
                    receiver_radius=RenderARGS["RT_receiver_radius"],
                )  # default =0.5

            simulationAttempts = 0
            simulationSuccess = False
            while simulationAttempts < 5:
                try:
                    # compute the rir
                    t.show("processing image_source_model...")
                    room.image_source_model()
                    if RenderARGS["useRayTracing"]:
                        t.show("processing ray_tracing...")
                        room.ray_tracing()
                    t.show("compute_rir")
                    room.compute_rir()
                    t.show("plot_rir")
                    room.plot_rir()
                    simulationSuccess = True
                    break
                except (ValueError, RuntimeError) as e:
                    t.show(f"compute_rir failed with {str(e)}")
                    simulationAttempts += 1
                    t.show(f">>>simulation failed {simulationAttempts}/5 attempts")
                if simulationAttempts == 5:
                    # print("forcing JSON export")
                    # writeJsonFile()
                    JSONData.update({f"[SKIPPED] {sourceLabel}{micID}": "skipped after 5 failed attempts"})
                    break
            t.show(f"simulation OK")

            if(simulationSuccess):

                # The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
                computedIRs = room.rir

                if len(computedIRs) == len(room.mic_array):
                
                    folderpath = f"{RenderARGS["exportPath"]}"
                    name = f"{sourceLabel}{micID}"
                    wavFileName = f"{name}.wav"
                    # wavFileName = f"{sourceLabel}{micID}-{times+1}.wav"
                    fileName = f"{folderpath}/{wavFileName}"
    
                    if not os.path.exists(folderpath):
                        os.makedirs(folderpath)
    
                    signal = customExportIRToWav(computedIRs=computedIRs, fileName=fileName)
    
                    # store json data
                    makeJsonData(
                        signal,
                        name,
                        fileName,
                        sourceLabel,
                        micID,
                        sourcePos,
                        micPos,
                        showGraph=False,
                    )
    
                    t.show(f">Export {wavFileName} {micIndex+1}/{len(computedIRs)}")
                    micIndex += 1
                else:
                    t.show(
                        f"There is {len(computedIRs)} computed IRs for {len(room.mic_array)} microphones"
                    )
                    raise Exception(f"IR data is missing some Microphones indexes")

            else:
                micIndex += 1

except KeyboardInterrupt:
    JSONData.update({"SCRIPT INTERUPTED AT": t.getElapsedTime()})
    writeJsonFile()

writeJsonFile()
t.show(">Json Export")
t.stop()