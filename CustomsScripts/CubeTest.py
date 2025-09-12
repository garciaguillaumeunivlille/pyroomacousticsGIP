from pathlib import Path
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import os, sys
import json


try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

# Debug file
JSONData = {}


def writeJsonFile(dataName, serializedData):

    JSONData.update({dataName: serializedData})
    #  encode dict as JSON
    data = json.dumps(JSONData, indent=1, ensure_ascii=True)
    #  set output path and file name (set your own)
    file_name = os.path.join("./log", "log.json")

    #  write JSON file
    with open(file_name, "w") as outfile:
        outfile.write(data + "\n")


def serializeRawIR(rawIR):

    # np.savetxt('log.txt', rawIR, fmt='%f')
 
    a = rawIR.tolist()
    map("{0:.16f}".format, a)
    writeJsonFile("raw", a)
    print(a)


stl_path = Path("../PyroomMeshes/cubeRoomNormalsOUT.stl")
# stl_path = Path("../PyroomMeshes/manyObjectsCube.stl")
# stl_path = Path("../PyroomMeshes/cubeRoomNormalsIN.stl")
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
fs = 16000

room = (
    pra.Room(
        walls,
        fs,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    ).add_source([0, 0, 1])
    # .add_microphone_array(np.c_[[1, 2, 3]])
    # .add_microphone_array(np.c_[[1, 2, 3], [3, 2, 1]])
    .add_microphone_array(pra.MicrophoneArray(np.array([[1, 2, 3]]).T, fs))
)

# compute the rir
room.image_source_model()
room.ray_tracing()
room.compute_rir()
room.plot_rir()

# plt.figure()

# this plots the RIR between the 1st source and the 1st microphone
# rir_time = np.arange(len(room.rir[0][0])) / room.fs
# plt.plot(rir_time, room.rir[0][0])
 
# serializeRawIR(room.rir)
print(room.rir)

# np.savetxt('log.wav', room.rir[0][0], fmt='%f')
np.save("./IR", room.rir[0][0])


# print(room.mic_array.to)
# room.mic_array.to_wav("./IR.wav", norm=True, bitdepth=np.int16)

# print(len(room.mic_array))
# plt.show()
# room.plot(img_order=3)
# np.set_printoptions(suppress=True, threshold=sys.maxsize)

# Modifications effectuées :
# export d'un default cube blender .stl en scale 10 à l'export
# réduction de la valeur "size_reduc_factor" 500-->1
# placement de la source DANS le volume du modele

# Remarques sur le modèle à exporter
# Les faces du mesh englobant doivent être orientés vers l'extérieur (unlink cycleacoustics)
# Dans ce mesh englobant, certains sous-mesh peuvent avoir des faces flip vers l'intérieur
# Peut importe si chaque mesh est join ou split en sous objets
"""
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
"""
