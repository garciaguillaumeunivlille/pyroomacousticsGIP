"""
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
"""

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path

fs, audio_anechoic = wavfile.read("../examples/samples/guitar_16k.wav")
stl_path = Path("../PyroomMeshes/manyObjectsCube.stl")

try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

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

# room dimension
room_dim = [5, 4, 6]

# Create the shoebox
shoebox = pra.ShoeBox(
    room_dim,
    absorption=0.2,
    fs=fs,
    max_order=15,
)

cubeRoom = pra.Room(
    walls=walls,
    fs=fs,
    max_order=3,
    ray_tracing=True,
    air_absorption=True,
)

# source and mic locations
cubeRoom.add_source([0, 0, 1], signal=audio_anechoic)
cubeRoom.add_microphone_array(np.c_[[1, 2, 3], [3, 2, 1]])
# cubeRoom.add_microphone_array(pra.MicrophoneArray(np.array([[1, 2, 3]]).T, cubeRoom.fs))

# plt.figure()
# run ism
cubeRoom.simulate()

# compute the rir
# cubeRoom.image_source_model()
# cubeRoom.ray_tracing()
# cubeRoom.compute_rir()
# cubeRoom.plot_rir()


# this plots the RIR between the 1st source and the 1st microphone
# rir_time = np.arange(len(cubeRoom.rir[0][0])) / cubeRoom.fs
# plt.plot(rir_time, cubeRoom.rir[0][0])


# plt.show()

audio_reverb = cubeRoom.mic_array.to_wav("AA.wav", norm=True, bitdepth=np.int16)
