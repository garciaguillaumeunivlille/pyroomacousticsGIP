from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import pyroomacoustics as pra

try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

stl_path = Path("../PyroomMeshes/cubeRoomNormalsOUT.stl")
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

room = (
        pra.Room(
            walls,
            fs=16000,
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )
        .add_source([0,0,.100])
        .add_microphone_array(np.c_[[.21,.10,.11],[.21,.16,.07]])
    )

# compute the rir
# room.image_source_model()
# room.ray_tracing()
# room.compute_rir()
# room.plot_rir();

plt.figure()
room.plot(img_order=1)
plt.show()