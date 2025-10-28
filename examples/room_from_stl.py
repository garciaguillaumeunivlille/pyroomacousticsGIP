from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import pyroomacoustics as pra
from timer import Timer

# Timer to log elapsed time
t = Timer()
t.start()


try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err


RenderARGS = {
    "exportPath": "Generated-IRs/MaterialsTest",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
}


material = pra.Material(energy_absorption=0.1, scattering=0.0)
# with numpy-stl
the_mesh = mesh.Mesh.from_file("data/INRIA_MUSIS.stl")
ntriang, nvec, npts = the_mesh.vectors.shape
size_reduc_factor = 500.0  # to get a realistic room size (not 3km)
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

room = pra.Room(
    walls,
    fs=RenderARGS["fs"],
    max_order=RenderARGS["IMS_Order"],
    ray_tracing=RenderARGS["useRayTracing"],
    air_absorption=True,
).add_microphone_array(
    np.c_[
        [-3.8, -3.75, 1.3],
        # [3.8, -3.75, 1.3],
        # [2.4077, -7.3239, 1.3],
        # [-2.4077, -7.3239, 1.3],
        # [-5.0, -2.0, 3.5],
        # [5.0, -2.0, 3.5],
        # [3.5, -8.2, 3.5],
        # [-3.5, -8.2, 3.5],
        # [-5.1, -2.0, 5.8],
        # [5.1, -2.0, 5.8],
        # [4.0, -8.5, 5.8],
        # [-4.0, -8.5, 5.8],
        # [-5.1, -2.0, 8.2],
        # [5.1, -2.0, 8.2],
        # [4.0, -8.5, 8.2],
        # [-4.0, -8.5, 8.2],
    ]
)

# attempting to add source in room externally to catch a common unresolved persistent error
atmptSources = 0
while atmptSources < 3:
    try:
        room.add_source(
            # [-1.75, 9.15, 3.3572],
            # [1.75, 9.15, 3.3572],
            # [3.0, 2.0, 3.3572],
            [-3.0, 2.0, 3.3572],
            # [-3.35, -1.0, 1.4],
            # [0.0, -1.0, 1.4],
            # [3.35, -1.0, 1.4],
        )
        break
    except ValueError:
        atmptSources += 1
        t.show(f">>>failed adding source {atmptSources}/3 attempts")
t.show(f"added source OK")


# compute the rir
room.image_source_model()
room.ray_tracing()
room.compute_rir()
room.plot_rir()
# show the room
room.plot(img_order=1)
plt.show()
