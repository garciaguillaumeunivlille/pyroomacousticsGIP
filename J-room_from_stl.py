from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
from scipy.io import wavfile
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

# IR computing parameters, except [Material, Mesh, Source-Micros locations]
RenderARGS = {
    "exportPath": "Generated-IRs/gitIgnored/TestMultiBande",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
}

name = "ReTest-1BandeAbs-3"


def exportIRToWav(computedIRs, norm, fileName, micIndex):
    signal = computedIRs[micIndex][0]  # [micro][source]
    if norm:
        from utilities import normalize

        signal = normalize(signal, bits=np.int8)

    float_types = [float, float, np.float32, np.float64]
    bitdepth = float_types[0]
    signal = np.array(signal, dtype=bitdepth)
    # create .wav file
    wavfile.write(fileName, RenderARGS["fs"], signal)
    return signal


# map containing the room split by materials, indexing path to file and material properties
meshMatMap = {
    "Theatre_Wood_Parquet": {
        "stlFileName": "Theatre_Wood_Parquet.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Walls": {
        "stlFileName": "Theatre_Wood_Walls.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Deco": {
        "stlFileName": "Theatre_Wood_Deco.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
    "Theatre_Limestone": {
        "stlFileName": "Theatre_Limestone.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
    "Theatre_Plaster": {
        "stlFileName": "Theatre_Plaster.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
    "Theatre_Fibre": {
        "stlFileName": "Theatre_Fibre.stl",
        "material": pra.Material(
            energy_absorption={
                "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
            },
            scattering=0.0,
        ),
    },
}


# sources/mic locations ( TODO : Remplacer la boucle inrange par in map.items pour le script final)
# sourcesMap = {
#     "A": [-1.75, 9.15, 3.3572],
#     "B": [1.75, 9.15, 3.3572],
#     "C": [3.0, 2.0, 3.3572],
#     "D": [-3.0, 2.0, 3.3572],
#     "E": [-3.35, -1.0, 1.4],
#     "F": [0.0, -1.0, 1.4],
#     "G": [3.35, -1.0, 1.4],
# }

# microphonesMap = {
#     1: [-3.8, -3.75, 1.3],
#     2: [3.8, -3.75, 1.3],
#     3: [2.4077, -7.3239, 1.3],
#     4: [-2.4077, -7.3239, 1.3],
#     5: [-5.0, -2.0, 3.5],
#     6: [5.0, -2.0, 3.5],
#     7: [3.5, -8.2, 3.5],
#     8: [-3.5, -8.2, 3.5],
#     9: [-5.1, -2.0, 5.8],
#     10: [5.1, -2.0, 5.8],
#     11: [4.0, -8.5, 5.8],
#     12: [-4.0, -8.5, 5.8],
#     13: [-5.1, -2.0, 8.2],
#     14: [5.1, -2.0, 8.2],
#     15: [4.0, -8.5, 8.2],
#     16: [-4.0, -8.5, 8.2],
# }

# Build room from geometry
walls = []
for k, v in meshMatMap.items():

    # import des fichiers stl
    stlFileName = v["stlFileName"]
    the_mesh = mesh.Mesh.from_file(Path(f"PyroomMeshes/ReworkedMeshes/{stlFileName}"))
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 1  # to get a realistic room size (not 3km)

    # create one wall per triangle
    for w in range(ntriang):
        # appliquer les matériaux indexés dans materials
        walls.append(
            pra.wall_factory(
                the_mesh.vectors[w].T / size_reduc_factor,
                v["material"].energy_absorption["coeffs"],
                v["material"].scattering["coeffs"],
            )
        )
t.show("Done STL imports")

# Instanciating room with geometry and some render parameters
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
        # [-5.1, -2.0, 5.8],# <-- D4
        # [5.1, -2.0, 5.8],
        # [4.0, -8.5, 5.8],
        # [-4.0, -8.5, 5.8],
        # [-5.1, -2.0, 8.2],
        # [5.1, -2.0, 8.2],
        # [4.0, -8.5, 8.2],
        # [-4.0, -8.5, 8.2],
    ],
)


# attempting to add source in room externally to catch a common unresolved persistent error
atmptSources = 0
while atmptSources < 3:
    try:
        room.add_source(
            # [-1.75, 9.15, 3.3572],
            # [1.75, 9.15, 3.3572]
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

room.set_ray_tracing(
    n_rays=RenderARGS["RT_n_rays"], receiver_radius=RenderARGS["RT_receiver_radius"]
)  # default =0.5

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

# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
computedIRs = room.rir

if len(computedIRs) == len(room.mic_array):
    # needed since we iterate from a map/dict with arbitrary int IDs
    for i in range(0, len(room.mic_array)):

        folderpath = f"{RenderARGS["exportPath"]}"
        # wavFileName = f"{name}-{i+1}.wav"
        wavFileName = f"{name}.wav"
        fileName = f"{folderpath}/{wavFileName}"

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        signal = exportIRToWav(
            computedIRs=computedIRs,
            norm=False,
            fileName=fileName,
            micIndex=(i),
        )

        t.show(f">Export {wavFileName} {i+1}/{len(computedIRs)}")

else:
    t.show(
        f"There is {len(computedIRs)} computed IRs for {len(room.mic_array)} microphones"
    )
    raise Exception(f"IR data is missing some Microphones indexes")

# show the room
room.plot(img_order=RenderARGS["IMS_Order"])
plt.show()
