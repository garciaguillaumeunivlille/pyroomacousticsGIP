import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import pyroomacoustics as pra
from timer import Timer
from scipy.io import wavfile

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
    "exportPath": "Generated-IRs/gitIgnored/SimpleTest/27-10",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
}


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


name = "8Bands-inria-parquet"
 

 

# material = pra.Material(       
#     energy_absorption={
#         "coeffs": [0.01],
#         "center_freqs": [62.5]
#         },
#     scattering=0.0
# )   

  
material = pra.Material(0.1,0.0)


# with numpy-stl
the_mesh = mesh.Mesh.from_file("examples/data/INRIA_MUSIS.stl") 
ntriang, nvec, npts = the_mesh.vectors.shape
size_reduc_factor = 500  # to get a realistic room size (not 3km)
# size_reduc_factor = 1  # to get a realistic room size (not 3km)
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
        # z 0-10  y0-12 z0-6
        [-4.000, 7.0000, 0.5000],
        [-8.0000, 7.0000, 3.0000],
        [-4.0000, 1.0000, 0.5000],
        [-7.0000, 4.0000, 0.5000],
        [-1.0000, 4.0000, 0.5000],
        [-4.0000, 7.0000, 3.0000],
        [-4.0000, 1.0000, 3.0000],
        [-7.0000, 4.0000, 3.0000],
        [-1.0000, 4.0000, 3.0000],
        [-4.0000, 10.0000, 3.0000],
    ]
)
  
# attempting to add source in room externally to catch a common unresolved persistent error
atmptSources = 0
while atmptSources < 3:
    try:
        room.add_source([-1,1,1])
        break
    except ValueError:
        atmptSources += 1
        t.show(f">>>failed adding source {atmptSources}/3 attempts")
t.show(f"added source OK")

# for j in range(1,9):

# compute the rir
room.image_source_model()
room.ray_tracing()
room.compute_rir()
room.plot_rir()


# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
computedIRs = room.rir
if len(computedIRs) == len(room.mic_array):
    # needed since we iterate from a map/dict with arbitrary int IDs
    for i in range(0, len(room.mic_array)):

        folderpath = f"{RenderARGS["exportPath"]}"#/{j}
        #{j}_
        wavFileName = f"{name}-{i+1}.wav"
        fileName = f"{folderpath}/{wavFileName}"

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        # signal = exportIRToWav(
        #     computedIRs=computedIRs,
        #     norm=False,
        #     fileName=f"{fileName}",
        #     micIndex=(i),
        # )

        t.show(f">Export {wavFileName} {i+1}/{len(computedIRs)}")

else:
    t.show(
            f"There is {len(computedIRs)} computed IRs for {len(room.mic_array)} microphones"
        )
    raise Exception(f"IR data is missing some Microphones indexes")

# show the room
# room.plot(img_order=1)
# plt.show()
