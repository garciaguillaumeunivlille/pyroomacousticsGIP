import os
import numpy as np
import pyroomacoustics as pra
from pathlib import Path
from scipy.io import wavfile
from stl import mesh


# IR computing parameters, except [Material, Mesh, Source-Micros locations]
RenderARGS = {
    "exportPath": f"Generated-IRs/gitIgnored/5-11",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
    "filenamePrefix": f"issue-script-1",
}

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


def customExportIRToWav(computedIRs, fileName, micIndex):
    signal = computedIRs[micIndex][0]  # [micro][source]
    signal = np.array(signal, dtype=np.float32)
    # map to 16-bit
    max_16bit = 2**15
    signal = signal * max_16bit
    # change the data type to PCM 16 bits
    signal = signal.astype(np.int16)
    # create .wav file
    wavfile.write(fileName, RenderARGS["fs"], signal)
    return signal


# Build room from geometry
walls = []
for k, v in meshMatMap.items():

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


# Instanciating room with geometry and some render parameters
room = pra.Room(
    walls,
    fs=RenderARGS["fs"],
    max_order=RenderARGS["IMS_Order"],
    ray_tracing=RenderARGS["useRayTracing"],
    air_absorption=True,
).add_microphone_array(
    np.c_[[-3.8, -3.75, 1.3],]
)

# attempting to add source in room externally to catch a common unresolved persistent error
atmptSources = 0
while atmptSources < 3:
    try:
        room.add_source([-3.0, 2.0, 3.3572])
        break
    except ValueError:
        atmptSources += 1
        print(f">>>failed adding source {atmptSources}/3 attempts")
print(f"added source OK")

if RenderARGS["useRayTracing"]:
    room.set_ray_tracing(
        n_rays=RenderARGS["RT_n_rays"], receiver_radius=RenderARGS["RT_receiver_radius"]
    )

simulationAttempts = 0
while simulationAttempts < 5:
    try:
        # compute the rir
        print("processing image_source_model...")
        room.image_source_model()
        if RenderARGS["useRayTracing"]:
            print("processing ray_tracing...")
            room.ray_tracing()
        print("compute_rir")
        room.compute_rir()
        print("plot_rir")
        room.plot_rir()
        break
    except (ValueError, RuntimeError) as e:
        print(f"compute_rir failed with {str(e)}")
        simulationAttempts += 1
        print(f">>>simulation failed {simulationAttempts}/5 attempts")
print(f"simulation OK")


# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
computedIRs = room.rir

if len(computedIRs) == len(room.mic_array):
    for i in range(0, len(room.mic_array)):

        folderpath = f"{RenderARGS["exportPath"]}"
        wavFileName = f"{RenderARGS["filenamePrefix"]}.wav"
        fileName = f"{folderpath}/{wavFileName}"

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        signal = customExportIRToWav(
            computedIRs=computedIRs,
            fileName=fileName,
            micIndex=(i),
        )
    print(f">Export {wavFileName} {i+1}/{len(computedIRs)}")
else:
    print(
        f"There is {len(computedIRs)} computed IRs for {len(room.mic_array)} microphones"
    )
    raise Exception(f"IR data is missing some Microphones indexes")
