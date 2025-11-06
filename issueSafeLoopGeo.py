import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
from scipy.io import wavfile
from timer import Timer
from stl import mesh

# Timer to log elapsed time
t = Timer()
t.start()

# IR computing parameters, except [Material, Mesh, Source-Micros locations]
RenderARGS = {
    "exportPath": "Generated-IRs/issueFullLoop/Geo",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
}


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
        "stlFileName": "Theatre_Wood_Parquet.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.07,
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Walls": {
        "stlFileName": "Theatre_Wood_Walls.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.06,
            scattering=0.0,
        ),
    },
    "Theatre_Wood_Deco": {
        "stlFileName": "Theatre_Wood_Deco.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.04,
            scattering=0.0,
        ),
    },
    "Theatre_Limestone": {
        "stlFileName": "Theatre_Limestone.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.02,
            scattering=0.0,
        ),
    },
    "Theatre_Plaster": {
        "stlFileName": "Theatre_Plaster.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.01,
            scattering=0.0,
        ),
    },
    "Theatre_Fibre": {
        "stlFileName": "Theatre_Fibre.Flip.Geo.stl",
        "material": pra.Material(
            energy_absorption=0.2,
            scattering=0.0,
        ),
    },
}

sourcesMapGeo = {
    "A": [-1.7511, 11.1895, -0.8404],
    "B": [1.7489, 11.1895, -0.8404],
    "C": [2.9989, 4.0395, -0.8404],
    "D": [-3.0011, 4.0395, -0.8404],
    "E": [-3.3511, 1.0395, -2.7976],
    "F": [-0.0011, 1.0395, -2.7976],
    "G": [3.3489, 1.0395, -2.7976]
}

microphonesMapGeo = {
    1: [-3.8011, -1.7105, -2.8976],
    2: [3.7989, -1.7105, -2.8976],
    3: [2.4066, -5.2844, -2.8976],
    4: [-2.4088, -5.2844, -2.8976],
    5: [-5.0011, 0.0395, -0.6976],
    6: [4.9989, 0.0395, -0.6976],
    7: [3.4989, -6.1605, -0.6976],
    8: [-3.5011, -6.1605, -0.6976],
    9: [-5.1011, 0.0395, 1.6024],
    10:[5.0989, 0.0395, 1.6024],
    11:[3.9989, -6.4605, 1.6024],
    12:[-4.0011, -6.4605, 1.6024],
    13:[-5.1011, 0.0395, 4.0024],
    14:[5.0989, 0.0395, 4.0024],
    15:[3.9989, -6.4605, 4.0024],
    16:[-4.0011, -6.4605, 4.0024]
}

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


for times in range(0, 2):
    for sourceLabel, sourcePos in sourcesMapGeo.items():
        micIndex = 0
        for micID, micPos in microphonesMapGeo.items():

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
                    break
                except (ValueError, RuntimeError) as e:
                    t.show(f"compute_rir failed with {str(e)}")
                    simulationAttempts += 1
                    t.show(f">>>simulation failed {simulationAttempts}/5 attempts")
            t.show(f"simulation OK")

            # The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
            computedIRs = room.rir

            if len(computedIRs) == len(room.mic_array):

                folderpath = f"{RenderARGS["exportPath"]}"
                wavFileName = f"{sourceLabel}{micID}-{times+1}.wav"
                fileName = f"{folderpath}/{wavFileName}"

                if not os.path.exists(folderpath):
                    os.makedirs(folderpath)

                signal = customExportIRToWav(computedIRs=computedIRs, fileName=fileName)

                t.show(f">Export {wavFileName} {micIndex+1}/{len(computedIRs)}")
                micIndex += 1
            else:
                t.show(
                    f"There is {len(computedIRs)} computed IRs for {len(room.mic_array)} microphones"
                )
                raise Exception(f"IR data is missing some Microphones indexes")
