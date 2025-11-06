from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
from scipy.io import wavfile
from timer import Timer

try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err


# Timer to log elapsed time
t = Timer()
t.start()

# IR computing parameters, except [Material, Mesh, Source-Micros locations]
##############
isCasA = False
##############
CasePrefix = "CasA" if isCasA else "CasB"
RenderARGS = {
    "isCasA": isCasA,
    "exportPath": f"Generated-IRs/gitIgnored/04-11/{CasePrefix}",
    "fs": 44100,
    "IMS_Order": 1,
    "useRayTracing": True,
    "RT_receiver_radius": 2,
    "RT_n_rays": 5000,
    "filenamePrefix": f"{CasePrefix}-flippedJoined",
}

CasAMat = pra.Material(
    energy_absorption={
        "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "center_freqs": [62.5, 125, 250, 500, 1000, 2000, 4000, 8000],
    },
    scattering=0.0,
)
CaseBMat = pra.Material(0.01, 0.0)

RenderMat = CasAMat if isCasA else CaseBMat
print(RenderMat)


def customExportIRToWav(computedIRs, norm, fileName, micIndex):
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
        "stlFileName": "Theatre_Wood_Parquet.Flip.stl",
        "material": RenderMat,
    },
    "Theatre_Wood_Walls": {
        "stlFileName": "Theatre_Wood_Walls.Flip.stl",
        "material": RenderMat,
    },
    "Theatre_Wood_Deco": {
        "stlFileName": "Theatre_Wood_Deco.Flip.stl",
        "material": RenderMat,
    },
    "Theatre_Limestone": {
        "stlFileName": "Theatre_Limestone.Flip.stl",
        "material": RenderMat,
    },
    "Theatre_Plaster": {"stlFileName": "Theatre_Plaster.stl", "material": RenderMat},
    "Theatre_Fibre": {"stlFileName": "Theatre_Fibre.stl", "material": RenderMat},
}

fullMesh = {
    "Theatre_Wood_Parquet": {
        "stlFileName": "Joined-TheatreEnveloppeOUT_filpped.stl",
        "material": RenderMat,
    }
}


# Build room from geometry
walls = []
for k, v in fullMesh.items():

    # import des fichiers stl
    stlFileName = v["stlFileName"]
    the_mesh = mesh.Mesh.from_file(Path(f"PyroomMeshes/ReworkedMeshes/{stlFileName}"))
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 1  # to get a realistic room size (not 3km)

    # create one wall per triangle
    for w in range(ntriang):
        # appliquer les matériaux indexés dans materials
        inverse_triangle = np.array(
            [
                the_mesh.vectors[w].T[2],
                the_mesh.vectors[w].T[1],
                the_mesh.vectors[w].T[0],
            ]
        )
        walls.append(
            pra.wall_factory(
                inverse_triangle,
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
    np.c_[[-3.8, -3.75, 1.3],]
)

anechoicAudioSource = wavfile.read(
    # "CustomSamples/Basic-808-Clap.wav"
    "CustomSamples/IR-Dirac-44100-20hz-22050hz-1s.wav"
)
# attempting to add source in room externally to catch a common unresolved persistent error
atmptSources = 0
while atmptSources < 3:
    try:
        room.add_source([-3.0, 2.0, 3.3572], anechoicAudioSource)
        break
    except ValueError:
        atmptSources += 1
        t.show(f">>>failed adding source {atmptSources}/3 attempts")
t.show(f"added source OK")

if RenderARGS["useRayTracing"]:
    room.set_ray_tracing(
        n_rays=RenderARGS["RT_n_rays"], receiver_radius=RenderARGS["RT_receiver_radius"]
    )

# compute the rir
# t.show("processing image_source_model...")
# room.image_source_model()
# if RenderARGS["useRayTracing"]:
#     t.show("processing ray_tracing...")
#     room.ray_tracing()
t.show("compute_rir")
room.compute_rir()
t.show("plot_rir")
room.plot_rir()


# The attribute rir is a list of lists so that the outer list is on microphones and the inner list over sources.
computedIRs = room.rir

if len(computedIRs) == len(room.mic_array):
     for i in range(0, len(room.mic_array)):

        folderpath = f"{RenderARGS["exportPath"]}"
        # wavFileName = f"{name}-{i+1}.wav"
        wavFileName = f"{RenderARGS["filenamePrefix"]}.wav"
        fileName = f"{folderpath}/{wavFileName}"

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        signal = customExportIRToWav(
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
