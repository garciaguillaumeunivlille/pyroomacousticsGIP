"""
This sample program demonstrate how to import a model from an STL file.
Currently, the materials need to be set in the program which is not very practical
when different walls have different materials.

The STL file was kindly provided by Diego Di Carlo (@Chutlhu).
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import pyroomacoustics as pra
from scipy.io import wavfile

try:
    from stl import mesh
except ImportError as err:
    print(
        "The numpy-stl package is required for this example. "
        "Install it with `pip install numpy-stl`"
    )
    raise err

#default_stl_path = Path(__file__).parent / "../PyroomMeshes/TheatreJoined.001.stl"
#default_stl_path = "../PyroomMeshes/cubic10m_outside.stl"
#default_stl_path = "../PyroomMeshes/cube_both.stl"
#default_stl_path = "../examples/data/INRIA_MUSIS.stl"
#default_stl_path = "../PyroomMeshes/cubic10m.stl"
#default_stl_path = "../PyroomMeshes/INRIA_MUSIC_scaled.stl"
#default_stl_path = "../PyroomMeshes/cube_subdiv1.stl"
#default_stl_path = "../PyroomMeshes/cube_subdiv2.stl"
#default_stl_path = "../PyroomMeshes/cube_subdiv3.stl"
#default_stl_path = "../PyroomMeshes/cube_subdiv4.stl"
#default_stl_path = "../PyroomMeshes/Theatre.stl"
#default_stl_path = "../PyroomMeshes/theatre_decimate.stl"
#default_stl_path = "../PyroomMeshes/ReworkedMeshes/TheatreP_Walls_OUT.stl"
default_stl_path = "../PyroomMeshes/theatre_out1.stl"

fs = 44100

def exportIRToWav(computedIRs, norm, fileName, micIndex):
    signal = computedIRs[micIndex][0]  # [micro][source]
    if norm:
        from utilities import normalize

        signal = normalize(signal, bits=np.int8)
    float_types = [float, float, np.float32, np.float64]
    bitdepth = float_types[0]
    signal = np.array(signal, dtype=bitdepth)
    # create .wav file
    wavfile.write(fileName, fs, signal)
    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic room from STL file example")
    parser.add_argument(
        "--file", type=str, default=default_stl_path, help="Path to STL file"
    )
    args = parser.parse_args()

    # Define the materials array
    test_mat = {
    "description": "Example ceiling material",
    "coeffs": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
    }    
    material = pra.Material(energy_absorption=0.001, scattering=0.0)

    # with numpy-stl
    the_mesh = mesh.Mesh.from_file(args.file)
    ntriang, nvec, npts = the_mesh.vectors.shape
    size_reduc_factor = 1  # to get a realistic room size (not 3km)

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
        #print(the_mesh.vectors[w].T)
    print("WALLS")
    #for i in len(walls):
    #    print(walls[i])

    room = (
        pra.Room(
            walls,
            fs=44100,
            max_order=2,
            ray_tracing=True,
            air_absorption=True,
        )
        .add_source([0.0, 0, 5.0])
        .add_microphone_array(np.c_[[0, 0, 6]])
    )
    room.set_ray_tracing(n_rays=5000, receiver_radius=2)  # default =0.5
    # compute the rir
    print("processing image_source_model...")
    room.image_source_model()
    print("processing ray_tracing...")
    room.ray_tracing()
    print("compute_rir")
    room.compute_rir()
    room.plot_rir()

    #room.rir
    signal = exportIRToWav(
                computedIRs=room.rir,
                norm=False,
                fileName="ir.wav",
                micIndex=0,
            )

    # show the room
    room.plot(img_order=1)
    plt.show()