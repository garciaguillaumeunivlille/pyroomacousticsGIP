import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
import pyroomacoustics as pra


corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
room = pra.Room.from_corners(corners)

# specify signal source
fs, signal = wavfile.read("../CustomSamples/IR-Dirac-44100-20hz-22050hz-1s.wav")

mat1Bande = pra.Material(energy_absorption=0.01, scattering=0.15)

# matStrRef = pra.Material(energy_absorption="rough_concrete", scattering=0.15)
# print(matStrRef.absorption_coeffs)

# matXBandes = pra.Material(
#             energy_absorption={
#                 "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05,0.01,0.05],
#                 "center_freqs": [125, 250, 500, 1000, 2000, 4000,45000,50000],
#             }, #Nombre de bandes de fréquences au choix
#             scattering=0.54,
#         )
# print(matXBandes.absorption_coeffs)


# set max_order to a low value for a quick (but less accurate) RIR
room = pra.Room.from_corners(corners, fs=fs, max_order=2, materials=mat1Bande, ray_tracing=True, air_absorption=True)
room.extrude(2., materials=pra.Material(0.2, 0.15))

# Set the ray tracing parameters
room.set_ray_tracing(receiver_radius=0.5, energy_thres=1e-5)

# add source and set the signal to WAV file content
room.add_source([1., 1., 0.5], signal=signal)

# add two-microphone array
R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])  # [[x], [y], [z]]
room.add_microphone(R)

# compute image sources
room.image_source_model()

print(room.rt_args["n_rays"])

# visualize 3D polyhedron room and image sources
fig, ax = room.plot(img_order=2)

# Pour 8 faces: 
# img_order=1 = 7 (F-1)
# img_order=2 = 7 (F*3)
# img_order=>1 = 24 (F*3)

plt.show()