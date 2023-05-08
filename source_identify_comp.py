import sys
import h5py
import numpy as np
from astropy.cosmology import Planck18 as cosmo

import matplotlib.pyplot as plt
import matplotlib as mpl

from synthesizer.imaging.images import ParticleImage
from synthesizer.kernel_functions import quintic

from utilities import total_lum, lum_to_flux

from unyt import kpc, erg, s, Hz, Msun, Mpc, nJy


# Which group and snapshot are we doing?
group_id = int(sys.argv[1])
snap_ind = int(sys.argv[2])
reg = sys.argv[3].zfill(2)

# What alpha will we use?
alpha = float(sys.argv[4])

# Get what snapshot we are doing
tags = flares_snaps = ['001_z014p000', '002_z013p000', '003_z012p000',
                       '004_z011p000', '005_z010p000', '006_z009p000',
                       '007_z008p000', '008_z007p000',
                       '009_z006p000', '010_z005p000']
snap = tags[snap_ind]

# Get redshift
z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

# Define image properties
resolution = 10 * 0.031 / cosmo.arcsec_per_kpc_proper(z).value * kpc
width = 200 * kpc
print("Making images with %.2f kpc resolution and a %.2f FOV" % (resolution, width))

# Define the path to the data
datapath = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
    + "flares.hdf5"

# Extract data from data file
hdf = h5py.File(datapath, "r")
reg_snap_grp = hdf[reg][snap]
grps = reg_snap_grp["Galaxy"]["GroupNumber"][...]
subgrps = reg_snap_grp["Galaxy"]["SubGroupNumber"][...]
s_length = reg_snap_grp["Galaxy"]["S_Length"][...]
s_begin = np.zeros(len(s_length), dtype=int)
s_begin[1:] = np.cumsum(s_length[:-1])
pos = reg_snap_grp["Particle"]["S_Coordinates"][...].T / (1 + z)
s_mass = reg_snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
ini_masses = reg_snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
s_mets = reg_snap_grp["Particle"]["S_Z"][...]
ages = reg_snap_grp["Particle"]["S_Age"][...]
los = reg_snap_grp["Particle"]["S_los"][...]
smls =reg_snap_grp["Particle"]["S_sml"][...]

print("Got data...")

# Extract this groups data
okinds = grps == group_id
grp_subgrps = subgrps[okinds]
grp_s_length = s_length[okinds]
grp_s_begin = s_begin[okinds]

# And make particle arrays for this group
grp_pos = []
grp_s_mass = []
grp_ini_masses = []
grp_s_mets = []
grp_ages = []
grp_los = []
grp_smls = []
subgrp_start = []
subgrp_length = []
for (ind, start), length in zip(enumerate(grp_s_begin), grp_s_length):
    subgrp_start.append(len(grp_los))
    subgrp_length.append(length)
    grp_pos.extend(pos[start: start + length, :])
    grp_s_mass.extend(s_mass[start: start + length])
    grp_ini_masses.extend(ini_masses[start: start + length])
    grp_s_mets.extend(s_mets[start: start + length])
    grp_ages.extend(ages[start: start + length])
    grp_los.extend(los[start: start + length])
    grp_smls.extend(smls[start: start + length])
grp_pos = np.array(grp_pos)
grp_s_mass = np.array(grp_s_mass)
grp_ini_masses = np.array(grp_ini_masses)
grp_s_mets = np.array(grp_s_mets)
grp_ages = np.array(grp_ages)
grp_los = np.array(grp_los)
grp_smls = np.array(grp_smls)
subgrp_start = np.array(subgrp_start)
subgrp_length = np.array(subgrp_length)

# Calculate the geometric centre of the group
centre = np.mean(grp_pos, axis=0)

print("Got the group data with %d particles" % len(grp_los))

# Compute luminosities for this group
lums = total_lum(grp_ini_masses, grp_s_mets, grp_ages, grp_los,
                 kappa=0.0795, BC_fac=1)

# Concert luminosity to flux
lums = lum_to_flux(lums, cosmo, z)

print("Got luminosities...")

# Get the group luminosity image
grp_lum_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_pos * Mpc,
    pixel_values=lums * nJy,
    smoothing_lengths=grp_smls * Mpc,
    centre=centre
)
grp_lum_img = grp_lum_obj.get_smoothed_img(quintic)

print("Got Luminosity Image")

# Get the group mass image
grp_mass_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_pos * Mpc,
    pixel_values=grp_s_mass * Msun,
    smoothing_lengths=grp_smls * Mpc,
    centre=centre
)
grp_mass_img = grp_mass_obj.get_smoothed_img(quintic)

print("Got Mass Image")

# Set up colormap and normalisation
norm = mpl.colors.Normalize(vmin=0, vmax=len(subgrp_start))
cmap = plt.get_cmap("plasma")

# Loop over subgroups and create subfind labelled image
subfind_img = np.zeros((grp_lum_img.shape[0], grp_lum_img.shape[1], 4))
subfind_id = 0
for start, length in zip(subgrp_start, subgrp_length):

    print("Making an image for subgroup %d" % subfind_id)

    # Get the subgroup mass image
    subgrp_mass_obj = ParticleImage(
        resolution,
        fov=width,
        cosmo=cosmo,
        positions=grp_pos[start: start + length, :] * Mpc,
        pixel_values=grp_s_mass[start: start + length] * Msun,
        smoothing_lengths=grp_smls[start: start + length] * Mpc,
        centre=centre
    )
    subgrp_mass_img = subgrp_mass_obj.get_smoothed_img(quintic)

    # Get the mask for nonzero pixels
    mask = subgrp_mass_img > 0

    # Create an image to hold this subgroup
    subgrp_img = np.zeros(subfind_img.shape)
    subgrp_img[mask] = cmap(norm(subfind_id))

    # Add it to the main image
    subfind_img[mask] = (
        (subfind_img[mask] * (1 - alpha)) + (subgrp_img[mask] * alpha)
    )
    
    subfind_id += 1

print("Got SUBFIND image")
    
# Create segmentation map

# Create plot
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# Plot images
ax1.imshow(grp_mass_img, norm=mpl.colors.Normalize(
    vmin=grp_s_mass.min(),
    vmax=grp_mass_img.max()
    - (grp_mass_img.max() * 0.3))
           )
ax2.imshow(grp_lum_img, norm=mpl.colors.Normalize(
    vmin=grp_lum_img[grp_lum_img != 0].min(),
    vmax=grp_lum_img.max()
    - (grp_lum_img.max() * 0.3))
           )
ax3.imshow(subfind_img)

fig.savefig("plots/source_ident_comp_%s_%s_%d.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100)
plt.close(fig)

