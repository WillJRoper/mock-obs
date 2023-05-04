import sys
import h5py
import numpy as np
from astropy.cosmology import Planck18 as cosmo

import matplotlib.pyplot as plt
import matplotlib as mpl

from synthesizer.imaging.images import ParticleImage
from synthesizer.kernel_functions import quintic

from utilities import total_lum


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
resolution = 0.031 / cosmo.arcsec_per_kpc_proper(z).value
width = 500

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
pos = reg_snap_grp["Particle"]["S_Coordinates"][...]
s_mass = reg_snap_grp["Particle"]["S_Mass"][...]
ini_masses = reg_snap_grp["Particle"]["S_MassInitial"][...]
s_mets = reg_snap_grp["Particle"]["S_Z"][...]
ages = reg_snap_grp["Particle"]["S_Age"][...]
los = reg_snap_grp["Particle"]["S_los"][...]

print("Got data...")

# Extract this groups data
print(grps[0], group_id)
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
subgrp_start = []
subgrp_length = []
for (ind, start), length in zip(enumerate(grp_s_begin), grp_s_length):
    print(start, length)
    subgrp_start.append(len(grp_los))
    subgrp_length.append(length)
    grp_pos.extend(pos[start: start + length, :])
    grp_s_mass.extend(s_mass[start: start + length])
    grp_ini_masses.extend(ini_masses[start: start + length])
    grp_s_mets.extend(s_mets[start: start + length])
    grp_ages.extend(ages[start: start + length])
    grp_los.extend(los[start: start + length])
grp_pos = np.array(grp_pos)
grp_s_mass = np.array(grp_s_mass)
grp_ini_masses = np.array(grp_ini_masses)
grp_s_mets = np.array(grp_s_mets)
grp_ages = np.array(grp_ages)
grp_los = np.array(grp_los)
subgrp_start = np.array(subgrp_start)
subgrp_length = np.array(subgrp_length)

# Calculate the geometric centre of the group
centre = np.mean(grp_pos, axis=0)
print(centre)
print("Got the group data with %d particles" % len(grp_los))

# Compute luminosities for this group
lums = total_lum(grp_ini_masses, grp_s_mets, grp_ages, grp_los,
                 kappa=0.0795, BC_fac=1)

print("Got luminosities...")

# Get the group luminosity image
grp_lum_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=pos,
    pixel_values=lums,
    smoothing_lengths=smls,
    centre=centre
)
grp_lum_img = grp_lum_obj.get_smoothed_img(quintic)

# Get the group mass image
grp_mass_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=pos,
    pixel_values=masses,
    smoothing_lengths=smls,
    centre=centre
)
grp_mass_img = grp_mass_obj.get_smoothed_img(quintic)

# Create an array of all lums and populate the ones we have calculated
all_lums = np.full(len(los), np.nan)
all_lums[okinds] = lums

# Set up colormap and normalisation
norm = mpl.colors.Normalize(vmin=0, vmax=len(subgrp_start))
cmap = plt.get_cmap("plasma")

# Loop over subgroups and create subfind labelled image
subfind_img = np.zeros((grp_lum_img.shape[0], grp_lum_img.shape[1], 4))
subfind_id = 0
for start, length in zip(subgrp_start, subgrp_length):

    # Get the subgroup mass image
    subgrp_mass_obj = ParticleImage(
        resolution,
        fov=width,
        cosmo=cosmo,
        positions=pos,
        pixel_values=masses,
        smoothing_lengths=smls,
        centre=centre
    )
    subgrp_mass_img = subgrp_mass_obj.get_smoothed_img(quintic)

    # Get the mask for nonzero pixels
    mask = subgrp_mass_img > 0

    # Create an image to hold this subgroup
    subgrp_img = np.zeros(subfind_img.shape)
    subgrp_img[mask] = cmap(norm(subfind_id))

    # Set the alpha
    subgrp_img[:, :, -1] = alpha

    # Add it to the main image
    subfind_img += subgrp_img
    
    subfind_id += 1
    
# Create segmentation map

# Create plot
fig = plt.figure()
ax1 = fig.add_suplot(221)
ax2 = fig.add_suplot(222)
ax3 = fig.add_suplot(223)
ax4 = fig.add_suplot(224)

# Plot images
ax1.imshow(grp_mass_img)
ax2.imshow(grp_lum_img)
ax3.imshow(subfind_img)

fig.savefig("plots/source_ident_comp.png", bbox_inches="tight", dpi=100)
plt.close(fig)

