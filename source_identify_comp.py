import sys
import os
import h5py
import numpy as np
from astropy.cosmology import Planck18 as cosmo

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from synthesizer.imaging.images import ParticleImage
from synthesizer.kernel_functions import quintic

import webbpsf
from utilities import total_lum, lum_to_flux
import photutils as phut
from unyt import kpc, erg, s, Hz, Msun, Mpc, nJy, pc


# Which group and snapshot are we doing?
group_id = int(sys.argv[1])
snap_ind = int(sys.argv[2])
reg = sys.argv[3].zfill(2)

# What alpha will we use?
alpha = float(sys.argv[4])

# Get some image properties
downsample = float(sys.argv[5])
width = float(sys.argv[6])
noise = float(sys.argv[7])

# Get what snapshot we are doing
tags = flares_snaps = ['001_z014p000', '002_z013p000', '003_z012p000',
                       '004_z011p000', '005_z010p000', '006_z009p000',
                       '007_z008p000', '008_z007p000',
                       '009_z006p000', '010_z005p000']
snap = tags[snap_ind]

# Get redshift
z_str = snap.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

soft = 0.001802390 / (0.6777 * (1 + z)) 

# Define image properties
resolution = downsample * 0.031 / cosmo.arcsec_per_kpc_proper(z).value * kpc
width = width * kpc
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
s_pos = reg_snap_grp["Particle"]["S_Coordinates"][...].T / (1 + z)
s_mass = reg_snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
ini_masses = reg_snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
s_mets = reg_snap_grp["Particle"]["S_Z"][...]
ages = reg_snap_grp["Particle"]["S_Age"][...] * 10 ** 3
los = reg_snap_grp["Particle"]["S_los"][...]
s_smls = reg_snap_grp["Particle"]["S_sml"][...]

dm_length = reg_snap_grp["Galaxy"]["DM_Length"][...]
dm_begin = np.zeros(len(dm_length), dtype=int)
dm_begin[1:] = np.cumsum(dm_length[:-1])
dm_pos = reg_snap_grp["Particle"]["DM_Coordinates"][...].T / (1 + z)

g_length = reg_snap_grp["Galaxy"]["G_Length"][...]
g_begin = np.zeros(len(g_length), dtype=int)
g_begin[1:] = np.cumsum(g_length[:-1])
g_pos = reg_snap_grp["Particle"]["G_Coordinates"][...].T / (1 + z)
g_mass = reg_snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
g_smls = reg_snap_grp["Particle"]["G_sml"][...]

print("Got data...")

# Make plot directory
if not os.path.exists("plots/%s_%s_%d" % (snap, reg, group_id)):
   os.makedirs("plots/%s_%s_%d" % (snap, reg, group_id))

# Extract this groups data
okinds = grps == group_id
grp_subgrps = subgrps[okinds]
grp_s_length = s_length[okinds]
grp_s_begin = s_begin[okinds]
grp_dm_length = dm_length[okinds]
grp_dm_begin = dm_begin[okinds]
grp_g_length = g_length[okinds]
grp_g_begin = g_begin[okinds]

# And make particle arrays for this group
grp_s_pos = []
grp_dm_pos = []
grp_g_pos = []
grp_s_mass = []
grp_dm_mass = []
grp_g_mass = []
grp_ini_masses = []
grp_s_mets = []
grp_ages = []
grp_los = []
grp_s_smls = []
grp_dm_smls = []
grp_g_smls = []
subgrp_sstart = []
subgrp_slength = []
subgrp_dmstart = []
subgrp_dmlength = []
subgrp_gstart = []
subgrp_glength = []
for (ind, sstart), slength, dmstart, dmlength, gstart, glength in zip(
        enumerate(grp_s_begin), grp_s_length,
        grp_dm_begin, grp_dm_length,
        grp_g_begin, grp_g_length):
    
    subgrp_sstart.append(len(grp_los))
    subgrp_slength.append(slength)
    subgrp_dmstart.append(len(grp_dm_mass))
    subgrp_dmlength.append(dmlength)
    subgrp_gstart.append(len(grp_g_mass))
    subgrp_glength.append(glength)
    
    grp_s_pos.extend(s_pos[sstart: sstart + slength, :])
    grp_s_mass.extend(s_mass[sstart: sstart + slength])
    grp_ini_masses.extend(ini_masses[sstart: sstart + slength])
    grp_s_mets.extend(s_mets[sstart: sstart + slength])
    grp_ages.extend(ages[sstart: sstart + slength])
    grp_los.extend(los[sstart: sstart + slength])
    grp_s_smls.extend(s_smls[sstart: sstart + slength])

    grp_dm_pos.extend(dm_pos[dmstart: dmstart + dmlength, :])
    grp_dm_mass.extend(np.ones(dmlength))
    grp_dm_smls.extend(np.full(dmlength, soft))
    
    grp_g_pos.extend(g_pos[gstart: gstart + glength, :])
    grp_g_mass.extend(g_mass[gstart: gstart + glength])
    grp_g_smls.extend(g_smls[gstart: gstart + glength])
    
grp_s_pos = np.array(grp_s_pos)
grp_s_mass = np.array(grp_s_mass)
grp_ini_masses = np.array(grp_ini_masses)
grp_s_mets = np.array(grp_s_mets)
grp_ages = np.array(grp_ages)
grp_los = np.array(grp_los)
grp_s_smls = np.array(grp_s_smls)
subgrp_sstart = np.array(subgrp_sstart)
subgrp_slength = np.array(subgrp_slength)

subgrp_dmstart = np.array(subgrp_dmstart)
subgrp_dmlength = np.array(subgrp_dmlength)
grp_dm_pos = np.array(grp_dm_pos)
grp_dm_mass = np.array(grp_dm_mass)
grp_dm_smls = np.array(grp_dm_smls)

subgrp_gstart = np.array(subgrp_gstart)
subgrp_glength = np.array(subgrp_glength)
grp_g_pos = np.array(grp_g_pos)
grp_g_mass = np.array(grp_g_mass)
grp_g_smls = np.array(grp_g_smls)

print("Min and max masses:", grp_s_mass.min(), grp_s_mass.max(),
      grp_ini_masses.min(), grp_ini_masses.max())

# Calculate the geometric centre of the group
centre = np.mean(grp_s_pos, axis=0)

print("Got the group data with %d particles" % len(grp_los))

# Compute luminosities for this group
grp_los *= 1 / pc
grp_los.to(1 / Mpc)
lums = total_lum(grp_ini_masses, grp_s_mets, grp_ages, grp_los,
                 kappa=0.0795, BC_fac=1)

# Get the group mass image
grp_smass_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_s_pos * Mpc,
    pixel_values=grp_s_mass,
    smoothing_lengths=grp_s_smls * Mpc,
    rest_frame=True,
    centre=centre
)
grp_smass_img = grp_smass_obj.get_smoothed_img(quintic)

print("Got Stellar Mass Image", np.min(grp_smass_img[grp_smass_img > 0]),
      np.max(grp_smass_img))

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_smass_img, norm=mpl.colors.Normalize(
    vmin=0,
    vmax=np.percentile(grp_smass_img, 99.99)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/stellarmass.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# Get the group mass image
grp_dmmass_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_dm_pos * Mpc,
    pixel_values=grp_dm_mass,
    smoothing_lengths=grp_dm_smls * Mpc,
    rest_frame=True,
    centre=centre
)
grp_dmmass_img = grp_dmmass_obj.get_smoothed_img(quintic)

print("Got Dark Matter Mass Image", np.min(grp_dmmass_img[grp_dmmass_img > 0]),
      np.max(grp_dmmass_img))

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_dmmass_img, norm=mpl.colors.Normalize(
    vmin=0,
    vmax=np.percentile(grp_dmmass_img, 99.9)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/darkmattermass.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# Get the group mass image
grp_gmass_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_g_pos * Mpc,
    pixel_values=grp_g_mass,
    smoothing_lengths=grp_g_smls * Mpc,
    rest_frame=True,
    centre=centre
)
grp_gmass_img = grp_gmass_obj.get_smoothed_img(quintic)

print("Got Gas Mass Image", np.min(grp_gmass_img[grp_gmass_img > 0]),
      np.max(grp_gmass_img))

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_gmass_img, norm=mpl.colors.Normalize(
    vmin=0,
    vmax=np.percentile(grp_gmass_img, 99.9)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/gasmass.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# Concert luminosity to flux
lums = lum_to_flux(lums, cosmo, z)

print("Got luminosities...")

# Get the PSF
nc = webbpsf.NIRCam()
nc.filter = 'F150W'
psf = nc.calc_psf(oversample=4)[0].data

print("Got the PSFs")

# Get the group luminosity image
grp_lum_obj = ParticleImage(
    resolution,
    fov=width,
    cosmo=cosmo,
    positions=grp_s_pos * Mpc,
    pixel_values=lums * nJy,
    smoothing_lengths=grp_s_smls * Mpc,
    psfs=psf,
    centre=centre,
    super_resolution_factor=2
)
grp_lum_img = grp_lum_obj.get_smoothed_img(quintic)

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_lum_img, norm=mpl.colors.Normalize(
    vmin=0,
    vmax=np.percentile(grp_lum_img, 99.99)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/stellarlum.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

grp_lum_img = grp_lum_obj.get_psfed_imgs()

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_lum_img, norm=mpl.colors.Normalize(
    vmin=0,
    vmax=np.percentile(grp_lum_img, 99.99)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/stellarlum_psf.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

grp_lum_img, grp_wht, grp_noise = grp_lum_obj.get_noisy_imgs(noise)

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_lum_img, norm=mpl.colors.Normalize(
    vmin=np.percentile(grp_lum_img, 36),
    vmax=np.percentile(grp_lum_img, 99.99)),
           cmap="Greys_r"
           )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/stellarlum_psfnoise.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(grp_noise,
          cmap="Greys_r"
          )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/noise.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()


print("Got Luminosity Image", np.min(grp_lum_img[grp_lum_img > 0]),
      np.max(grp_lum_img))

# Set up colormap and normalisation
norm = mpl.colors.Normalize(vmin=0, vmax=len(subgrp_sstart))
cmap = plt.get_cmap("plasma")

# Loop over subgroups and create subfind labelled image
subfind_img = np.zeros((grp_lum_img.shape[0], grp_lum_img.shape[1]))
subfind_id = 1
for start, length in zip(subgrp_sstart, subgrp_slength):

    print("Making an image for subgroup %d" % subfind_id)

    # Get the subgroup mass image
    subgrp_mass_obj = ParticleImage(
        resolution,
        fov=width,
        cosmo=cosmo,
        positions=grp_s_pos[start: start + length, :] * Mpc,
        pixel_values=grp_s_mass[start: start + length] * Msun,
        smoothing_lengths=grp_s_smls[start: start + length] * Mpc,
        centre=centre
    )
    subgrp_mass_img = subgrp_mass_obj.get_smoothed_img(quintic)

    # Get the mask for nonzero pixels
    mask = subgrp_mass_img > 0

    # Create an image to hold this subgroup
    subgrp_img = np.zeros(subfind_img.shape)
    subgrp_img[mask] = subfind_id

    # Add it to the main image
    subfind_img[mask] = (
        (subfind_img[mask] * (1 - alpha)) + (subgrp_img[mask] * alpha)
    )
    
    subfind_id += 1

print("Got SUBFIND image")

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(subfind_img,
          cmap="plasma"
          )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/subfind_stars.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# Loop over subgroups and create subfind labelled image
subfind_img = np.zeros((grp_lum_img.shape[0], grp_lum_img.shape[1]))
subfind_id = 1
for sstart, slength, dmstart, dmlength, gstart, glength in zip(subgrp_sstart, subgrp_slength,
                                                               subgrp_dmstart, subgrp_dmlength,
                                                               subgrp_gstart, subgrp_glength):

    print("Making an image for subgroup %d" % subfind_id)

    pos = np.concatenate((
        grp_s_pos[sstart: sstart + slength, :],
        grp_dm_pos[dmstart: dmstart + dmlength, :],
        grp_g_pos[gstart: gstart + glength, :],
    )) * Mpc
    mass = np.ones(slength + dmlength + glength)
    smls = np.concatenate((
        grp_s_smls[sstart: sstart + slength],
        grp_dm_smls[dmstart: dmstart + dmlength],
        grp_g_smls[gstart: gstart + glength],
    )) * Mpc

    # Get the subgroup mass image
    subgrp_mass_obj = ParticleImage(
        resolution,
        fov=width,
        cosmo=cosmo,
        positions=pos,
        pixel_values=mass,
        smoothing_lengths=smls,
        centre=centre
    )
    subgrp_mass_img = subgrp_mass_obj.get_smoothed_img(quintic)

    # Get the mask for nonzero pixels
    mask = subgrp_mass_img > 0

    # Create an image to hold this subgroup
    subgrp_img = np.zeros(subfind_img.shape)
    subgrp_img[mask] = subfind_id

    # Add it to the main image
    subfind_img[mask] = (
        (subfind_img[mask] * (1 - alpha)) + (subgrp_img[mask] * alpha)
    )
    
    subfind_id += 1

print("Got SUBFIND image")

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(subfind_img,
          cmap="plasma"
          )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/subfind_all.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# Create the signal image
    
# Create segmentation map
sig_image = grp_lum_img / noise
print(sig_image[sig_image > 0].min(), sig_image.max())
segm = phut.detect_sources(sig_image, 2.5, npixels=5)
# segm = phut.deblend_sources(det_img, segm,
#                             npixels=5, nlevels=32,
#                             contrast=0.001)
print(np.unique(segm))

fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.imshow(segm.data,
          cmap="plasma"
          )
ax.axis('off')
fig.savefig("plots/%s_%s_%d/segm.png" % (snap, reg, group_id),
            bbox_inches="tight", dpi=100, pad_inches=0)
plt.close()

# # Remove pixels below the background from the subfind ID image
# subfind_img[segm.data == segm.data.min()] = 0

# # Create plot
# fig = plt.figure(figsize=(7, 7))
# gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, wspace=0, hspace=0)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 0])
# ax4 = fig.add_subplot(gs[1, 1])

# # Turn off axes
# ax1.axis('off')
# ax2.axis('off')
# ax3.axis('off')
# ax4.axis('off')

# # plot images
# ax1.imshow(grp_mass_img, norm=mpl.colors.Normalize(
#     vmin=0,
#     vmax=np.percentile(grp_lum_img, 99.9)),
#         cmap="Greys_r"
#         )
# ax2.imshow(grp_lum_img, norm=mpl.colors.Normalize(
#     vmin=np.percentile(grp_lum_img, 36),
#     vmax=np.percentile(grp_lum_img, 99.9)),
#            cmap="Greys_r"
#            )
# ax3.imshow(subfind_img, cmap="plasma")
# ax4.imshow(segm.data, cmap="plasma")

# fig.savefig("plots/%s_%s_%d/source_ident_comp.png" % (snap, reg, group_id),
#             bbox_inches="tight", dpi=100)
# plt.close(fig)

