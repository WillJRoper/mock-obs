import sys
import os
import h5py
import numpy as np
from astropy.cosmology import Planck18 as cosmo

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from synthesizer.imaging.images import ParticleImage
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.stars import Stars
from synthesizer.kernel_functions import quintic
from synthesizer.filters import FilterCollection as Filters
from synthesizer.sed import m_to_fnu
from synthesizer.grid import Grid

import webbpsf
from utilities import total_lum, lum_to_flux
import photutils as phut
from unyt import kpc, erg, s, Hz, Msun, Mpc, nJy, pc


# Define the path to the data
datapath = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
    + "flares.hdf5"

# Set up the grid
grid = Grid("bc03_chabrier03_cloudy-v17.03_log10Uref-2", grid_dir="grids/")

# Define the list of subgroups to image
object_ids = ["007_z008p000_17_2_0",
              "007_z008p000_11_3_0",
              "007_z008p000_04_3_0",
              "008_z007p000_11_6_0",
              "008_z007p000_04_8_0",
              "008_z007p000_13_3_0",
              "008_z007p000_03_2_0",
              "008_z007p000_04_14_0",
              "008_z007p000_09_1_0",
              "008_z007p000_17_2_0",
              "008_z007p000_10_2_0",
              "008_z007p000_12_14_0",
              "009_z006p000_00_29_0",
              "009_z006p000_01_12_0",
              "009_z006p000_01_18_0",
              "009_z006p000_02_42_0",
              "009_z006p000_12_1_6",
              "009_z006p000_21_9_0",
              "010_z005p000_00_4_1",
              "010_z005p000_00_23_0",
              "010_z005p000_18_24_0",
              "010_z005p000_01_13_0"]

# Define filter list
filter_codes = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F250W",
]

# Set up filter object
filters = Filters(filter_codes, new_lam=grid.lam)
depths = {f: m_to_fnu(29.0) for f in filters.filter_codes}

# Get the PSF
psfs = {}
for f in filters.filter_codes:
    nc = webbpsf.NIRCam()
    nc.filter = f.split(".")[-1]
    psfs[f] = nc.calc_psf(oversample=4)[0].data

print("Got the PSFs")

for obj_id in object_ids:

    n, snap, reg, group_id, subgroup_id = obj_id
    snap = "_".join([n, snap])
    group_id = int(group_id)
    subgroup_id = int(subgroup_id)

    print(snap, reg, group_id, subgroup_id)

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

    hdf.close()
    
    print("Got data...")


    # What alpha will we use?
    alpha = float(sys.argv[1])

    # Get some image properties
    downsample = float(sys.argv[2])
    width = float(sys.argv[3])

    # Get redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    soft = 0.001802390 / (0.6777 * (1 + z)) 

    # Define image properties
    resolution = downsample * 0.031 / cosmo.arcsec_per_kpc_proper(z).value * kpc
    width = width * kpc
    print("Making images with %.2f kpc resolution and a %.2f FOV" % (resolution, width))

    # Make plot directory
    if not os.path.exists("plots/subgroup_%s_%s_%d_%d" % (snap, reg, group_id, subgroup_id)):
       os.makedirs("plots/subgroup_%s_%s_%d_%d" % (snap, reg, group_id, subgroup_id))

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

    # Create stars object
    stars = Stars(grp_ini_masses, grp_ages, grp_s_mets,
                  redshift=z, coordinates=grp_s_pos,
                  coord_units=Mpc, smoothing_lengths=grp_s_smls * Mpc)

    # Create galaxy object
    galaxy = Galaxy(stars=stars)

    print("Min and max masses:", grp_s_mass.min(), grp_s_mass.max(),
          grp_ini_masses.min(), grp_ini_masses.max())

    # Calculate the geometric centre of the group
    centre = np.mean(grp_s_pos, axis=0)

    print("Got the group data with %d particles" % len(grp_los))

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
        vmax=np.percentile(grp_smass_img, 99.9)),
               cmap="Greys_r"
               )
    ax.axis('off')
    fig.savefig("plots/%s_%s_%d_%d/stellarmass.png" % (snap, reg, group_id, subgroup_id),
                bbox_inches="tight", dpi=100, pad_inches=0)
    plt.close()

    # Calculate the stars SEDs
    sed = galaxy.generate_particle_spectra(grid, sed_object=True,
                                           spectra_type="total")
    # Calculate the stars SEDs
    int_sed = galaxy.generate_spectra(grid, sed_object=True,
                                      spectra_type="total")

    # Make the images
    grp_lum_obj = galaxy.make_image(resolution, fov=width, img_type="smoothed",
                                    filters=filters, psfs=psfs, depths=depths,
                                    snrs=5, aperture=1, kernel_func=quintic,
                                    rest_frame=False, cosmo=cosmo,
                                    super_resolution_factor=2)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(int_sed.lam, int_sed._lnu)
    fig.savefig("plots/%s_%s_%d_%d/stellar_spectra.png" % (snap, reg, group_id, subgroup_id),
                bbox_inches="tight", dpi=100, pad_inches=0)
    plt.close()

    for f in filters.filter_codes:
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs[f], norm=mpl.colors.Normalize(
            vmin=0,
            vmax=np.percentile(grp_lum_obj.imgs[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig(
            "plots/%s_%s_%d_%d/stellarflux_%s.png" % (snap, reg, group_id, subgroup_id, f.repalce("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs_psf[f], norm=mpl.colors.Normalize(
            vmin=0,
            vmax=np.percentile(grp_lum_obj.imgs_psf[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig("plots/%s_%s_%d_%d/stellarflux_psf_%s.png" % (snap, reg, group_id, subgroup_id, f.repalce("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs_noise[f], norm=mpl.colors.Normalize(
            vmin=0,
            vmax=np.percentile(grp_lum_obj.imgs_noise[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig("plots/%s_%s_%d_%d/stellarflux_psfnoise_%s.png" % (snap, reg, group_id, subgroup_id, f.repalce("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()




    
