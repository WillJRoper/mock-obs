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
from synthesizer.utils import m_to_fnu
from synthesizer.grid import Grid

import webbpsf
from utilities import total_lum, lum_to_flux
import photutils as phut
from unyt import kpc, erg, s, Hz, Msun, Mpc, nJy, pc


# Define the path to the data
datapath = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
    + "flares.hdf5"

# Set up the grid
grid = Grid(
    "bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy-v17.03-U_modelfixed",
    grid_dir="grids/"
)

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
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W"
]

# Set up filter object
rest_filters = Filters(filter_codes, new_lam=grid.lam)
depths = {f: m_to_fnu(float(sys.argv[3])) for f in rest_filters.filter_codes}

# Get the PSF
psfs = {}
for f in rest_filters.filter_codes:
    nc = webbpsf.NIRCam()
    nc.filter = f.split(".")[-1]
    psfs[f] = nc.calc_psf(oversample=4)[0].data

print("Got the PSFs")

for obj_id in object_ids:

    n, snap, reg, group_id, subgroup_id = obj_id. split("_")
    snap = "_".join([n, snap])
    group_id = int(group_id)
    subgroup_id = int(subgroup_id)

    print(snap, reg, group_id, subgroup_id)

    # Get redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Extract data from data file
    hdf = h5py.File(datapath, "r")
    reg_snap_grp = hdf[reg][snap]
    grps = reg_snap_grp["Galaxy"]["GroupNumber"][...]
    subgrps = reg_snap_grp["Galaxy"]["SubGroupNumber"][...]
    cops = reg_snap_grp["Galaxy"]["COP"][...].T / (1 + z)
    s_length = reg_snap_grp["Galaxy"]["S_Length"][...]
    s_begin = np.zeros(len(s_length), dtype=int)
    s_begin[1:] = np.cumsum(s_length[:-1])
    s_pos = reg_snap_grp["Particle"]["S_Coordinates"][...].T / (1 + z)
    s_mass = reg_snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
    ini_masses = reg_snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
    s_mets = reg_snap_grp["Particle"]["S_Z"][...]
    ages = reg_snap_grp["Particle"]["S_Age"][...] * 10 ** 3
    print(ages)
    los = reg_snap_grp["Particle"]["S_los"][...]
    s_smls = reg_snap_grp["Particle"]["S_sml"][...]
    
    # g_length = reg_snap_grp["Galaxy"]["G_Length"][...]
    # g_begin = np.zeros(len(g_length), dtype=int)
    # g_begin[1:] = np.cumsum(g_length[:-1])
    # g_pos = reg_snap_grp["Particle"]["G_Coordinates"][...].T / (1 + z)
    # g_mass = reg_snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
    # g_smls = reg_snap_grp["Particle"]["G_sml"][...]

    hdf.close()
    
    print("Got data...")

    # Get some image properties
    downsample = float(sys.argv[1])
    width = float(sys.argv[2])

    soft = 0.001802390 / (0.6777 * (1 + z)) 

    # Define image properties
    resolution = downsample * 0.031 / cosmo.arcsec_per_kpc_proper(z).value * kpc
    width = width * kpc
    print("Making images with %.2f kpc resolution and a %.2f FOV" % (resolution, width))

    # Extract this groups data
    okinds = np.logical_and(grps == group_id, subgrps == subgroup_id)
    if len(s_length[okinds]) == 0:
        print(obj_id, "Couldn't be found")
        continue
    slength = s_length[okinds][0]
    sstart = s_begin[okinds][0]
    centre = cops[okinds, :][0]

    grp_s_pos = s_pos[sstart: sstart + slength, :]
    grp_s_mass = s_mass[sstart: sstart + slength]
    grp_ini_masses = ini_masses[sstart: sstart + slength]
    grp_s_mets = s_mets[sstart: sstart + slength]
    grp_ages = ages[sstart: sstart + slength]
    grp_los = los[sstart: sstart + slength]
    grp_s_smls = s_smls[sstart: sstart + slength]

    # grp_g_pos.extend(g_pos[gstart: gstart + glength, :])
    # grp_g_mass.extend(g_mass[gstart: gstart + glength])
    # grp_g_smls.extend(g_smls[gstart: gstart + glength])

    # Create stars object
    stars = Stars(grp_ini_masses, grp_ages, grp_s_mets,
                  redshift=z, coordinates=grp_s_pos,
                  coord_units=Mpc, smoothing_lengths=grp_s_smls * Mpc)

    print(stars)

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

    # Make plot directory
    if not os.path.exists("plots/subgroup_%s_%s_%d_%d" % (snap, reg, group_id, subgroup_id)):
       os.makedirs("plots/subgroup_%s_%s_%d_%d" % (snap, reg, group_id, subgroup_id))

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.imshow(grp_smass_img, norm=mpl.colors.Normalize(
        vmin=grp_smass_img[grp_smass_img > 0].min() - 1,
        vmax=np.percentile(grp_smass_img, 99.9)),
               cmap="Greys_r"
               )
    ax.axis('off')
    fig.savefig("plots/subgroup_%s_%s_%d_%d/stellarmass.png" % (snap, reg, group_id, subgroup_id),
                bbox_inches="tight", dpi=100, pad_inches=0)
    plt.close()

    # Calculate the stars SEDs
    sed = galaxy.generate_particle_spectra(grid, sed_object=True,
                                           spectra_type="total")
    sed.get_fnu(cosmo, stars.redshift, igm=None)
    # Calculate the stars SEDs
    int_sed = galaxy.generate_spectra(grid, sed_object=True,
                                      spectra_type="total")
    int_sed.get_fnu(cosmo, stars.redshift, igm=None)

    filters = Filters(filter_codes, new_lam=sed.lamz)

    # Make the images
    grp_lum_obj = galaxy.make_image(resolution, fov=width, img_type="smoothed",
                                    sed=sed, filters=filters, psfs=psfs, depths=depths,
                                    snrs=5, aperture=0.5, kernel_func=quintic,
                                    rest_frame=False, cosmo=cosmo,
                                    super_resolution_factor=2)

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[6, 3])
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax.loglog()
    ax1.semilogx()
    ax.grid(True)
    ax2.grid(True)
    for f in rest_filters:
        ax.plot(int_sed.lam, int_sed._lnu * f.t)
        ax1.plot(f.lam, f.t, label=f.filter_code)
    ax.plot(int_sed.lam, int_sed._lnu)
    ax.set_ylim(10 ** 30., 10**36.)
    ax1.set_xlabel("$\lambda/ [\AA]$")
    ax.set_ylabel("$L / [\mathrm{erg} / \mathrm{s} / \mathrm{Hz}]$")
    ax1.set_ylabel("$T$")
    ax1.set_ylim(0., 0.6)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    fig.savefig("plots/subgroup_%s_%s_%d_%d/spectra_luminosity.png" % (snap, reg, group_id, subgroup_id),
                bbox_inches="tight", dpi=100, pad_inches=0)
    plt.close()

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[6, 3])
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax.loglog()
    ax1.semilogx()
    ax.grid(True)
    ax2.grid(True)
    for f in filters:
        ax.plot(int_sed.lamz, int_sed._fnu * f.t)
        ax1.plot(f.lam, f.t, label=f.filter_code)
    ax.plot(int_sed.lamz, int_sed._fnu)
    ax1.set_xlabel("$\lambda/ [\AA]$")
    ax.set_ylabel("$F / [\mathrm{nJy}]$")
    ax1.set_ylabel("$T$")
    ax.set_ylim(10**-8., 10**6.9)
    ax1.set_ylim(0., 0.6)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    fig.savefig("plots/subgroup_%s_%s_%d_%d/spectra_flux.png" % (snap, reg, group_id, subgroup_id),
                bbox_inches="tight", dpi=100, pad_inches=0)
    plt.close()

    for f in filters.filter_codes:

        print(f, np.std(grp_lum_obj.noise_arrs[f]), grp_lum_obj.imgs[f].max(),
              grp_lum_obj.imgs[f].shape,
              grp_lum_obj.imgs_psf[f].shape,
              grp_lum_obj.imgs_noise[f].shape)
        
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs[f], norm=mpl.colors.Normalize(
            vmin=grp_lum_obj.imgs[f][grp_lum_obj.imgs[f] > 0].min() - 1,
            vmax=np.percentile(grp_lum_obj.imgs[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig(
            "plots/subgroup_%s_%s_%d_%d/stellarflux_%s.png" % (snap, reg, group_id, subgroup_id, f.replace("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs_psf[f], norm=mpl.colors.Normalize(
            vmin=grp_lum_obj.imgs_psf[f][grp_lum_obj.imgs_psf[f] > 0].min() - 1,
            vmax=np.percentile(grp_lum_obj.imgs_psf[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig("plots/subgroup_%s_%s_%d_%d/stellarflux_psf_%s.png" % (snap, reg, group_id, subgroup_id, f.replace("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(grp_lum_obj.imgs_noise[f], norm=mpl.colors.Normalize(
            vmin=--np.percentile(grp_lum_obj.imgs_noise[f], 32),
            vmax=np.percentile(grp_lum_obj.imgs_noise[f], 99.9)),
                  cmap="Greys_r"
                  )
        ax.axis('off')
        fig.savefig("plots/subgroup_%s_%s_%d_%d/stellarflux_psfnoise_%s.png" % (snap, reg, group_id, subgroup_id, f.replace("/", ".")),
                    bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

    # Also, lets make an RGB images
    fig, ax, rgb_img = grp_lum_obj.plot_rgb_image(
        rgb_filters={"R": ["JWST/NIRCam.F444W",],
                     "G": ["JWST/NIRCam.F356W",],
                     "B": ["JWST/NIRCam.F200W",]},
        img_type="standard",
    )
        
    fig.savefig(
        "plots/subgroup_%s_%s_%d_%d/stellarflux_RGB.png" % (snap, reg, group_id, subgroup_id),
        bbox_inches="tight", dpi=300)
    plt.close()
    
    fig, ax, rgb_img = grp_lum_obj.plot_rgb_image(
        rgb_filters={"R": ["JWST/NIRCam.F444W",],
                     "G": ["JWST/NIRCam.F356W",],
                     "B": ["JWST/NIRCam.F200W",]},
        img_type="psf",
    )
    
    fig.savefig(
        "plots/subgroup_%s_%s_%d_%d/stellarflux_psf_RGB.png" % (snap, reg, group_id, subgroup_id),
        bbox_inches="tight", dpi=300)
    plt.close()
    
    fig, ax, rgb_img = grp_lum_obj.plot_rgb_image(
        rgb_filters={"R": ["JWST/NIRCam.F444W",],
                     "G": ["JWST/NIRCam.F356W",],
                     "B": ["JWST/NIRCam.F200W",]},
        img_type="noise",
    )
    
    fig.savefig(
        "plots/subgroup_%s_%s_%d_%d/stellarflux_psfnoise_RGB.png" % (snap, reg, group_id, subgroup_id),
        bbox_inches="tight", dpi=300)
    plt.close()


    
