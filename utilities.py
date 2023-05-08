import numpy as np
from synthobs.sed import models
import flare
import flare.filters


def lum_to_flux(lum, cosmo, z):

    # Calculate the luminosity distance
    lum_dist = cosmo.luminosity_distance(z).to('cm').value
    
    
    return 1E9 * 1E23 * lum * (1.+ z) / (4. * np.pi * lum_dist ** 2)


def DTM_fit(Z, Age):
    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5 / (D0 * Z)
    DTM = D0 + (D1 - D0) * (1. - np.exp(-alpha * (Z ** beta)
                                        * ((Age / (1e3 * tau)) ** gamma)))
    if np.isnan(DTM) or np.isinf(DTM):
        DTM = 0.

    return DTM


def total_lum(ini_masses, s_mets, ages, los, kappa,
              BC_fac, IMF='Chabrier_300',
              filters=('JWST.NIRCAM.F150W',), log10t_BC=7.):

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -

    # Define extinction
    model.dust_ISM = (
        'simple', {'slope': -1.})  # Define dust curve for ISM
    model.dust_BC = ('simple', {
        'slope': -1.})  # Define dust curve for birth cloud component

    # --- create rest-frame luminosities
    F = flare.filters.add_filters(filters, new_lam=model.lam)
    model.create_Lnu_grid(
        F)  # --- create new L grid for each filter. In units of erg/s/Hz

    part_dtm = np.zeros(len(ages))
    for ind in range(len(ages)):
        part_dtm[ind] = DTM_fit(s_mets[ind], ages[ind])
    los = part_dtm * los

    # --- calculate V-band (550nm) optical depth for each star particle
    tauVs_ISM = kappa * los
    tauVs_BC = BC_fac * (s_mets / 0.01)
    fesc = 0.0
    for f in filters:
        print(F)
        print(f)
        Lnu = models.generate_Lnu_array(model, F, f, ini_masses, ages, s_mets,
                                        tauVs_ISM, tauVs_BC,
                                        fesc=fesc, log10t_BC=log10t_BC)

    return Lnu
