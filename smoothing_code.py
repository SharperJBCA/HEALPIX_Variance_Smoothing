import numpy as np 
import healpy as hp 
from tqdm import tqdm
from integration.legendre_transform import get_polynomials, Legendre
from matplotlib import pyplot 

def produce_dec_mask(nside, map_coord, cut_min, cut_max):
    """                                                                                                                                                          
    Return a declination mask, masking region between cut_min and cut_max                                                                                          
    Pixels are set to 0 within the mask and 1 outside.                                                                                                           

    Parameters                                                                                                                                                   
    ----------                                                                                                                                                   
    nside : int                                                                                                                                                  
            nside of mask, e.g. nside=512                                                                                                                        
    rot_to : str, {'C','G','E'}                                                                                                                                   
            Coordinate system to rotate mask to.                                                                                                                 
    cut_min : float                                                                                                                                               
             Lower declination bound, e.g. cut_min=-90.                                                                                                           
    cut_max : float                                                                                                                                               
             Upper declination bound, e.g. cut_max=0.                                                                                                             


    Returns                                                                                                                                                      
    -------                                                                                                                                                      
    fullR : array_like                                                                                                                                           
            HEALPix map.                                                                                                                                         
    """
    NPIX = hp.nside2npix(nside)
    IPIX = np.arange(NPIX).astype(np.int64)
    theta, phi = hp.pix2ang(nside, IPIX)

    if map_coord == 'G':
        rot = hp.Rotator(coord=['G', 'C'])
        theta, phi = rot(theta, phi)

    declination = 90 - theta * 180 / np.pi

    mask = (declination <= cut_max) & (declination >= cut_min)

    return mask.astype(float)

def square_beam(Bl, nside, lmax, theta_bins=1000):
    """
    Square the beam transfer function in sky frame and return it. 

    Parameters
    ----------
    Bl : array-like
        The beam transfer function.
    lmax : int
    
    """
    PIXAREA = hp.nside2pixarea(nside)

    theta = np.linspace(0, np.pi, theta_bins)
    btheta = hp.bl2beam(Bl, theta)**2

    Bl2 = hp.beam2bl(btheta, theta, lmax=lmax) 
    Bl2 /= Bl2[0]
    return Bl2 #* PIXAREA 

def smooth_map(I, Q, U, fwhm_out=1.0, Bl=None, lmax=None, nside_out=None):
    """
    Smooth the input maps to the output resolution.

    Parameters
    ----------
    I : array-like
        The input I map.
    Q : array-like
        The input Q map.
    U : array-like
        The input U map.
    fwhm_out : float
        The output resolution in radians.
    Bl : array-like
        The input beam transfer function. 
        Polarised B_l has l=0, and l=1 terms set to zero.
        C_l = B_l^2 C_l^true. 
        If None, it is assumed fwhm_in is given.
    """

    # Constants
    NPIX = len(I)
    NSIDE = hp.npix2nside(NPIX)
    PIXAREA = hp.nside2pixarea(NSIDE)

    # SETUP INPUTS 
    if isinstance(nside_out, type(None)):
        nside_out = NSIDE
        npix_out = 12*nside_out**2

    if isinstance(lmax, type(None)):
        lmax = 3 * NSIDE - 1

    if isinstance(Bl, type(None)):
        Bl = hp.gauss_beam(fwhm_out, lmax=lmax)

    pixel_window = hp.pixwin(nside_out)[:lmax + 1]
    Bl[:lmax + 1] *= pixel_window

    # Richard's code 
    theta = np.linspace(0, np.pi, 1000)
    beam = hp.bl2beam(Bl/Bl[0], theta)
    p0l, d2p0l = get_polynomials(theta, lmax)
    transformer = Legendre(theta = theta,
                    beam = beam,
                    lmax = lmax,
                    p0l = p0l,
                    d2p0l = d2p0l)
    bl_raw_spin0 = transformer.spin0_transform()
    bl_raw_spin2 = transformer.spin2_transform()
    nl = transformer.get_normalisation()
    bl_raw_spin0 /= nl
    bl_raw_spin2 /= nl

    # Deconvolve Maps
    alm_I, alm_E, alm_B = hp.map2alm([I,Q,U], lmax=lmax)
    dalm_I = hp.almxfl(alm_I, bl_raw_spin0[:lmax + 1])
    dalm_E = hp.almxfl(alm_E, bl_raw_spin2[:lmax + 1])
    dalm_B = hp.almxfl(alm_B, bl_raw_spin2[:lmax + 1])

    dI, dQ, dU = hp.alm2map((dalm_I, dalm_E, dalm_B), nside=nside_out)
    dI[I[0] == hp.UNSEEN] = hp.UNSEEN
    dQ[Q[1] == hp.UNSEEN] = hp.UNSEEN
    dU[U[2] == hp.UNSEEN] = hp.UNSEEN

    return dI, dQ, dU

def smooth_variance_map_numerical(II, QQ, UU, IQ=None, IU=None, QU=None, fwhm_out=1.0, fwhm_in=None, Bl=None, lmax=None, niter=1000):
    """
    Smooth input variance maps numerically.

    """

    # Constants
    NPIX = len(II)
    NSIDE = hp.npix2nside(NPIX)
    PIXAREA = hp.nside2pixarea(NSIDE)

    # SETUP INPUTS 
    if isinstance(lmax, type(None)):
        lmax = 3 * NSIDE - 1

    if isinstance(Bl, type(None)):
        Bl = hp.gauss_beam(fwhm_out, lmax=lmax)

    dII, dQQ, dUU, dIQ, dIU, dQU = np.zeros(NPIX), np.zeros(NPIX), np.zeros(NPIX), np.zeros(NPIX), np.zeros(NPIX), np.zeros(NPIX)
    for i in tqdm(range(niter), desc='Smoothing Variance Maps Numerically'): 
        I = np.random.normal(II, scale=np.sqrt(II))
        Q = np.zeros(NPIX)
        U = np.zeros(NPIX)
        for ipix in range(NPIX):
            Q[ipix],U[ipix] = np.random.multivariate_normal([0,0], [[QQ[ipix], QU[ipix]], [QU[ipix], UU[ipix]]]).T

        dI, dQ, dU = hp.smoothing([I,Q,U], fwhm_out, lmax=lmax)
        dI -= np.mean(dI)
        dQ -= np.mean(dQ)
        dU -= np.mean(dU)
        dII += dI**2/niter
        dQQ += dQ**2/niter
        dUU += dU**2/niter
        dQU += dQ*dU/niter
        dIQ += dI*dQ/niter
        dIU += dI*dU/niter
        
    return dII, dQQ, dUU, dIQ, dIU, dQU

def smooth_variance_map(II, QQ, UU, IQ=None, IU=None, QU=None, fwhm_out=1.0,  Bl=None, lmax=None, nside_out=None):
    """
    Smooth input variance maps.

    Parameters
    ----------
    II : array-like
        The input II map.
    QQ : array-like
        The input QQ map.
    UU : array-like
        The input UU map.
    IQ : array-like (optional)
        The input IQ map.
    IU : array-like (optional)
        The input IU map.
    QU : array-like (optional)
        The input QU map.
    fwhm_out : float
        The output resolution in radians.
    Bl : array-like
        The input beam transfer function. 
        Polarised B_l has l=0, and l=1 terms set to zero.
        C_l = B_l^2 C_l^true. 
        If None, it is assumed fwhm_in is given.
    """

    # Constants
    NPIX = len(II)
    NSIDE = hp.npix2nside(NPIX)
    PIXAREA = hp.nside2pixarea(NSIDE)

    if isinstance(nside_out, type(None)):
        nside_out = NSIDE
        npix_out = 12*nside_out**2

    # SETUP INPUTS 
    if isinstance(lmax, type(None)):
        lmax = 3 * nside_out - 1

    if isinstance(Bl, type(None)):
        Bl = hp.gauss_beam(fwhm_out, lmax=lmax)

    pixel_window = hp.pixwin(nside_out)[:lmax + 1]
    Bl[:lmax + 1] *= pixel_window

    # Richard's code 
    theta = np.linspace(0, np.pi, 1000)
    beam = hp.bl2beam(Bl/Bl[0], theta)
    p0l, d2p0l = get_polynomials(theta, lmax)
    transformer = Legendre(theta = theta,
                    beam = beam**2,
                    lmax = lmax,
                    p0l = p0l,
                    d2p0l = d2p0l)
    bl_raw_spin0 = transformer.spin0_transform()
    bl_raw_spin2 = transformer.spin2_transform()
    nl = transformer.get_normalisation()
    bl_raw_spin0 /= nl
    bl_raw_spin2 /= nl


    alm_I = hp.map2alm(II, lmax=lmax)
    alm_E = hp.map2alm(QQ, lmax=lmax)
    alm_B = hp.map2alm(UU, lmax=lmax)
    dalm_I = hp.almxfl(alm_I, bl_raw_spin0[:lmax + 1])
    dalm_E = hp.almxfl(alm_E, bl_raw_spin0[:lmax + 1])
    dalm_B = hp.almxfl(alm_B, bl_raw_spin0[:lmax + 1])
    dII = hp.alm2map(dalm_I, nside=nside_out)
    dQQ = hp.alm2map(dalm_E, nside=nside_out)
    dUU = hp.alm2map(dalm_B, nside=nside_out)

    if not isinstance(IQ, type(None)):
        null_map = np.zeros_like(II)
        alm_QU_E, alm_QU_B = hp.map2alm_spin([QU,null_map], spin=2,lmax=lmax)
        dalm_QU_E = hp.almxfl(alm_QU_E, bl_raw_spin2[:lmax + 1])
        dalm_QU_B = hp.almxfl(alm_QU_B, bl_raw_spin2[:lmax + 1])
        dQU,_ = hp.alm2map_spin((dalm_QU_E,dalm_QU_B), spin=2,lmax=lmax, nside=nside_out)
    else:
        dQU = np.zeros(npix_out) 

    dIU = np.zeros(npix_out)
    dIQ = np.zeros(npix_out)

    dII[II == hp.UNSEEN] = hp.UNSEEN
    dQQ[QQ == hp.UNSEEN] = hp.UNSEEN
    dUU[UU == hp.UNSEEN] = hp.UNSEEN
    dIQ[IQ == hp.UNSEEN] = hp.UNSEEN
    dIU[IU == hp.UNSEEN] = hp.UNSEEN
    dQU[QU == hp.UNSEEN] = hp.UNSEEN

    dII[dII != hp.UNSEEN] *= PIXAREA * (2*np.pi)
    dQQ[dQQ != hp.UNSEEN] *= PIXAREA * (2*np.pi)
    dUU[dUU != hp.UNSEEN] *= PIXAREA * (2*np.pi)
    dIQ[dIQ != hp.UNSEEN] *= PIXAREA * (2*np.pi)
    dIU[dIU != hp.UNSEEN] *= PIXAREA * (2*np.pi)
    dQU[dQU != hp.UNSEEN] *= PIXAREA * (2*np.pi)

    return dII, dQQ, dUU, dIQ, dIU, dQU 