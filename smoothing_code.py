import numpy as np 
import healpy as hp 
from tqdm import tqdm
import sys 
from integration.legendre_transform import get_polynomials, Legendre
from matplotlib import pyplot 
from scipy.interpolate import LinearNDInterpolator

def interpolate(values, theta_in, phi_in, theta_out, phi_out):
    # Convert spherical to cartesian coordinates
    x_in = np.sin(theta_in) * np.cos(phi_in)
    y_in = np.sin(theta_in) * np.sin(phi_in)
    z_in = np.cos(theta_in)
    
    x_out = np.sin(theta_out) * np.cos(phi_out)
    y_out = np.sin(theta_out) * np.sin(phi_out)
    z_out = np.cos(theta_out)
    
    # Create interpolator
    points_in = np.column_stack((x_in, y_in, z_in))
    points_out = np.column_stack((x_out, y_out, z_out))
    
    interpolator = LinearNDInterpolator(points_in, values)
    print(np.sum(interpolator(points_out)),np.sum(values))
    return interpolator(points_out)

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

    II_mask = hp.ud_grade(I, nside_out) == hp.UNSEEN

    dI[II_mask] = hp.UNSEEN
    dQ[II_mask] = hp.UNSEEN
    dU[II_mask] = hp.UNSEEN

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

def gen_pixel_window(nside, lmax=None):
    """Approximate pixel window function"""

    ell = np.arange(lmax + 1) 

    # Okay, so we are approximating the high-ell values of the pixel
    # window function by transforming a circular top beam into l-space.
    # This is pretty close. 
    theta = np.linspace(0,np.pi,10000) 
    # top hat beam 
    beam = np.zeros_like(theta) 
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    beam[theta<np.radians(3.6/np.pi*0.5*pixel_area**0.5)] = 1
    bl = hp.beam2bl(beam, theta, lmax=lmax) 

    return  bl/bl[0] 

def create_bl(beam, theta, lmax, normalise=False):
    """Wrapper for generating B_l with richard's code"""
    p0l, d2p0l = get_polynomials(theta, lmax)
    transformer = Legendre(theta = theta,
                    beam = beam,
                    lmax = lmax,
                    p0l = p0l,
                    d2p0l = d2p0l)
    bl_raw_spin0 = transformer.spin0_transform()
    bl_raw_spin2 = transformer.spin2_transform()
    nl = transformer.get_normalisation()
    if normalise:
        bl_raw_spin0 /= nl
        bl_raw_spin2 /= nl
    return bl_raw_spin0, bl_raw_spin2 

def fix_poles_rotation(dalms, nside_out, fwhm):
    """Rotate E and B to a new coordinate system, interpolate over poles"""

    rotate     = hp.Rotator(coord=['G','C'])
    rotate_inv = hp.Rotator(coord=['C','G'])

    dalms_rot = [rotate.rotate_alm(dalm) for dalm in dalms]
    maps_rot  = hp.alm2map(dalms_rot, nside=nside_out)
    maps      = hp.alm2map(dalms, nside=nside_out)

    mask = np.zeros(hp.nside2npix(nside_out))
    angles = np.linspace(0,np.radians(30),30)
    averages = np.zeros_like(angles)
    for i,deg in enumerate(angles[::-1]):
        q = hp.query_disc(nside_out, hp.ang2vec(0,0), deg)
        mask[q] += 1

    for i, deg in enumerate(angles):
        averages[i] = np.mean(maps[1][mask == i+1])
    
    pyplot.figure() 
    pyplot.plot(np.degrees(angles),averages[::-1])
    pyplot.savefig('test_images/averages.png')
    pyplot.close()

    mask = np.zeros(hp.nside2npix(nside_out))
    angles = np.linspace(0,np.radians(30),30)
    averages = np.zeros_like(angles)
    for i,deg in enumerate(angles[::-1]):
        q = hp.query_disc(nside_out, hp.ang2vec(0,0), deg)
        mask[q] += 1
    for i, deg in enumerate(angles):
        averages[i] = np.mean(maps_rot[1][mask == i+1])
    
    pyplot.figure() 
    pyplot.plot(np.degrees(angles),averages[::-1])
    pyplot.savefig('test_images/averages_rot.png')
    pyplot.close()

    pyplot.figure(figsize=(8,16))
    hp.mollview(mask) 
    hp.graticule()
    pyplot.savefig('test_images/mask.png')
    pyplot.close()


    pyplot.figure(figsize=(8,16))
    hp.mollview(maps_rot[0], title='smooth_I',sub=(3,1,1),coord=['C','G'],rot=[0,90])
    hp.mollview(maps_rot[1], title='smooth_Q',sub=(3,1,2),coord=['C','G'],rot=[0,90])
    hp.mollview(maps_rot[2], title='smooth_U',sub=(3,1,3),coord=['C','G'],rot=[0,90])
    hp.graticule()
    pyplot.savefig('test_images/smooth_map_rot.png')
    pyplot.close()

    # pole in original coordinate system is
    theta = 0
    phi   = 0 

    # select pixels near the pole in the new coordinate system
    pixels             = hp.query_disc(nside_out, hp.ang2vec(theta, phi), fwhm*3) 
    theta_pix, phi_pix = hp.pix2ang(nside_out, pixels) 

    # rotate theta_pix_rot and phi_pix_rot to original coordinate system 
    theta_pix_rot, phi_pix_rot = rotate_inv(theta_pix, phi_pix)
    pixels_rot = hp.ang2pix(nside_out, theta_pix_rot, phi_pix_rot)

    # mask unseen 
    mask = (maps_rot[0][pixels_rot] != hp.UNSEEN)
    # interpolate over the pole
    maps[1][pixels] = interpolate(maps_rot[1][pixels_rot[mask]], theta_pix[mask], phi_pix[mask], theta_pix, phi_pix)
    maps[2][pixels] = interpolate(maps_rot[2][pixels_rot[mask]], theta_pix[mask], phi_pix[mask], theta_pix, phi_pix)

    return maps

def fix_poles(maps, beams, nside_out, fwhm, lmax=None):
    """Smooth Q and U with a spin-0 field, interpolate over poles"""

    # Spin - 2 transform
    alms = hp.map2alm(maps, lmax=lmax)
    dalms = [hp.almxfl(alm, beam) for alm, beam in zip(alms,beams)]
    output_maps = hp.alm2map(dalms, nside=nside_out)

    # Spin-0 transform
    alm_Q = hp.map2alm(maps[1], lmax=lmax)
    alm_U = hp.map2alm(maps[2], lmax=lmax)
    dalm_Q = hp.almxfl(alm_Q, beams[0])
    dalm_U = hp.almxfl(alm_U, beams[0])
    Q_0 = hp.alm2map(dalm_Q, nside=nside_out)
    U_0 = hp.alm2map(dalm_U, nside=nside_out)

    # Now calculate theta, phi for all pixels 
    pixels = np.arange(hp.nside2npix(nside_out))
    theta, phi = hp.pix2ang(nside_out, pixels) 

    # We want a cosine window function in theta that starts at 2*fwhm and is fwhm/5 wide 
    wl = fwhm 
    window = (np.cos(2*np.pi*(theta - fwhm*2)/wl) + 1)*0.5
    window[theta < fwhm*2] = 1
    window[theta > fwhm*2 + wl/2] = 0

    # Now we want to fold together the two maps 
    output_maps[1] = output_maps[1]*(1-window) + Q_0*window
    output_maps[2] = output_maps[2]*(1-window) + U_0*window

    return output_maps



def smooth_variance_intensity_pspace(var_maps, beam): 

    if len(var_maps) == 1:
        II = var_maps[0]
    else:
        II = var_maps

    nside = hp.npix2nside(len(II)) 
    pixel_area = hp.nside2pixarea(nside)
    vec = hp.ang2vec(0,0) # pole 
    R = np.radians(10) # radius

    pixels = hp.query_disc(nside, vec, R) 
    theta, phi = hp.pix2ang(nside, pixels)
    rot = hp.Rotator(rot=[0,90]) 
    theta, phi = rot(theta, phi) 

    from scipy.linalg import circulant 

    Bij = circulant(np.interp(theta, np.linspace(0,np.pi,len(beam)), beam))

    II_ij = Bij @ II[pixels] * pixel_area**2 

    print(II_ij) 

def smooth_variance_intensity(var_maps, bl_spin0, lmax=None, nside_out=None):
    """
    Smooth intensity variance maps.

    Parameters
    ----------
    var_maps : list/np.ndarray 
        List containing II map or array containing II map.
    bl_spin0 : array-like
        Spin 0 transfer function for the beam.
    lmax : int
        Maximum l value.
    nside_out : int
        Output nside.
    """
    if len(var_maps) == 1:
        II = var_maps[0]
    else:
        II = var_maps

    nside = hp.npix2nside(len(II))
    pix_area = hp.nside2pixarea(nside)

    theta = np.linspace(0, np.pi, 100000)
    beam  = hp.bl2beam(bl_spin0, theta)
    bl_squared_spin0, _ = create_bl(beam**2, theta, lmax, normalise=False)

    alm_I= hp.map2alm(II, lmax=lmax)
    dalm_I = hp.almxfl(alm_I, bl_squared_spin0[:lmax + 1])
    dII = hp.alm2map(dalm_I, nside=nside_out)

    II_mask = hp.ud_grade(II, nside_out) == hp.UNSEEN
    dII[II_mask] = hp.UNSEEN
    dII[~II_mask] *= pix_area 

    return dII

def smooth_variance_polarisation(var_maps, bl_spin0, lmax=None, nside_out=None):
    """
    Smooth polarisation variance maps.

    Parameters
    ----------
    var_maps : list/np.ndarray 
        List containing II, QQ, UU or II, QQ, UU, IQ, IU, QU maps
    
    bl_spin0 : array-like
        Spin 0 transfer function for the beam.

    bl_spin2 : array-like
        Spin 2 transfer function for the beam.

    lmax : int  
        Maximum l value.
    
    nside_out : int
        Output nside.
    """
    if len(var_maps) == 3:
        II, QQ, UU = var_maps
        cross_terms = False 
    elif len(var_maps) == 6:
        II, QQ, UU, IQ, IU, QU = var_maps
        cross_terms = True
    else:
        raise ValueError("Input map list must be length 3 or 6") 
    
    nside = hp.npix2nside(len(II))
    pix_area = hp.nside2pixarea(nside)

    theta = np.linspace(0, np.pi, 100000)
    beam  = hp.bl2beam(bl_spin0, theta)
    r_width = np.sum(theta*beam**2)/np.sum(beam**2) * 5.0/1.2

    bl_squared_spin0, bl_squared_spin2 = create_bl(beam**2, theta, lmax, normalise=False)

    alms = hp.map2alm([II,QQ,UU], lmax=lmax)
    beams = [bl_squared_spin0[:lmax + 1], bl_squared_spin2[:lmax + 1], bl_squared_spin2[:lmax + 1]]
    dalms = [hp.almxfl(alm, beam) for alm, beam in zip(alms,beams)]
    output_maps = fix_poles([II,QQ,UU], beams, nside_out, r_width)
    #output_maps = hp.alm2map(dalms, nside=nside_out)

    if cross_terms:
        alms_QU = hp.map2alm_spin([QU,np.zeros_like(QU)], spin=2, lmax=lmax)
        dalms_QU = [hp.almxfl(alm, bl_squared_spin2[:lmax + 1]) for alm in alms_QU]
        QU,_ = hp.alm2map_spin(dalms_QU, spin=2, lmax=lmax, nside=nside_out)
        IQ = np.zeros_like(QU)
        IU = np.zeros_like(QU)
        output_maps = np.concatenate([output_maps, [IQ, IU, QU]])
    
    II_mask = hp.ud_grade(II, nside_out) == hp.UNSEEN
    for i in range(len(output_maps)):
        output_maps[i][II_mask] = hp.UNSEEN
        output_maps[i][~II_mask] *= pix_area


    return output_maps

def smooth_variance_map(var_maps, bl_spin0, lmax=None, nside_out=None):
                        #bl_spin2, II, QQ, UU, IQ=None, IU=None, QU=None, lmax=None, nside_out=None):
    """
    Smooth input variance maps.

    Parameters
    ----------
    var_maps : list 
        List of input variance maps. 
        If len > 6 or len == 1, assume II map.
        If len == 3, assume II, QQ, UU maps.
        If len == 6, assume II, QQ, UU, IQ, IU, QU maps. 
    bl_spin0 : array-like
        Spin 0 transfer function for the beam.

    NB: bl_spin2 is calculated internally as we need to square the beam function in real space. 
    """
    if len(var_maps) > 6:
        nside = hp.npix2nside(len(var_maps))
    else:
        nside = hp.npix2nside(len(var_maps[0]))

    if isinstance(lmax, type(None)):
        lmax = 3 * nside - 1

    if len(var_maps) == 1 or len(var_maps) > 6:
        output_maps = smooth_variance_intensity(var_maps, bl_spin0, lmax=lmax, nside_out=nside_out)
    elif len(var_maps) == 3 or len(var_maps) == 6:
        output_maps = smooth_variance_polarisation(var_maps, bl_spin0, lmax=lmax, nside_out=nside_out)

    return output_maps