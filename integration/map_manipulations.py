import healpy as hp
import numpy as np


def var_map_scaling(in_var_maps, ratio_I, ratio_P, dec_mask, num_noise_sims=3):
    
    out_var_map_scale_factor = np.array([0, 0, 0])
    
    for i in range(0, num_noise_sims):
        out_var_map, out_noise_map = smooth_var_map(np.array([in_var_maps[0], in_var_maps[1], in_var_maps[2]]),
                                                    np.array([ratio_I, ratio_P, ratio_P]),
                                                    dg_nside=hp.get_nside(in_var_maps[0]))
        
        for j in range(3):
            out_noise_map[j][dec_mask == 0] = hp.UNSEEN
            out_var_map[j][dec_mask == 0] = hp.UNSEEN
        
        # Scale factors
        goodIdxI = np.where(out_var_map[0] > 0.)[0]
        goodIdxQ = np.where(out_var_map[1] > 0.)[0]
        goodIdxU = np.where(out_var_map[2] > 0.)[0]
        out_var_map_scale_factor = out_var_map_scale_factor + \
                               np.array([np.var(out_noise_map[0][goodIdxI] / np.sqrt(out_var_map[0][goodIdxI])),
                                         np.var(out_noise_map[1][goodIdxQ] / np.sqrt(out_var_map[1][goodIdxQ])),
                                         np.var(out_noise_map[2][goodIdxU] / np.sqrt(out_var_map[2][goodIdxU]))])
    out_var_map_scale_factor = out_var_map_scale_factor / (num_noise_sims * 1.0)

    return out_var_map_scale_factor
    

def smooth_var_map(in_var_maps, ratio, dg_with_alms=False, dg_nside=1024):
    """Reweights the variance map in alm space with ratio and does the same to a realisation of the noise.


    Parameters
    ----------
    in_var_maps : array_like
            Variance maps either var(I) with shape (npix,) or [var(I),var(Q),var(U)] with shape (3,npix).
            e.g., in_var_maps = np.array([in_var_maps[0],in_var_maps[1],in_var_maps[2]])
    ratio : array_like
            alm reweighting functions, with shape (lmax+1,) if just I or (3,lmax+1) if I,Q and U.
            e.g., ratio = np.array([ratio_I,ratio_P,ratio_P])
    dg_with_alms : book, False
                  If True, downgrade during alm2map else don't downgrade
    dg_nside : int, 64
                 The nside to downgrade to

    Returns
    -------
    out_var_map : array_like
                smoothed var map with same shape as in_var_maps
    out_noise_map : array_like
                smoothed noise realisation map with same shape as in_var_maps
    """
    var_info = hp.maptype(in_var_maps)
    nside = hp.get_nside(in_var_maps)
    lmax = 4 * nside
    ###################################
    # Simulate noise maps
    ###################################
    sim_noise = np.random.standard_normal(np.shape(in_var_maps)) * np.sqrt(in_var_maps)
    sim_noise[in_var_maps == hp.UNSEEN] = hp.UNSEEN
    ###################################
    # Calculate alms
    ###################################
    in_var_maps_alms = hp.map2alm(in_var_maps, lmax=lmax)
    sim_noise_alms = hp.map2alm(sim_noise, lmax=lmax)
    ###################################
    # Reweight
    ###################################
    if var_info == 0:
        # Just I
        out_var_alms = hp.almxfl(in_var_maps_alms, ratio)
        out_noise_alms = hp.almxfl(sim_noise_alms, ratio)
        # Make map
        if dg_with_alms == True:
            mask = np.ones(hp.nside2npix(nside))
            mask[in_var_maps == hp.UNSEEN] = hp.UNSEEN
            out_noise_map = hp.alm2map(out_noise_alms, nside, pixwin=False)
            out_var_map = hp.alm2map(out_var_alms, nside, pixwin=False)
            out_var_map = np.abs(out_var_map)
            out_noise_map[mask == hp.UNSEEN] = hp.UNSEEN
            out_var_map[mask == hp.UNSEEN] = hp.UNSEEN
            out_noise_map = hp.ud_grade(out_noise_map, dg_nside)
            out_var_map = hp.ud_grade(out_var_map, dg_nside)
        else:
            out_noise_map = hp.alm2map(out_noise_alms, nside, pixwin=False)
            out_var_map = hp.alm2map(out_var_alms, nside, pixwin=False)
            out_var_map = np.abs(out_var_map)
            out_noise_map[in_var_maps == hp.UNSEEN] = hp.UNSEEN
            out_var_map[in_var_maps == hp.UNSEEN] = hp.UNSEEN
    elif var_info == 3:
        # I Q U
        out_var_alms = np.zeros(np.shape(in_var_maps_alms), dtype=np.complex_)
        out_noise_alms = np.zeros(np.shape(sim_noise_alms), dtype=np.complex_)
        out_var_alms[0] = hp.almxfl(in_var_maps_alms[0], ratio[0])
        out_var_alms[1] = hp.almxfl(in_var_maps_alms[1], ratio[1])
        out_var_alms[2] = hp.almxfl(in_var_maps_alms[2], ratio[2])
        out_noise_alms[0] = hp.almxfl(sim_noise_alms[0], ratio[0])
        out_noise_alms[1] = hp.almxfl(sim_noise_alms[1], ratio[1])
        out_noise_alms[2] = hp.almxfl(sim_noise_alms[2], ratio[2])
        # Make maps
        if dg_with_alms == True:
            mask = np.ones([3, hp.nside2npix(nside)])
            mask[0][in_var_maps[0] == hp.UNSEEN] = hp.UNSEEN
            mask[1][in_var_maps[1] == hp.UNSEEN] = hp.UNSEEN
            mask[2][in_var_maps[2] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map_I = hp.alm2map(out_noise_alms[0], nside, pixwin=False)
            out_noise_map_E = hp.alm2map(out_noise_alms[1], nside, pixwin=False)
            out_noise_map_B = hp.alm2map(out_noise_alms[2], nside, pixwin=False)
            out_var_map_I = hp.alm2map(out_var_alms[0], nside, pixwin=False)
            out_var_map_E = hp.alm2map(out_var_alms[1], nside, pixwin=False)
            out_var_map_B = hp.alm2map(out_var_alms[2], nside, pixwin=False)
            out_noise_map_I[mask[0] == hp.UNSEEN] = hp.UNSEEN
            out_var_map_I[mask[0] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map_E[mask[1] == hp.UNSEEN] = hp.UNSEEN
            out_var_map_E[mask[1] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map_B[mask[2] == hp.UNSEEN] = hp.UNSEEN
            out_var_map_B[mask[2] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map_I = hp.ud_grade(out_noise_map_I, dg_nside)
            out_noise_map_E = hp.ud_grade(out_noise_map_E, dg_nside)
            out_noise_map_B = hp.ud_grade(out_noise_map_B, dg_nside)
            out_var_map_I = hp.ud_grade(out_var_map_I, dg_nside)
            out_var_map_E = hp.ud_grade(out_var_map_E, dg_nside)
            out_var_map_B = hp.ud_grade(out_var_map_B, dg_nside)
            out_noise_alms_I = hp.map2alm(out_noise_map_I, lmax=4 * dg_nside)
            out_noise_alms_E = hp.map2alm(out_noise_map_E, lmax=4 * dg_nside)
            out_noise_alms_B = hp.map2alm(out_noise_map_B, lmax=4 * dg_nside)
            out_var_alms_I = hp.map2alm(out_var_map_I, lmax=4 * dg_nside)
            out_var_alms_E = hp.map2alm(out_var_map_E, lmax=4 * dg_nside)
            out_var_alms_B = hp.map2alm(out_var_map_B, lmax=4 * dg_nside)
            out_noise_map = hp.alm2map([out_noise_alms_I, out_noise_alms_E, out_noise_alms_B], dg_nside, pixwin=False)
            out_var_map = np.abs(hp.alm2map([out_var_alms_I, out_var_alms_E, out_var_alms_B], dg_nside, pixwin=False))
        else:
            out_noise_map = hp.alm2map([out_noise_alms[0], out_noise_alms[1], out_noise_alms[2]], nside, pixwin=False)
            out_var_map = hp.alm2map([out_var_alms[0], out_var_alms[1], out_var_alms[2]], nside, pixwin=False)
            out_var_map = np.abs(out_var_map)
            out_noise_map[0][in_var_maps[0] == hp.UNSEEN] = hp.UNSEEN
            out_var_map[0][in_var_maps[0] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map[1][in_var_maps[1] == hp.UNSEEN] = hp.UNSEEN
            out_var_map[1][in_var_maps[1] == hp.UNSEEN] = hp.UNSEEN
            out_noise_map[2][in_var_maps[2] == hp.UNSEEN] = hp.UNSEEN
            out_var_map[2][in_var_maps[2] == hp.UNSEEN] = hp.UNSEEN
    else:
        print('Variance map wrong size')
        out_var_map = None
        out_var_mapScaleFactor = None
        outNosieMap = None

    return out_var_map, out_noise_map


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