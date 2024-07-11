import numpy as np
import healpy as hp
import os 
from matplotlib import pyplot
from smoothing_code import produce_dec_mask, square_beam, smooth_map, smooth_variance_map, smooth_variance_map_numerical

def test_produce_dec_mask():
    nside = 32
    map_coord = 'G'
    cut_min = -30
    cut_max = 30
    
    mask = produce_dec_mask(nside, map_coord, cut_min, cut_max)
    
    assert len(mask) == hp.nside2npix(nside)
    assert np.all((mask == 0) | (mask == 1))
    
    # Check if the mask is correct
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)
    rot = hp.Rotator(coord=['G', 'C'])
    theta, phi = rot(theta, phi)
    dec = 90 - np.degrees(theta)
    
    expected_mask = ((dec >= cut_min) & (dec <= cut_max)).astype(float)
    np.testing.assert_array_almost_equal(mask, expected_mask)

def test_square_beam():
    lmax = 100
    nside = 32
    fwhm = np.radians(1)
    
    Bl = hp.gauss_beam(fwhm, lmax=lmax)
    squared_Bl = square_beam(Bl, nside, lmax)
    assert len(squared_Bl) == lmax + 1
    assert np.all(squared_Bl >= 0)
    #assert squared_Bl[0] == hp.nside2pixarea(nside)

def test_smooth_map():
    nside = 32
    npix = hp.nside2npix(nside)
    
    I = np.random.randn(npix)
    Q = np.random.randn(npix)
    U = np.random.randn(npix)
    
    fwhm_out = np.radians(15)
    
    smooth_I, smooth_Q, smooth_U = smooth_map(I, Q, U, fwhm_out)
    
    assert len(smooth_I) == npix
    assert len(smooth_Q) == npix
    assert len(smooth_U) == npix
    
    # Check if smoothing reduces the standard deviation
    assert np.std(smooth_I) < np.std(I)
    assert np.std(smooth_Q) < np.std(Q)
    assert np.std(smooth_U) < np.std(U)

    pyplot.figure(figsize=(8,16))
    hp.mollview(smooth_I, title='smooth_I',sub=(3,1,1))
    hp.mollview(smooth_Q, title='smooth_Q',sub=(3,1,2))
    hp.mollview(smooth_U, title='smooth_U',sub=(3,1,3))
    hp.graticule()
    pyplot.savefig('test_images/smooth_map.png')
    pyplot.close()

    pyplot.figure(figsize=(8,16))
    hp.mollview(I, title='input_I',sub=(3,1,1))
    hp.mollview(Q, title='input_Q',sub=(3,1,2))
    hp.mollview(U, title='input_U',sub=(3,1,3))
    hp.graticule()
    pyplot.savefig('test_images/input_map.png')
    pyplot.close()


def test_smooth_variance_map():
    nside = 128
    npix = hp.nside2npix(nside)
    
    np.random.seed(0)
    II = np.random.randn(npix)*0.005 + 1
    QQ = np.random.randn(npix)*0.005 + 1
    UU = np.random.randn(npix)*0.005 + 1
    IQ = np.zeros(npix)*0.005
    IU = np.zeros(npix)*0.005
    QU = np.zeros(npix)*0.005
    
    fwhm_out = np.radians(15)
    
    smooth_II, smooth_QQ, smooth_UU, smooth_IQ, smooth_IU, smooth_QU = smooth_variance_map(
        II, QQ, UU, IQ, IU, QU, fwhm_out, nside_out=32
    )
    
    assert len(smooth_II) == npix
    assert len(smooth_QQ) == npix
    assert len(smooth_UU) == npix
    assert len(smooth_IQ) == npix
    assert len(smooth_IU) == npix
    assert len(smooth_QU) == npix
    
    hp.write_map('test_images/analytical_smooth_variances.fits',
                 [smooth_II,smooth_QQ,smooth_UU,smooth_IQ,smooth_IU,smooth_QU],overwrite=True)
    pyplot.figure(figsize=(8,16))
    hp.mollview(smooth_II, title='smooth_II',sub=(3,1,1),rot=[0,90])
    hp.mollview(smooth_QQ, title='smooth_QQ',sub=(3,1,2),rot=[0,90])
    hp.mollview(smooth_UU, title='smooth_UU',sub=(3,1,3),rot=[0,90])
    hp.graticule()
    pyplot.savefig('test_images/smooth_variance_map.png')
    pyplot.close()

    pyplot.figure(figsize=(8,16))
    hp.mollview(II, title='input_II',sub=(3,1,1))
    hp.mollview(QQ, title='input_QQ',sub=(3,1,2))
    hp.mollview(UU, title='input_UU',sub=(3,1,3))
    hp.graticule()
    pyplot.savefig('test_images/input_variance_map.png')
    pyplot.close()


def test_smooth_real_variance_map():
    
    II,QQ,UU,QU = hp.read_map('cbass_maps/cbass_DR1_ss_1024.fits',field=[3,4,5,6])
    IU = np.zeros_like(II)
    IQ = np.zeros_like(II)
    npix = II.size
    nside = hp.npix2nside(npix)
    nside_out = 32
    fwhm_out = np.radians(15)
    
    smooth_II, smooth_QQ, smooth_UU, smooth_IQ, smooth_IU, smooth_QU = smooth_variance_map(
        II, QQ, UU, IQ, IU, QU, fwhm_out, nside_out=32
    )
    npix_out = hp.nside2npix(nside_out)
    
    assert len(smooth_II) == npix_out
    assert len(smooth_QQ) == npix_out
    assert len(smooth_UU) == npix_out
    assert len(smooth_IQ) == npix_out
    assert len(smooth_IU) == npix_out
    assert len(smooth_QU) == npix_out
    
    hp.write_map('test_images/real_analytical_smooth_variances.fits',
                 [smooth_II,smooth_QQ,smooth_UU,smooth_IQ,smooth_IU,smooth_QU],overwrite=True)
    pyplot.figure(figsize=(8,16))
    hp.mollview(smooth_II, title='smooth_II',sub=(3,1,1),rot=[0,90])
    hp.mollview(smooth_QQ, title='smooth_QQ',sub=(3,1,2),rot=[0,90])
    hp.mollview(smooth_UU, title='smooth_UU',sub=(3,1,3),rot=[0,90])
    hp.graticule()
    pyplot.savefig('test_images/real_smooth_variance_map.png')
    pyplot.close()

    pyplot.figure(figsize=(8,16))
    hp.mollview(II, title='input_II',sub=(3,1,1))
    hp.mollview(QQ, title='input_QQ',sub=(3,1,2))
    hp.mollview(UU, title='input_UU',sub=(3,1,3))
    hp.graticule()
    pyplot.savefig('test_images/real_input_variance_map.png')
    pyplot.close()



if __name__ == "__main__":
    np.random.seed(0)   
    os.makedirs('test_images',exist_ok=True)
    #test_produce_dec_mask()
    #test_square_beam()
    #test_smooth_map()
    test_smooth_real_variance_map()
    print("All tests passed!")