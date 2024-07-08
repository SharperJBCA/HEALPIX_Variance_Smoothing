import jax.numpy as jnp
import numpy as np
from cbass_pipeline.pipeline.Deconvolution.integration.interpolate import InterpolatedUnivariateSpline


def hanning(l, lstart, lcut):
    return jnp.cos(np.pi * (l - lstart) / (lcut - lstart) / 2.0) ** 2


class TransferFunction:

    def __init__(self,
                 spin0_bl: jnp.DeviceArray,
                 spin2_bl: jnp.DeviceArray,
                 gauss_fwhm: float,
                 spline_lmin: int = 300,
                 spline_lmax: int = 700,
                 spline_order: int = 3,
                 derivative_order: int = 1):
        """
        Class for computing tapered transfer functions for deconvolving the C-BASS maps.

        :param spin0_bl: spin-0 Legendre transform of the beam.
        :param spin2_bl: spin-2 Legendre transform of the beam.
        :param gauss_fwhm: FWHM of the target Gaussian beam in degrees.
        :param spline_lmin: minimum multipole to use in transfer function spline fit.
        :param spline_lmax: maximum multipole to use in transfer function spline fit.
        :param spline_order: transfer function spline order.
        :param derivative_order: order of the spline derivative used to determine the start of tapering.
        """

        if len(spin0_bl) != len(spin2_bl):
            raise ValueError('spin0_bl and spin2_bl must have the same size.')
        if spline_order > 3:
            raise NotImplementedError
        if derivative_order > spline_order:
            raise ValueError('derivative_order must be less than or equal to spline_order.')

        self.spin0_bl = spin0_bl
        self.spin2_bl = spin2_bl
        # Avoid divide by zero error in spin-2 stuff.
        self.spin2_bl = self.spin2_bl.at[0:2].set(1.0)
        self.ell = jnp.arange(len(spin0_bl))

        gauss_sigma = np.radians(gauss_fwhm) / np.sqrt(8.0 * np.log(2.0))
        self.spin0_gl = jnp.exp(-self.ell * (self.ell + 1.0) * gauss_sigma ** 2 / 2.0)
        self.spin2_gl = jnp.exp(-(self.ell * (self.ell + 1.0) - 4.0) * gauss_sigma ** 2 / 2.0)
        self.spin2_gl = self.spin2_gl.at[0:2].set(0)

        self.spline_lmin = spline_lmin
        self.spline_lmax = spline_lmax
        print(type(self.spline_lmax))
        self.spline_order = spline_order
        self.derivative_order = derivative_order

        self.lcut = None
        self.lstart = None
        self.spin0_rl_spline = None
        self.spin2_rl_spline = None
        self.raw_spin0_rl = None
        self.raw_spin2_rl = None
        self.hann = None
        self.tapered_spin0_rl = None
        self.tapered_spin2_rl = None

    def _calculate_splines(self):
        self.spin0_rl_spline = InterpolatedUnivariateSpline(self.ell[self.spline_lmin:self.spline_lmax + 1],
                                                            self.raw_spin0_rl[self.spline_lmin:self.spline_lmax + 1],
                                                            k=self.spline_order)
        self.spin2_rl_spline = InterpolatedUnivariateSpline(self.ell[self.spline_lmin:self.spline_lmax + 1],
                                                            self.raw_spin2_rl[self.spline_lmin:self.spline_lmax + 1],
                                                            k=self.spline_order)

    def transfer_functions(self):
        self.raw_spin0_rl = self.spin0_gl / self.spin0_bl
        self.raw_spin2_rl = self.spin2_gl / self.spin2_bl
        self._calculate_splines()

        deriv_spin0_rl = self.spin0_rl_spline.derivative(self.ell[self.spline_lmin:self.spline_lmax + 1],
                                                         n=self.derivative_order)
        min_pos_deriv_idx = jnp.amin(jnp.where(deriv_spin0_rl > 0)[0])
        self.lstart = self.ell[self.spline_lmin:self.spline_lmax + 1][min_pos_deriv_idx]
        self.lcut = jnp.amin(jnp.where(self.spin0_bl < 0.0)[0])

        self.hann = jnp.ones(len(self.spin0_bl))
        taper_idx = jnp.where(jnp.logical_and(self.ell >= self.lstart, self.ell <= self.lcut))[0]
        self.ell_window_min = self.ell[taper_idx][0]
        self.ell_window_max = self.ell[taper_idx][-1]
        self.hann = self.hann.at[taper_idx].set(hanning(self.ell[taper_idx], self.lstart, self.lcut))
        self.hann = self.hann.at[jnp.amax(taper_idx) + 1:].set(0.0)

        self.tapered_spin0_rl = self.raw_spin0_rl * self.hann
        self.tapered_spin2_rl = self.raw_spin2_rl * self.hann