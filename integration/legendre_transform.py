from jax.numpy import trapz
import jax.numpy as jnp
import jax
from jax.config import config
import numpy as np
from integration.newton_cotes import NewtonCotesIntegrator
config.update("jax_enable_x64", True)


def get_polynomials(theta: jnp.DeviceArray, lmax: int):
    ell = jnp.arange(lmax + 1)
    cos = jnp.cos(theta)
    p0 = jnp.ones_like(theta, dtype=jnp.float64) / np.sqrt(4.0 * np.pi)
    p1 = cos * np.sqrt(3.0 / (4.0 * np.pi))
    p0l = jnp.zeros((lmax + 1, theta.shape[0]), dtype=jnp.float64)
    p0l = p0l.at[0, :].set(p0)
    p0l = p0l.at[1, :].set(p1)

    print(f'Calculating Legendre polynomials up to l = {lmax}')

    def loop_fn(l, p0l):
        p2 = jnp.sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0)) * (
                cos * p0l[l - 1, :] - (l - 1.0) * p0l[l - 2, :] / jnp.sqrt(4.0 * (l - 1.0) ** 2 - 1.0)) / l
        p0l = p0l.at[l, :].set(p2)
        return p0l
    p0l = jax.lax.fori_loop(2, lmax + 1, loop_fn, p0l)

    print(f'Calculating second derivatives of Legendre polynomials up to l = {lmax}')
    d2p = lambda l: l * (2.0 * cos * jnp.sqrt((2.0 * l + 1.0) / (2.0 * l - 1.0)) * p0l[l - 1, :]
                         + ((l - 1.0) * cos ** 2 - l - 1.0) * p0l[l, :]) / (cos ** 2 - 1.0) ** 2
    d2p0l = jax.vmap(d2p)(ell)
    d2p0l = d2p0l.at[:, 0].set(
        jnp.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi)) * ell * (ell + 1.0) * (
                ell ** 2 + ell - 2.0) / 8.0)
    d2p0l = d2p0l.at[:, theta.shape[0] - 1].set(
        jnp.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi)) * (-1) ** ell * ell * (ell + 1.0) * (
                ell ** 2 + ell - 2.0) / 8.0)

    return p0l, d2p0l


class Legendre:

    def __init__(self,
                 theta: jnp.DeviceArray,
                 beam: jnp.DeviceArray,
                 lmax: int,
                 p0l: jnp.DeviceArray = None,
                 d2p0l: jnp.DeviceArray = None,
                 integrator: str = 'trapezium'):

        """
        Class for performing Legendre transforms of beam profiles ...

        :param theta: angles at which beam is samples (radians).
        :param beam: beam profile values corresponding to theta.
        :param lmax: maximum multipole we want to evaluate transfer functions to.
        :param p0l: Legendre polynomials up to lmax, evaluated at beam sample locations.
        :param d2p0l: Second derivative of the Legendre polynomials up to lmax, evaluated at beam sample locations.
        :param integrator: integration method to use. Either trapezium or boole.
        """

        self.theta = jnp.asarray(theta, dtype=jnp.float64)
        self.beam = jnp.asarray(beam, dtype=jnp.float64)
        self.lmax = lmax
        self.integrator = integrator
        self.ell = jnp.arange(self.lmax + 1)
        self.norm_scaling = None

        if p0l is None or d2p0l is None:
            self.p0l, self.d2p0l = get_polynomials(self.theta, self.lmax)
        else:
            self.p0l = p0l
            self.d2p0l = d2p0l

        if len(self.theta) != len(self.beam):
            raise ValueError("Beam and theta must have same size!")
        if self.integrator not in ('trapezium', 'boole'):
            raise ValueError("Integrator must be one of trapezium or boole.")

    def spin0_transform(self):
        sin = jnp.sin(self.theta)
        if self.integrator == 'boole':
            wl0 = jax.vmap(lambda l: NewtonCotesIntegrator().integrate_data(self.theta,
                                                                            self.beam * self.p0l[l, :] * sin,
                                                                            order=4))(self.ell)
        elif self.integrator == 'trapezium':
            wl0 = jax.vmap(lambda l: trapz(self.beam * self.p0l[l, :] * sin, self.theta))(self.ell)
        wl0 *= 2.0 * np.pi * jnp.sqrt(4.0 * np.pi / (2.0 * self.ell + 1.0))
        self.norm_scaling = wl0[0]

        return wl0

    def spin2_transform(self):
        cos = jnp.cos(self.theta)
        sin = jnp.sin(self.theta)

        spin2_term = lambda l: (l + 2.0) * (cos - 2.0) * jnp.sqrt(
            (2.0 * l + 1.0) / (2.0 * l - 1.0)) * self.d2p0l[l - 1, :] + (2.0 * (l - 1.0) - 0.5 * l * (
                    l - 1.0) * sin ** 2 - l + 4.0) * self.d2p0l[l, :]
        if self.integrator == 'boole':
            wl2 = jax.vmap(lambda l: NewtonCotesIntegrator().integrate_data(self.theta,
                                                                            self.beam * spin2_term(l) * sin,
                                                                            order=4))(self.ell[2:])
        elif self.integrator == 'trapezium':
            wl2 = jax.vmap(lambda l: trapz(self.beam * spin2_term(l) * sin, self.theta))(self.ell[2:])
        nl2 = 2.0 / (self.ell[2:] * (self.ell[2:] + 2.0) * (self.ell[2:] + 1.0) * (self.ell[2:] - 1.0))
        wl2 *= 2.0 * np.pi * nl2 * jnp.sqrt(4.0 * np.pi / (2.0 * self.ell[2:] + 1.0))
        wl2 = jnp.insert(wl2, 0, jnp.zeros(2))

        return wl2

    def get_normalisation(self):
        if self.norm_scaling is None:
            self.spin0_transform()
        return self.norm_scaling
