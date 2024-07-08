import jax
import jax.numpy as jnp


# Coefficients for Newton-Cotes quadrature
#
# These are the points being used
#  to construct the local interpolating polynomial
#  a are the weights for Newton-Cotes integration
#  B is the error coefficient.
#  error in these coefficients grows as N gets larger.
#  or as samples are closer and closer together

# You can use maxima to find these rational coefficients
#  for equally spaced data using the commands
#  a(i,N) := integrate(product(r-j,j,0,i-1) * product(r-j,j,i+1,N),r,0,N) / ((N-i)! * i!) * (-1)^(N-i);
#  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
#  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
#  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
#
# pre-computed for equally-spaced weights
#
# num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
#
#  a = num_a*array(int_a)/den_a
#  B = num_B*1.0 / den_B
#
#  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
#    where k = N // 2
#
_builtincoeffs = {
    1: (1,2,[1,1],-1,12),
    2: (1,3,[1,4,1],-1,90),
    3: (3,8,[1,3,3,1],-3,80),
    4: (2,45,[7,32,12,32,7],-8,945),
    5: (5,288,[19,75,50,50,75,19],-275,12096),
    6: (1,140,[41,216,27,272,27,216,41],-9,1400),
    7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),
    8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
        -2368,467775),
    9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
                 15741,2857], -4671, 394240),
    10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
                   -260550,272400,-48525,106300,16067],
         -673175, 163459296),
    11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
                      15493566,15493566,-9595542,25226685,-3237113,
                      13486539,2171465], -2224234463, 237758976000),
    12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
                      87516288,-87797136,87516288,-51491295,35725120,
                      -7587864,9903168,1364651], -3012, 875875),
    13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
                           156074417954,-151659573325,206683437987,
                           -43111992612,-43111992612,206683437987,
                           -151659573325,156074417954,-31268252574,
                           56280729661,8181904909], -2639651053,
         344881152000),
    14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
                         -6625093363,12630121616,-16802270373,19534438464,
                         -16802270373,12630121616,-6625093363,3501442784,
                         -770720657,710986864,90241897], -3740727473,
         1275983280000)
    }


class NewtonCotesIntegrator:
    def __init__(self):
        """
        Class for performing Newton-Cotes integration of y over the x range.
        """

    @staticmethod
    def _newton_cotes_coeffs(rn: int or jnp.DeviceArray, equal: int = 0):
        """
        Jax conversion of the scipy.integrate.newton_cotes routine.
        :param rn: the integer order for equally-spaced data or the relative positions of the samples with the first sample
            at 0 and the last at N, where N+1 is the length of rn. N is the order of the Newton-Cotes integration.
        :param equal: set to 1 to enforce equally space data.

        :returns an: 1-D array of weights to apply to the function at the provided sample positions.
        :returns B: error coefficient.
        """
        try:
            N = len(rn) - 1
            if equal:
                rn = jnp.arange(N + 1)
            elif jnp.all(jnp.diff(rn) == 1):
                equal = 1
        except Exception:
            N = rn
            rn = jnp.arange(N + 1)
            equal = 1

        if equal and N in _builtincoeffs:
            na, da, vi, nb, db = _builtincoeffs[N]
            an = na * jnp.array(vi, dtype=float) / da
            return an, float(nb) / db

        if (rn[0] != 0) or (rn[-1] != N):
            raise ValueError("The sample positions must start at 0 and end at N")
        yi = rn / float(N)
        ti = 2 * yi - 1
        nvec = jnp.arange(N + 1)
        C = ti ** nvec[:, jnp.newaxis]
        Cinv = jnp.linalg.inv(C)
        # improve precision of result
        for i in range(2):
            Cinv = 2 * Cinv - Cinv.dot(C).dot(Cinv)
        vec = 2.0 / (nvec[::2] + 1)
        ai = Cinv[:, ::2].dot(vec) * (N / 2.)

        if (N % 2 == 0) and equal:
            BN = N / (N + 3.)
            power = N + 2
        else:
            BN = N / (N + 2.)
            power = N + 1

        BN = BN - jnp.dot(yi ** power, ai)
        p1 = power + 1
        fac = power * jnp.log(N) - jax.scipy.special.gammaln(p1)
        fac = jnp.exp(fac)
        return ai, BN * fac

    def _integrate_interval(self, x_sub: jnp.DeviceArray, y_sub: jnp.DeviceArray):
        """
        Performs Newton-Cotes integration over some sub-interval.
        :param x_sub: x-values over sub-interval.
        :param y_sub: y-values over sub-interval.
        :return: Newton-Cotes integration value.
        """
        if len(y_sub) != len(x_sub):
            raise ValueError('x_sub and y_sub must have the same length.')
        x_max = jnp.max(x_sub)
        x_min = jnp.min(x_sub)
        this_order = len(x_sub) - 1
        dx = (x_max - x_min) / this_order
        x_sub = (x_sub - x_min) / (x_max - x_min) * this_order
        x_sub = x_sub.at[0].set(x_sub[0].astype(jnp.int32))
        x_sub = x_sub.at[-1].set(x_sub[-1].astype(jnp.int32))
        a_i, B = self._newton_cotes_coeffs(x_sub, equal=0)

        return jnp.sum(a_i * y_sub) * dx, B

    def integrate_data(self, x: jnp.DeviceArray, y: jnp.DeviceArray, order: int):
        """
        Integrate the y data over the x-range, splitting data into intervals of length order.
        :param x: x-values at which function is evaluated.
        :param y: corresponding function values at each x.
        :param order: Newton-Cotes integration order.
        :return: integration result.
        """
        sort_arg = jnp.argsort(x)
        x = x.at[:].set(x[sort_arg])
        y = y.at[:].set(y[sort_arg])
        integral = 0.

        for sub_interval_no in range(0, jnp.ceil(len(x) / order).astype(jnp.int32)):
            sub_interval_limits = [sub_interval_no * order, (sub_interval_no + 1) * order + 1]
            x_sub = x[sub_interval_limits[0]:sub_interval_limits[1]]
            y_sub = y[sub_interval_limits[0]:sub_interval_limits[1]]
            if len(x_sub) == 1:
                continue
            integral_chunk, _ = self._integrate_interval(x_sub, y_sub)
            integral += integral_chunk

        return integral
