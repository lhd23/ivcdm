import numpy as np

from scipy.integrate import odeint,quad,solve_ivp
from scipy.interpolate import interp1d,CubicSpline
from scipy.special import hyp2f1,factorial
from numpy.polynomial import chebyshev,polynomial


THREEHALF = 1.5
THIRD = 1./3.
TWOTHIRD = 2./3.
FOURTHIRD = 4./3

SPEEDOFLIGHT = 2.99792458e8 #m/s

# H0_PLK15 = 67.3
# SIGMA80_PLK15 = 0.829

class LCDM: #background/base class

    """
    Note we pass all parameters, even ones not needed;
    kwargs will catch (non-cosmological) parameters not
    used (these cannot be referenced)
    """

    def __init__(self, Om0, Ol0, Ob0=None,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        self.H0 = float(H0)   # units = km/s/Mpc
        self.Om0 = float(Om0) # Omega_{m0}
        self.Ol0 = float(Ol0) # Omega_{Lambda0}

        if Ob0 is None:
            self.Ob0 = Ob0
        else:
            self.Ob0 = float(Ob0)
            if self.Ob0 < 0.0:
                raise ValueError('Baryonic density parameter cannot be negative')
            # if self._Ob0 > self._Om0:
            #     raise ValueError("Baryonic density can not be larger than "
            #                      "total matter density")

        if sigma80 is None:
            self.sigma80 = sigma80
        else:
            self.sigma80 = float(sigma80)
            if self.sigma80 < 0.0:
                raise ValueError('sigma_80 cannot be negative')

        # other parameters
        self.Tcmb = Tcmb
        self.Neff = Neff #effective no. of neutrinos

        # growth history splines
        self.D_spl = None
        self.dD_spl = None
        self.D0 = None

        # problem points when doing integrals (e.g. discontinuities)
        self._points = None

    @property
    def dH(self):
        return (SPEEDOFLIGHT/1.e3)/self.H0 # c/H_0 [Mpc]

    @property
    def h(self):
        return self.H0/100.0

    @property
    def Oc0(self): # Omega_{cdm0}
        if self.Ob0 is None:
            return None
        else:
            return self.Om0-self.Ob0

    @property
    def Oy0(self): #photon density parameter
        return 2.47282e-5/self.h**2 #assuming Tcmb=2.7255K
    
    @property
    def On0(self): #neutrino density parameter
        return self.Oy0*self.Neff*(7./8)*pow(4./11,FOURTHIRD)

    @property
    def Or0(self): #radiation density parameter
        return self.Oy0+self.On0

    @property
    def Ok0(self):
        return 1.0-self.Om0-self.Ol0-self.Or0

    @property
    def wb(self): # omega_b = Omega_{b0} * h^2
        return self.Ob0*self.h**2

    @property
    def wm(self): # omega_m = Omega_{m0} * h^2
        return self.Om0*self.h**2

    def scale_factor(self,z):
        return 1.0/(1.0+z)

    def a2z(self,a): # scale factor to redshift
        return 1.0/a-1.0

    def Omega_k(self,a):
        return self.Ok0/a**2

    def Omega_b(self,a):
        return self.Ob0/a**3

    def Omega_r(self,a):
        return self.Or0/a**4

    def Omega_cdm(self,a):
        return self.Oc0/a**3

    def Omega_vac(self,a):
        return self.Ol0

    def Omega_mat(self,a):
        return self.Om0/a**3

    def E_Hub(self,a):
        """
        Computes E(z) = H(z)/H0
        """
        Omv = self.Omega_vac(a)
        Omm = self.Omega_mat(a)
        Omk = self.Omega_k(a)
        Omr = self.Omega_r(a)
        E2 = Omm+Omv+Omk+Omr
        # E2=Omm+Omv+Omk
        if np.all(E2 > 0.0):
            return np.sqrt(E2)
        else:
            return np.NaN

    def Hub(self,z): # note arg is REDSHIFT z not a
        a = self.scale_factor(z)
        return self.H0*self.E_Hub(a)

# perturbations

    def solve_growth(self):
        """
        The equations are integrated starting from
        a time when the Universe was matter dominated
        so that D ~ a and dD/da ~ 1 (EdS). We will
        take this time to be at z ~ 30 or a ~ 0.03
        """
        a_arr = np.linspace(0.03,1.01,150)
        dDi, Di = 1.0, a_arr[0]
        y0 = [dDi,Di]
        y = odeint(self._solve_growth,y0,a_arr)
        dDda, D = y[:,0], y[:,1]
        # spline the growth history and cache it
        self.D_spl = interp1d(a_arr,D)
        self.dD_spl = interp1d(a_arr,dDda)
        self.D0 = self.D_spl(1.0)

    def D(self,a):
        if self.D_spl is None:
            self.solve_growth()
        return self.D_spl(a) #unitless

    def dDda(self,a):
        if self.dD_spl is None:
            self.solve_growth()
        return self.dD_spl(a) #unitless

    def f_growthrate(self,a):
        """
        Computes f := d(lnD)/d(lna) = (a/D)*dD/da
        note D normalised so D(a=1)=1
        """
        D = self.D(a)/self.D0
        dDda = self.dDda(a)/self.D0
        return (a/D)*dDda

    def fsigma_8(self,z):
        """
        f times sigma8 at redshift z

        Notes
        -----
        This function calculates at redshift NOT scale factor
        D normalised so D(a=1)=1
        """
        if self.sigma80 is None:
            raise ValueError('sigma80 cannot be None type')
        a = self.scale_factor(z)
        D = self.D(a)/self.D0
        f = self.f_growthrate(a)
        sigma8 = self.sigma80*D
        return f*sigma8

    def _solve_growth(self,y,x):
        # y0=D'; y1=D; x=scale factor
        # with curvature
        E = self.E_Hub(x) # H/H0
        Omv = self.Omega_vac(x)/E**2
        Omm = self.Omega_mat(x)/E**2
        Omk = self.Omega_k(x)/E**2
        A = (3.0-THREEHALF*Omm-Omk)/x
        B = -(THREEHALF*Omm)/x**2
        dy0 = -A*y[0]-B*y[1]
        dy1 = y[0]
        return [dy0,dy1]

# distance measures

    @property
    def zstar(self):
        """
        Redshift at photon decoupling based on
        fitting formula in Hu & Sugiyama (1996),
        see equation E1 in appendix
        """
        wm = self.wm #matter density
        wb = self.wb #baryon density

        g1 = 0.0783*pow(wb, -0.238)/(1.0+39.5*pow(wb, 0.763))
        g2 = 0.560/(1.0+21.1*pow(wb, 1.81))
        return 1048.0*(1.0+0.00124*pow(wb, -0.738)) * \
            (1.0+g1*pow(wm, g2))

    @property
    def zdrag(self):
        """
        Redshift at baryon decoupling based on
        fitting formula either in:

        (A) Hu & Sugiyama (1996): eqn (E2)
        (B) Eisenstein & Hu (1998): eqn (4)

        """
        # Hu & Sugiyama (1996)
        a1 = 1345.0

        # Eisenstein & Hu (1998)
        # a1 = 1291.0

        wm = self.wm #matter density
        wb = self.wb #baryon density

        b1 = 0.313*pow(wm, -0.419)*(1.0+0.607*pow(wm, 0.674))
        b2 = 0.238*pow(wm, 0.223)
        return a1*pow(wm, 0.251)/(1.0+0.659*pow(wm, 0.828)) * \
            (1.0+b1*pow(wb, b2))

    @property
    def r_s_star(self):
        """
        The dimensionful physical sound horizon [Mpc]
        at photon decoupling

        Returns
        -------
        r_s_
        """
        return self.dH*self.r_s(self.zstar)

    @property
    def dA_star(self):
        """
        The dimensionful angular diameter distance [Mpc] to the
        last scattering surface
        """
        return self.dH*self.dA(self.zstar)

    @property
    def theta_star(self):
        """
        Angular size of sound horizon at photon decoupling

        Computes the dimensionless quantity

            theta_* = r_s(z_*) / d_A(z_*)

        where r_s is the _physical_ sound horizon (units Mpc)
        d_A the _physical_ angular diamter distance (units Mpc).
        (We could also have used the _comoving_ rs and dA;
        it does not matter for ratios/angles)
        """
        return self.r_s_star/self.dA_star

    @property
    def lA(self):
        """Position of the first acoustic peak

            l_A := pi * DA(zstar) / rs_comoving(zstar)
                 = pi / theta_star
        """
        return np.pi/self.theta_star

    @property
    def R_shift(self):
        """
        Shift parameter (Efstathiou & Bond 1999) defined as

            R = sqrt[Om0 * H0^2] * D_A(z_star) / c 
              = sqrt[Om0] * D_A(z_star) / dH

        where dH=c/H0 and D_A(z) = (1+z) * d_A is the _comoving_ angular
        diameter distance. R is dimensionless. 

        Notes
        -----
        In this class we work with dimensionless distances hence
        we do not need to divide by dH.
        """
        zstar = self.zstar
        DA = (1.0+zstar)*self.dA(zstar) # comoving
        return np.sqrt(self.Om0)*DA

    def _integrand(self,z):
        """
        Computes 1/E(z)=H0/H(z) at redshift z
        """
        a = self.scale_factor(z)
        E = self.E_Hub(a)
        return 1./E

    def dL(self,z):
        """
        Computes the dimensionless quantity dL/dH, dH=c/H0
        """
        if np.isclose(z,0.0):
            return 0.0
        zp1 = z+1.0
        Ok0 = self.Ok0
        Om0 = self.Om0
        Ol0 = self.Ol0
        if np.isfinite(self._integrand(z)): # prevent negative square roots

            if np.isclose(Ok0, 0.0, atol=1e-8): # spatially flat
                if np.isclose(Om0, 1.0, atol=1e-8, rtol=1e-10): # EdS
                    return 2.*zp1*(1.-1./np.sqrt(zp1))
                elif np.isclose(Ol0, 1.0, atol=1e-8, rtol=1e-10): # dS
                    return zp1*z
                else:
                    I,err = quad(self._integrand, 0.0, z, points=self._points, limit=100)
                    return zp1*I

            if Ok0 < 0.0: # closed
                I,err = quad(self._integrand, 0.0, z, points=self._points)
                s = np.sqrt(np.absolute(Ok0))
                return zp1*np.sin(s*I)/s
            else: # Ok0 > 0 (open)
                if np.isclose(Ok0, 1.0, atol=1e-8, rtol=1e-10): # Milne
                    return 0.5*(zp1**2-1.0) # = zp1*np.sinh(np.log(zp1))
                else:
                    I,err = quad(self._integrand, 0.0, z, points=self._points)
                    s = np.sqrt(Ok0)
                    return zp1*np.sinh(s*I)/s
        else:
            return float(1e7)

    def dA(self,z):
        """
        Computes angular diameter distance divided by dH
        using the reciprocity relation
        """
        return self.dL(z)/(1.+z)**2

    def dist_mod(self,z):
        dL_ = self.dL(z)
        if dL_ < 0.0:
            raise ValueError('luminosity distance cannot be negative: \
                              check cosmological parameters are valid')
        else:
            return 5.*np.log10(self.dH*dL_) + 25.

    def D_V(self,z):
        """
        Return dimensionless quantity D_V/dH where

           D_V(z) = [(1+z)^2 dA^2 cz/H]^{1/3}

        To get in units length multiply by dH=c/H0
        """
        a = self.scale_factor(z)
        # Parallel to l.o.s.,
        lpara = z/self.E_Hub(a) #unitless quantity = (cz/H)/dH

        # Perpendicular to l.o.s.
        lperp = (1.+z)*self.dA(z) #unitless qty = ((1+z)dA)/dH
        D_V_3 = lpara*lperp**2
        return pow(D_V_3, THIRD)

    def r_s_comoving(self,z):
        """
        The _comoving_ sound horizon r_s divided dH: the distance
        the sound waves have travelled by a redshift of z. Usually
        this redshift will be either at baryon decoupling (zdrag)
        or photon decoupling (z_*).

        Notes
        -----
        It is assumed that interaction is negligible at z > 1000
        so that we can use the standard formula (see eg eqn 8.82
        in Dodelson)
        """
        fb = self.Ob0/self.Om0
        fy = self.Oy0/self.Or0 # should be approx 0.6
        aeq = self.Or0/self.Om0
        a = self.scale_factor(z)
        y = a/aeq
        Req = 0.75*fb/fy
        R = Req*y
        p1 = np.sqrt(R+Req)
        p2 = np.sqrt(R+1.0)
        p3 = 1.0+np.sqrt(Req)
        A = 2./np.sqrt(3.*self.Om0)*np.sqrt(aeq/Req)
        return A*np.log((p1+p2)/p3)

    def r_s(self,z):
        """The physical sound horizon divided by dH"""
        return self.r_s_comoving(z)*self.scale_factor(z)

    def d_z(self,z):
        if np.isclose(z, 0.0, atol=1e-8):
            print 'Warning: d_z at z=0 is singular'
            return float(1e11)
        return self.r_s_comoving(self.zdrag)/self.D_V(z)


class IVCDM(LCDM):
    """
    This class extends LCDM. It makes more sense to make IVCDM
    the parent class from which LCDM inherit as child class. 
    But the reason I do not do this is because eventually this 
    LCDM should work standalone.

    By default this class assumes the interaction term is
    that of the generalised Chalygin gas.
    I.e. with interaction term

        Q = 3 alpha H V rho_c / (rho_c + V)
          = 3 alpha H V [1 - V / (rho_c + V)]

    Replaces methods Omega_{cdm,vac}
    """
    def __init__(self, Om0, Ol0, Ob0, q,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        self.q = float(q) # this must be set before calling LCDM
        LCDM.__init__(self, Om0, Ol0, Ob0,
                      sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff)

        if self.Ob0 is None:
            raise ValueError('In class IVCDM Ob0 cannot be None type')

    @property
    def OgCg0(self):
        return self.Oc0+self.Ol0

    def _Omega_gCg(self,a):
        alpha = THIRD*self.q
        b = self.Oc0/self.OgCg0
        f = (1.0-b)+b*pow(a, -3.0*(1.0+alpha))
        try:
            index = 1.0/(1.0+alpha)
        except:
            raise ValueError('Divide by zero: alpha is {}'.format(alpha))
        if f > 0.0:
            return pow(f, index)
        else:
            # negative number cannot be raised to a fractional power
            return np.NaN

    def Omega_gCg(self,a):
        return self.OgCg0*self._Omega_gCg(a)

    def Omega_cdm(self,a):
        return self.Omega_gCg(a)-self.Omega_vac(a)

    def Omega_mat(self,a):
        return self.Omega_cdm(a)+self.Omega_b(a)

    def Omega_vac(self,a):
        alpha = THIRD*self.q
        _Omg = self._Omega_gCg(a)
        return self.Ol0/pow(_Omg, alpha)

    def _solve_growth(self,y,x):
        # y0=D'; y1=D; x=scale factor
        # with curvature
        E2 = self.E_Hub(x)**2 # (H/H0)^2
        Omv = self.Omega_vac(x)/E2
        Omc = self.Omega_cdm(x)/E2
        Omm = self.Omega_mat(x)/E2
        Omk = self.Omega_k(x)/E2
        Qtilde = self.q*Omc*Omv/(Omc+Omv)/Omm # Q/(H rho_m)
        dQ_rho = -Qtilde*(THREEHALF*Omm+Omk-(self.q+3.0)*Omc/(Omc+Omv)) \
                -Qtilde**2*(Omm/Omc-1.0) # 1/H d/dlna (Q/rho_m) see eqn (87)
        A = (3.0-THREEHALF*Omm-Omk-Qtilde)/x
        B = -(THREEHALF*Omm+2.0*Qtilde+dQ_rho)/x**2
        dy0 = -A*y[0]-B*y[1]
        dy1 = y[0]
        return [dy0,dy1]


class FlatLCDM(LCDM):
    # this inherits from LCDM not extends (restricts functionality)
    def __init__(self, Om0, Ob0=None,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        Ol0 = 1.0 - Om0 # must be called first; assumes radn is tiny
        LCDM.__init__(self, Om0, Ol0, Ob0,
                      sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff)

    @property
    def Ok0(self):
        return 0.0


class FlatIVCDM(IVCDM):
    # this inherits from IVCDM (restricts functionality)
    def __init__(self, Om0, Ob0, q,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        Ol0 = 1.0 - Om0
        IVCDM.__init__(self, Om0, Ol0, Ob0, q, 
                       sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff)

    @property
    def Ok0(self):
        return 0.0


class IVCDM_binned(IVCDM):
    """
    Replaces methods relating to the particular Q(a) in IVCDM
    with

    Q(a) = q(a) H(a) V

    where V = rho_vac and q(a) is a piecewise constant function
    with amplitudes given by the 1d array qs. The bin sizes are
    delineated by the 1d array zbins
    """
    def __init__(self, Om0, Ol0, Ob0, qs, zbins=None, zmax=1.5,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        IVCDM.__init__(self, Om0, Ol0, Ob0, q=0.0,
                      sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff)

        self.qs = np.asarray(qs) # array of interaction amplitudes
        self.Nbins = self.qs.size

        if zbins is None:
            self.zbins = np.linspace(0.0, zmax, self.Nbins+1)
        else:
            self.zbins = np.asarray(zbins) # location of bin edges
        self.abins = self.scale_factor(self.zbins)

        # ensure binning starts at z=0 and the
        # 1d arrays are right size
        assert self.zbins[0] == 0.0

        # break integrals up at discontinuities
        self._points = list(self.zbins)

    def _get_bin(self,a):
        """
        Note the bin is a number between 1 and Nbins
        I.e. the first bin is 1 NOT 0.
        If a is outside bin range then either 0 will
        be returned OR len(zbins) = Nbins + 1

        Membership in the ith bin is defined as
        [z_i, z_{i+1}) i.e. if the point is on the
        right edge then it is counted in the next bin
        """
        z = self.a2z(a) # convert to redshift
        return np.digitize(z, self.zbins) # index

    def _is_in_bin(self,a):
        """
        Checks scale factor is in binning range;
        used before calling _get_bin

        Since the right edge is not counted in the
        binning range (default behaviour of np.digitize)
        it must be strictly less than this upper limit
        """
        z = self.a2z(a)
        if self.zbins[0] <= z < self.zbins[-1]:
            return True
        else:
            return False

    def q_hist(self,a): # computes q(a)
        if self._is_in_bin(a):
            m = self._get_bin(a)
            assert 1 <= m <= self.Nbins
            return self.qs[m-1] # since bin 1 has index 0
        else:
            return 0.0

    def Qtilde(self,a): # Q/(H rho_m)
        Omv = self.Omega_vac(a)
        Omm = self.Omega_mat(a)
        return self.q_hist(a)*Omv/Omm

# overrides IVCDM parent methods

    def Omega_vac(self,a):
        # note we have to subtract 1 from the index of q
        # because is it one entry smaller than abins/zbins
        if self._is_in_bin(a):
            m = self._get_bin(a)
            assert 1 <= m <= self.Nbins
            Omv_ = pow(a/self.abins[m-1], self.qs[m-1])
            for i in range(1,m): #1,2,...,m-1
                Omv_ *= pow(self.abins[i]/self.abins[i-1], self.qs[i-1])
            return self.Ol0*Omv_
        else: # outside range so use standard LCDM result (q=0)
            return self.Ol0

    def Omega_cdm_old(self,a):
        if self._is_in_bin(a):
            m = self._get_bin(a)
            assert 1 <= m <= self.Nbins
            Omc_ = (1.0/a**3) * pow(self.abins[m-1]/a, self.qs[m-1])
            for i in range(1,m):
                Omc_ *= pow(self.abins[i-1]/self.abins[i], self.qs[i-1])
            return self.Oc0*Omc_
        else: # outside range so use standard result (q=0)
            return self.Oc0/a**3

    def _rq(self,m):
        r = 1.0
        if m > 1:
            for i in range(1,m):
                r *= pow(self.abins[i-1]/self.abins[i], self.qs[i-1])
        return r

    def Omega_cdm(self,a):
        # note we have to subtract 1 from the index of q
        # because is it one entry smaller than abins/zbins
        q = self.qs
        abins = self.abins
        if self._is_in_bin(a):
            m = self._get_bin(a)
            assert 1 <= m <= self.Nbins
            # first m-1 terms contribution (full bins)
            f1 = 0.0
            if m > 1:
                for i in range(1,m): #1,2,...,m-1
                    f1 += q[i-1]/(q[i-1]+3.0)*self._rq(i) \
                          *(abins[i]/a)**3 \
                          *((abins[i-1]/abins[i])**3 - pow(abins[i]/abins[i-1], q[i-1]))
            # last term contribution (partial bin)
            f2 = q[m-1]/(q[m-1]+3.0)*self._rq(m) \
                 *((abins[m-1]/a)**3 - pow(a/abins[m-1], q[m-1]))
            return self.Oc0/a**3 + self.Ol0*(f1+f2)
        else: # outside range so use standard result (q=0)
            return self.Oc0/a**3

    def Omega_gCg(self,a):
        return self.Omega_cdm(a)+self.Omega_vac(a)

    def _solve_growth(self,y,x):
        # y0=D'; y1=D; x=scale factor
        # with curvature
        E2 = self.E_Hub(x)**2 # (H/H0)^2
        Omv = self.Omega_vac(x)/E2
        Omc = self.Omega_cdm(x)/E2
        Omm = self.Omega_mat(x)/E2
        Omk = self.Omega_k(x)/E2
        vom = Omv/Omm
        Qtilde = self.Qtilde(x)
        dQ_rho = self.q_hist(x) \
                *(-vom*(THREEHALF*Omm+Omk) + Qtilde*(1.0+vom) + 3.0*vom)
        A = (3.0 - THREEHALF*Omm - Omk - Qtilde)/x
        B = -(THREEHALF*Omm + 2.0*Qtilde + dQ_rho)/x**2
        dy0 = -A*y[0]-B*y[1]
        dy1 = y[0]
        return [dy0,dy1]


class FlatIVCDM_binned(IVCDM_binned):
    # this inherits from IVCDM (restricts functionality)
    def __init__(self, Om0, Ob0, qs, zbins=None, zmax=1.5,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        Ol0 = 1.0 - Om0
        IVCDM_binned.__init__(self, Om0, Ol0, Ob0, qs,
                                zbins=zbins, zmax=zmax,
                                   sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff, **kwargs)

    @property
    def Ok0(self):
        return 0.0


class IVCDM_smooth(IVCDM):
    """
    Replaces methods relating to the particular Q(a) in IVCDM
    with

    Q(a) = q(a) H(a) V

    where V = rho_vac and q(a) is expanded using a smooth
    basis set e.g. polynomials
    """
    def __init__(self, Om0, Ol0, Ob0, qs, basis='Chebyshev', zmin=0.0, zmax=2.0, free_knots=None,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        IVCDM.__init__(self, Om0, Ol0, Ob0, q=0.0,
                      sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff)

        self.qs = np.asarray(qs) # array of expansion coefficients
        self.Nq = self.qs.size

        if basis in ['A', 'B', 'C', 'Chebyshev', 'cubic_spline']:
            self.basis = str(basis)
        else:
            self.basis = 'A'

        if self.basis == 'A': #1/(1+z)^n
            self.cn_n = np.zeros(self.Nq)
            self.cn_n[1:] = np.array([self.qs[n]/n for n in range(1,self.Nq)])
            self.expsumc = np.exp(-np.sum(self.cn_n))

            # construct coefficients (gamma) for rho cdm
            N = self.Nq - 1 # index of last q parameter
            alpha = np.zeros(2*N+1)
            alpha[:self.Nq] = [self.qs[k]/(self.qs[0]+k+3) for k in range(self.Nq)]

            beta = np.zeros_like(alpha)
            for k in range(1, self.Nq):
                s = np.sum([self.qs[n]*self.qs[k-n]/(k-n) for n in range(0, k-1 + 1)])
                beta[k] = s/(self.qs[0]+k+3)
            for k in range(self.Nq, 2*self.Nq-1):
                s = np.sum([self.qs[n]*self.qs[k-n]/(k-n) for n in range(k-N, N + 1)])
                beta[k] = s/(self.qs[0]+k+3)
            self.gamma = self.expsumc * (alpha + beta)
            self.sum_gamma = np.sum(self.gamma)

        # different parametrisation of interaction
        elif self.basis == 'Chebyshev':
            self.zmin = zmin
            self.zmax = zmax
            self.Delta_z = (zmax - zmin)/2.0
            self.zbar = (zmax + zmin)/2.0
            self.zbarp1 = self.zbar + 1.0

            # chebyshev coefficients of dDelta/dx and d^2Delta/dx^2
            self.dcheb_coeff = chebyshev.chebder(self.qs, m=1) # 1st deriv
            self.ddcheb_coeff = chebyshev.chebder(self.qs, m=2) # 2nd deriv
            self.dxdz = 1.0/self.Delta_z

            # q0 - q1 + q2 - q3 + ... (-1)^N qN
            alt_ones = np.ones((self.Nq,))
            alt_ones[1::2] = -1.0
            self.Delta0 = np.sum(alt_ones * self.qs)
            self.Delta0p1 = self.Delta0 + 1.0

        elif self.basis == 'B': # (1+z)^n
            self.qs_zder1 = polynomial.polyder(self.qs, m=1) # coefficients of dDelta/dz
            self.qs_zder2 = polynomial.polyder(self.qs, m=2) # coefficients of d2Delta/dz2
            self.cn_np3 = np.zeros(self.Nq+3)
            self.cn_np3[3:] = np.array([self.qs[n]/(n+3.0) for n in range(self.Nq)])
            self.sum_cn_np3 = np.sum(self.cn_np3)
            self.Delta0 = np.sum(self.qs)
            self.Delta0p1 = self.Delta0 + 1.0

        elif self.basis == 'C': # 1/(1+z)^n = a^n
            self.qs_ader1 = polynomial.polyder(self.qs, m=1) # coefficients of dDelta/da
            self.qs_ader2 = polynomial.polyder(self.qs, m=2) # coefficients of d2Delta/da2
            self.cn_nm3 = np.zeros(self.Nq-3)
            self.cn_nm3[1:] = np.array([self.qs[n]/(n-3.0) for n in range(4,self.Nq)])
            self.sum_cn_nm3 = np.sum(self.cn_nm3)
            self.Delta0 = np.sum(self.qs)
            self.Delta0p1 = self.Delta0 + 1.0

        elif self.basis == 'cubic_spline':
            assert free_knots[0] < free_knots[-1]
            assert free_knots.size == self.Nq
            assert free_knots[-1] < 1.0

            # The _free_ knots are uniformly spaced in units of scale factor.
            # The first and last knots are fixed and we insert them here:
            a_first = 0.97*free_knots[0]
            a_last = 1.0
            self.a_knots = np.concatenate(([a_first],free_knots,[a_last])) # x
            self.q_knots = np.concatenate(([0.0],self.qs,[0.0])) # y

            # Construct splines (zero, first and second derivatives):
            # 'clamped': first derivative is zero at endpoint
            # 'natural': second derivative is zero at endpoint
            self.Delta_spl = CubicSpline(x=self.a_knots, y=self.q_knots, bc_type=('clamped','natural'))
            self.dDelta_spl = self.Delta_spl.derivative(nu=1)
            self.d2Delta_spl = self.Delta_spl.derivative(nu=2)
            self._points = list(1.0/self.a_knots[::-1]-1.0)

        self.Omega_cdm_spl = None
        self.Omega_vac_spl = None # only for cubic spline


    def q_hist(self,a): # computes q(a)
        z = self.a2z(a)
        if self.basis == 'A':
            x = 1.0/(1.0+z)
            return polynomial.polyval(x, self.qs)
        elif self.basis in ['B', 'C', 'Chebyshev']:
            dDelta_dlna = a * self.dDelta_da(a)
            Omc_bar = self.Oc0/a**3
            Omv = self.Omega_vac(a)
            return -dDelta_dlna * (Omc_bar/Omv) / self.Delta0p1
        elif self.basis == 'cubic_spline':
            dDelta_dlna = a * self.dDelta_da(a)
            Omc = self.Omega_cdm(a)
            Omv = self.Omega_vac(a)
            return -dDelta_dlna * (Omc/Omv)

    def Qtilde(self,a): # Q/(H rho_m)
        Omv = self.Omega_vac(a)
        Omm = self.Omega_mat(a)
        return self.q_hist(a)*Omv/Omm

    def Delta(self,a):
        z = self.a2z(a)
        if self.basis == 'Chebyshev':
            if self.zmin <= z <= self.zmax:
                x = cheb2.z2x(z, self.zmin, self.zmax)
                return chebyshev.chebval(x, self.qs)
            else:
                return 0.0
        elif self.basis == 'B':
            x = 1.0 + z
            return polynomial.polyval(x, self.qs)
        elif self.basis == 'C':
            x = 1.0/(1.0+z)
            return polynomial.polyval(x, self.qs)
        elif self.basis == 'cubic_spline':
            if self.a_knots[0] <= a <= self.a_knots[-1]:
                return self.Delta_spl(a)
            else:
                return 0.0

    def dDelta_da(self,a): # d/da Delta
        z = self.a2z(a)
        if self.basis == 'Chebyshev':
            if self.zmin <= z <= self.zmax:
                x = cheb2.z2x(z, self.zmin, self.zmax)
                dzda = -1.0/a**2
                dxda = self.dxdz * dzda
                return dxda * chebyshev.chebval(x, self.dcheb_coeff)
            else:
                return 0.0
        elif self.basis == 'B':
            x = 1.0 + z
            dzda = -1.0/a**2
            return dzda * polynomial.polyval(x, self.qs_zder1)
        elif self.basis == 'C':
            x = 1.0/(1.0+z)
            return polynomial.polyval(x, self.qs_ader1)
        elif self.basis == 'cubic_spline':
            if self.a_knots[0] <= a <= self.a_knots[-1]:
                return self.dDelta_spl(a)
            else:
                return 0.0

    def d2Delta_da2(self,a): # (d/da)^2 Delta
        z = self.a2z(a)
        if self.basis == 'Chebyshev':
            if self.zmin <= z <= self.zmax:
                x = cheb2.z2x(z, self.zmin, self.zmax)
                dzda = -1.0/a**2
                dxda = self.dxdz * dzda
                d2x_da2 = (2.0/a**3) * self.dxdz # d^x/da^2
                return d2x_da2 * chebyshev.chebval(x, self.dcheb_coeff) \
                    + (dxda**2) * chebyshev.chebval(x, self.ddcheb_coeff)
            else:
                return 0.0
        elif self.basis == 'B':
            x = 1.0 + z
            dzda = -1.0/a**2
            d2z_da2 = 2.0/a**3
            return d2z_da2 * polynomial.polyval(x, self.qs_zder1) \
                + (dzda**2) * polynomial.polyval(x, self.qs_zder2)
        elif self.basis == 'C':
            x = 1.0/(1.0+z)
            return polynomial.polyval(x, self.qs_ader2)
        elif self.basis == 'cubic_spline':
            if self.a_knots[0] <= a <= self.a_knots[-1]:
                return self.d2Delta_spl(a)
            else:
                return 0.0

# overrides IVCDM parent methods

    def Omega_cdm(self, a, with_spline=False):
        if with_spline:
            if self.Omega_cdm_spl is None:
                self._get_Omega_cdm_spl()
            return self.Omega_cdm_spl(a) #unitless
        else:
            return self._Omega_cdm(a)

    def _Omega_cdm(self,a):
        def _dOcdm(_a,_n):
            return pow(_a, _n+2)*self.Omega_vac(_a)

        if self.basis == 'A':
            ## Method 1: Exact (slow)
            # I_arr = np.zeros(self.Nq)
            # for n in range(0,self.Nq):
            #     I_arr[n], err = quad(_dOcdm, a, 1.0, args=(n,))
            #     I_arr[n] *= self.qs[n]
            # return self.Oc0/a**3 + (1.0/a**3) * np.sum(I_arr) # don't need times Ol0

            ## Method 2: Approximate soln when |c_n|<<1 for n>0
            sum_poly = polynomial.polyval(a, self.gamma)
            sum_term_over_a3 = (1.0/a**3)*self.sum_gamma - (a**self.qs[0])*sum_poly
            return self.Oc0/a**3 + self.Ol0*sum_term_over_a3
        elif self.basis == 'Chebyshev':
            z = self.a2z(a)
            if self.zmin <= z <= self.zmax:
                return (self.Oc0/a**3) * (1.0+self.Delta(a)) / self.Delta0p1
            else:
                return self.Oc0/a**3
        elif self.basis in ['B', 'C']:
            return (self.Oc0/a**3) * (1.0+self.Delta(a)) / self.Delta0p1
        elif self.basis == 'cubic_spline':
            if self.a_knots[0] <= a <= self.a_knots[-1]:
                return (self.Oc0/a**3)*np.exp(self.Delta(a))
            else:
                return self.Oc0/a**3

    def _get_Omega_cdm_spl(self):
        a_init = 1.0/(1.0+1200.0)
        a0 = 1.02 # since growth solver overshoots a bit
        a_arr = np.linspace(a_init, a0, 100)
        Omc_arr = np.array([self._Omega_cdm(a) for a in a_arr])
        self.Omega_cdm_spl = interp1d(a_arr, Omc_arr, kind='linear')

    def Omega_vac(self,a):
        """
        Notes
        ----
        For Chebyshev basis, for optimization reasons
        we return Ol0 rather than doing the integrals
        which should yield zero anyway.
        """
        qs = self.qs
        if self.basis == 'A':
            aq0 = pow(a, qs[0])
            f1 = polynomial.polyval(a, self.cn_n)
            return self.Ol0 * self.expsumc * aq0 * np.exp(f1)
        elif self.basis == 'Chebyshev':
            z = self.a2z(a)
            if self.zmin <= z <= self.zmax:
                x = cheb2.z2x(z, self.zmin, self.zmax)
                I1, I2, I3 = 3*(np.zeros(self.Nq),)
                for n in range(self.Nq):
                    I1[n] = cheb2.integral(xa=-1.0, xb=x, m=0, n=n)
                    I2[n] = cheb2.integral(xa=-1.0, xb=x, m=1, n=n)
                    I3[n] = cheb2.integral(xa=-1.0, xb=x, m=2, n=n)
                I_arr = (self.zbarp1**2)*I1 + 2.*self.zbarp1*self.Delta_z*I2 + (self.Delta_z**2)*I3
                integral = 3.0 * self.Delta_z * np.sum(qs*I_arr)
                return self.Ol0 + self.Oc0 * (self.Delta0/self.Delta0p1) \
                    - (self.Oc0/a**3) * (self.Delta(a)/self.Delta0p1) \
                    + (self.Oc0/self.Delta0p1) * integral
            else:
                return self.Ol0
        elif self.basis == 'B':
            z = self.a2z(a)
            x = 1.0 + z
            # integral = sum_n c_n I_n = 3 times sum_n c_n [(1+z)^{n+3} - 1] / (n+3)
            integral = 3.0 * (polynomial.polyval(x, self.cn_np3) - self.sum_cn_np3)
            return self.Ol0 + self.Oc0 * (self.Delta0/self.Delta0p1) \
                - (self.Oc0/a**3) * (self.Delta(a)/self.Delta0p1) \
                + (self.Oc0/self.Delta0p1) * integral
        elif self.basis == 'C':
            # integral = sum_n c_n I_n
            integral = \
            - 1.0 * qs[0] * (1.0 - 1.0/a**3) \
            - 1.5 * qs[1] * (1.0 - 1.0/a**2) \
            - 3.0 * qs[2] * (1.0 - 1.0/a) \
            - 3.0 * qs[3] * np.log(a) \
            - 3.0 * (polynomial.polyval(a, self.cn_nm3) - self.sum_cn_nm3)
            return self.Ol0 + self.Oc0 * (self.Delta0/self.Delta0p1) \
                - (self.Oc0/a**3) * (self.Delta(a)/self.Delta0p1) \
                + (self.Oc0/self.Delta0p1) * integral
        elif self.basis == 'cubic_spline':
            if self.Omega_vac_spl is None:
                self._get_Omega_vac_spl()
            if self.a_knots[0] <= a <= self.a_knots[-1]:
                return self.Omega_vac_spl(a)
            else:
                return self.Ol0

    def _get_Omega_vac_spl(self):
        def _dOmega_vac(a,y):
            return -self.Omega_cdm(a)*self.dDelta_spl(a)
        # integrate backwards in time from a=1.
        # The solver starts with t=t_span[0] and integrates until it reachs t_span[-1]
        t = np.linspace(self.a_knots[-1], self.a_knots[0], 300)
        sol = solve_ivp(_dOmega_vac, t_span=(t[0],t[-1]), y0=[self.Ol0], t_eval=t)
        self.Omega_vac_spl = interp1d(sol.t, sol.y[0], kind='linear')

    def Omega_gCg(self,a):
        return self.Omega_cdm(a) + self.Omega_vac(a)

    def _solve_growth(self,y,x):
        """
        y0 = D'; y1 = D; x = scale factor
        with curvature

        dQ_rho := (1/H) d/dlna (Q/rho_m)
        """
        a = x
        E2 = self.E_Hub(a)**2 # (H/H0)^2

        # normalise densities so between 0 and 1
        Omv = self.Omega_vac(a)
        Omv_norm = Omv/E2
        Omc_norm = self.Omega_cdm(a)/E2
        Omm_norm = self.Omega_mat(a)/E2
        Omk_norm = self.Omega_k(a)/E2
        vom = Omv_norm/Omm_norm
        q_hist = self.q_hist(a)

        if self.basis == 'A':
            ncn = [n*self.qs[n] for n in range(self.Nq)] #n * c_n list (first entry 0)
            dq_dlna = polynomial.polyval(x, ncn)
        elif self.basis in ['B', 'C', 'Chebyshev']:
            Omc_bar = self.Oc0/a**3
            Omc_Omv = Omc_bar/Omv
            dOmc_bar = -3.0*Omc_bar/a # d/da \bar\Omega_c
            dOmv = q_hist * (1.0/a) * Omv
            dOmc_Omv = dOmc_bar/Omv - Omc_Omv * (dOmv/Omv)
            dDelta = self.dDelta_da(a)
            ddDelta = self.d2Delta_da2(a)
            dq_dlna = -(a/self.Delta0p1) \
                * (dDelta*Omc_Omv + a*ddDelta*Omc_Omv + a*dDelta*dOmc_Omv)
        elif self.basis == 'cubic_spline':
            Omc = self.Omega_cdm(a)
            Omc_Omv = Omc/Omv
            dOmc_Omv = -(q_hist + Omc_Omv*(3.+q_hist))/a # d(rho_c/rho_X)/da
            dDelta = self.dDelta_da(a)
            ddDelta = self.d2Delta_da2(a)
            dq_dlna = -a*(dDelta*Omc_Omv + a*ddDelta*Omc_Omv + a*dDelta*dOmc_Omv)

        Qtilde = self.Qtilde(a)
        dQ_rho1 = dq_dlna*vom # q term
        dQ_rho2 = -q_hist * vom*(THREEHALF*Omm_norm+Omk_norm) # H term
        dQ_rho3 = q_hist * (Qtilde*(1.0+vom) + 3.0*vom) # V/rho_m term
        dQ_rho = dQ_rho1 + dQ_rho2 + dQ_rho3

        A = (3.0 - THREEHALF*Omm_norm - Omk_norm - Qtilde)/a
        B = -(THREEHALF*Omm_norm + 2.0*Qtilde + dQ_rho)/a**2
        dy0 = -A*y[0] - B*y[1]
        dy1 = y[0]
        return [dy0, dy1]


class FlatIVCDM_smooth(IVCDM_smooth):

    def __init__(self, Om0, Ob0, qs, basis='Chebyshev', zmin=0.0, zmax=2.0, free_knots=None,
                 sigma80=None, H0=67.3, Tcmb=2.7255, Neff=3.046, **kwargs):

        Ol0 = 1.0 - Om0
        IVCDM_smooth.__init__(self, Om0, Ol0, Ob0, qs, basis=basis, zmin=zmin, zmax=zmax, free_knots=free_knots,
                                sigma80=sigma80, H0=H0, Tcmb=Tcmb, Neff=Neff, **kwargs)

    @property
    def Ok0(self):
        return 0.0


class cheb2: # helper class for analytic chebyshev results

    @staticmethod
    def _d_nk(n,k):
        n = int(n)
        k = int(k)
        p = factorial(n-k-1)
        q = factorial(k) * factorial(n-2*k)
        return (-1)**k * pow(2.0, n-2*k) * (n/2.0) * (p/q)

    @staticmethod
    def fun_integral(x,m,n): # integrates x^m T_n(x) and evaluates at x
        m = int(m)
        n = int(n)
        mp1 = m + 1
        if n == 0:
            return x**mp1 / mp1
        elif n > 0:
            K = int(np.floor(n/2.0))
            coeff = np.zeros(n+1)
            for k in range(K+1):
                coeff[n-2*k] = cheb2._d_nk(n,k) / (n - 2*k + mp1)
            return x**mp1 * polynomial.polyval(x, coeff)

    @staticmethod
    def integral(xa,xb,m,n):
        """
        Integrates x^m T_n(x) between xa and xb
        """
        Ia = cheb2.fun_integral(xa,m,n)
        Ib = cheb2.fun_integral(xb,m,n)
        return Ib - Ia

    @staticmethod
    def z2x(z,za,zb): # map z\in [za,zb] onto [-1,1]
        Delta_z = (zb-za)/2.0
        zbar = (zb+za)/2.0
        return (z-zbar)/Delta_z


if __name__ == '__main__':
    pass
