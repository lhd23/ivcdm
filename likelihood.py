import numpy as np
import sys
import os
import glob
import abc

from scipy import linalg,optimize
from scipy.interpolate import interp1d
from scipy.stats import uniform,norm,reciprocal,multivariate_normal
from scipy.special import erf,gamma,gammaincc

# NB gammaincc is the reguarlized upper gamma function it takes values in [0,1]

import ivcdm

try:
    import numdifftools as nd
except:
    pass

thismodule = sys.modules[__name__]

dirname = os.path.dirname(os.path.abspath(__file__))

HALF = 0.5

# DESI fiducial uses Planck 2013 pars
h_PLK13 = 0.6777
H0_PLK13 = 100.*h_PLK13
Ob0_PLK13 = 0.02214/h_PLK13**2
Om0_PLK13 = 0.1414/h_PLK13**2
sigma80_PLK13 = 0.826
DESI_FID = ivcdm.FlatLCDM(H0=H0_PLK13, Om0=Om0_PLK13, Ob0=Ob0_PLK13, sigma80=sigma80_PLK13)

class loglike_gaussian(object):
    _metaclass_ = abc.ABCMeta
    """
    Base abstract class for Gaussian likelihood function.
    Cannot be called. By setting x,y,cov as properties
    they are essentially protected variables that cannot
    be reassigned
    """
    def __init__(self, x=None, y=None, covariance=None):
        self._x = x
        self._y = y
        self._cov = covariance
        self.N = y.size

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def cov(self):
        return self._cov

    def __call__(self,pars):
        pars = np.asarray(pars)
        return self._loglike(pars)

    def m2loglike(self,pars): #chi2 + ln det C
        return -2.0*self.__call__(pars)

    @abc.abstractmethod
    def mu(self,pars): # mu = <y>
        raise NotImplementedError('mu not implemented')

    def total_cov(self,pars):
        # default assumes covariance supplied is total cov
        return self.cov

    def residual(self,pars):
        return self.y-self.mu(pars)

    def _loglike(self,pars):
        """Returns ln(likelihood), where ln is the natural log."""
        chi2,lndet = self.chi_squared(pars)
        if np.isnan(chi2):
            return -1e20
        else:
            return -HALF*chi2 \
                   -HALF*self.N*np.log(2.*np.pi) \
                   -HALF*lndet

    def chi_squared(self,pars):
        """
        The part in the gaussian pdf that we shall call the
        'chi squared' i.e. the argument of the exp times minus two.

        Notes
        -----
        The inversion of the covariance matrix C is done via Cholesky
        decomposition, which is generally faster for large matrices:

            C = LL^T, where L is lower triangle matrix.
 
        Instead of computing C^{-1} the system of eqns Cu = y-mu is
        solved for x = C^{-1} (y-mu). The 'chi squared' part is then
        (y-mu)^T C^{-1} (y-mu) = (y-mu)^T x

        Returns
        -------
        2-tuple (chi2, ln(det(C)))
        """
        res = self.residual(pars)
        if np.any(np.isnan(res)):
            return np.NaN,np.NaN

        cov = self.total_cov(pars)
        try:
            chol_fac = linalg.cho_factor(cov,overwrite_a=True,lower=True)
        except linalg.LinAlgError: # when not positive definite
            return np.NaN,np.NaN
        except ValueError:
            return np.NaN,np.NaN

        try:
            chi2 = np.dot(res, linalg.cho_solve(chol_fac,res))
        except ValueError:
            print 'pars: ', pars
            return np.NaN,np.NaN

        # Determinant of lower tri is given by prod of diag entries
        # det(C) = det(LL^T) = det(L)^2 = prod(diag(L))^2

        lndet = 2.*np.sum(np.log(np.diag(chol_fac[0])))
        return chi2,lndet

    def chi2_per_dof(self,pars):
        """
        The chi squared per degree of freedom where the number of
        degrees of freedom (dof) is the number of data points (N) MINUS
        the number of free parameters to be fitted (Npars).

        Returns
        -------
        chi^2/(N-Npars)
        """
        Npars = pars.size
        chi2,_ = self.chi_squared(pars)
        if np.isnan(chi2):
            raise ValueError('parameters unphysical')
        else:
            return chi2/float(self.N-Npars)

# derivatives of likelihood

    def dmu(self, pars, scale=1e-3, **kwargs):
        """
        Computes 2d array of derivatives for all parameters
        at all data points shape is (num_points, num_params)
        """
        dmu_arr = np.zeros((self.N, pars.size))
        # step_size = scale*pars
        step_size = 1e-4 # for testing only
        for i in range(self.N):
            fmu = lambda x: self.mu(x)[i] #scalar

            # Uncomment one only (returns 1d array):
            # Method 1: Finite forward difference (scipy method; inaccurate; avoid)
            dmu_arr[i,:] = optimize.approx_fprime(pars, fmu, step_size)

            # Method 2: Finite central difference (accurate but requires special library)
            # dmu_arr[i,:] = nd.Gradient(fmu, step=step_size)(pars)
            # dmu_arr[i,:] = nd.Gradient(fmu)(pars)
        return dmu_arr

    def dcov(self, pars, **kwargs):
        # default assumes covariance does not depend on pars
        return np.zeros([pars.size,self.N,self.N])

    def Fisher(self, pars, scale=1e-3, cov_is_diag=False, **kwargs):
        """
        This is the full Fisher matrix needed when the
        covariance matrix depends on parameters

        # C3i := C^-1 C_{,i} C^-1

        Optional parmaeter scale gives step size
        """
        Npars = pars.size
        C = self.total_cov(pars)
        dC = self.dcov(pars, **kwargs)
        if cov_is_diag:
            C_diag = np.diag(C)
            inv_C = np.diag(1./C_diag)
        else:
            inv_C = linalg.inv(C)
        dmu = self.dmu(pars, scale, **kwargs)
        F = np.zeros([Npars,Npars])
        for i in range(Npars):
            for j in range(i,Npars):
                C3i = np.dot(inv_C, np.dot(dC[i], inv_C))
                F[i,j] = np.dot(dmu[:,i], np.dot(inv_C, dmu[:,j])) \
                       + HALF*np.trace(np.dot(C3i, dC[j]))
                F[j,i] = F[i,j] # because Fisher is symmetric
        return F


class cosmolike(loglike_gaussian):

    """
    Extend loglike to include cosmology methods.
    Abstract class cannot be called. Defines number of parameters,
    their names and the cosmological model.

    All parameters are passed into the cosmology methods as keyword
    variables. This means that the parameter array can be in any
    order; however the order of pars must be consistent with
    pars_name as this determines the mapping.
    """
    def __init__(self, x, y, covariance,   # loglike arguments
                 cosmo_name, pars_names, use_omegab=True, **kwargs):  # cosmology arguments
        loglike_gaussian.__init__(self,x,y,covariance)

        # before anything else is done
        self.validate_parameters(cosmo_name, pars_names, self._data_name)
        self.ind = {par: i for i,par in enumerate(pars_names)}
        self.cosmo_name = cosmo_name
        self.pars_names = pars_names
        self.Npars = len(pars_names)
        self._cosmo = None
        self.cosmo_kwargs = kwargs
        self.use_omegab = use_omegab # for CMB

    @property
    def cosmo(self):
        return self._cosmo
    
    @cosmo.setter
    def cosmo(self,obj):
        self._cosmo = obj

    @staticmethod
    def validate_parameters(cosmo_name, pars_names, data_name, pars=None):
        """
        Checks model is consistent with parameter names supplied
        """
        # check the cosmology is available or spelled correctly
        if cosmo_name not in ['LCDM', 'FlatLCDM', 'IVCDM', 'FlatIVCDM',
                              'IVCDM_binned', 'FlatIVCDM_binned',
                              'IVCDM_smooth', 'FlatIVCDM_smooth']:
            raise ValueError('Cosmology {} not found'.format(cosmo_name))

        # check q is specified for interacting model
        if 'q' not in pars_names and cosmo_name in ['IVCDM', 'FlatIVCDM']:
            raise ValueError('Parameter q not found')
        else:
            pass

        # check Ol0 is specified for non-flat models
        if 'Ol0' not in pars_names and cosmo_name in ['LCDM', 'IVCDM']:
            raise ValueError('Parameter Ol0 not found')
        else:
            pass

        # check data constrainable parameters specified
        if 'sigma80' not in pars_names and data_name is 'RSD':
            raise ValueError('RSD data: sigma80 not found')

        if 'Ob0' not in pars_names and data_name is 'BAO':
            raise ValueError('BAO data: Ob0 not found')

        # check there is same number of names and values
        if pars is not None:
            if len(pars) != len(pars_names):
                raise ValueError('Inconsistent parameters/names')
            else:
                pass

    def get_qs(self,pars):
        X = zip(self.pars_names,pars) # preserve order
        return np.array([val for name,val in X if name.startswith('q')]) #q1,q2,...

    def get_dict(self,pars): #get parameter dictionary
        X = zip(self.pars_names,pars) # preserve order
        d = dict(X)
        #@! temporary hack
        if self.cosmo_name in ['IVCDM_binned', 'FlatIVCDM_binned',
                               'IVCDM_smooth', 'FlatIVCDM_smooth']:
            d.update({'qs': self.get_qs(pars)})
        return d

    def get_model(self,pars):
        """
        Selects the cosmological model (LCDM, FlatIVCDM etc)
        and instantiates it with the given cosmological parameters;
        returns the instance.

        This method interfaces with your cosmology code. Here we
        use ivcdm but it can be used with astropy. Just need to
        pass the parameters in the into the object call cosmo_model.
        """
        cosmo_dict = {}
        cosmo_dict.update(self.get_dict(pars))
        cosmo_dict.update(self.cosmo_kwargs)
        cosmo_model = getattr(ivcdm, self.cosmo_name) # e.g. ivcdm.FlatLCDM
        return cosmo_model(**cosmo_dict)


class cosmo_CC(cosmolike):

    _data_name = 'CC' # cosmic chronometers

    def mu(self,pars): # concrete class
        self.cosmo = self.get_model(pars)
        return np.array([self.cosmo.Hub(z) for z in self.x])

class cosmo_RSD(cosmolike):

    _data_name = 'RSD'

    def mu(self,pars): # concrete class
        self.cosmo = self.get_model(pars)
        return np.array([self.cosmo.fsigma_8(z) for z in self.x])


class cosmo_CMB(cosmolike):

    _data_name = 'CMB'

    def mu(self,pars): # concrete class
        self.cosmo = self.get_model(pars)
        if self.use_omegab:
            return np.array([self.cosmo.R_shift, self.cosmo.lA, self.cosmo.wb])
        else:
            return np.array([self.cosmo.R_shift, self.cosmo.lA])


class cosmo_BAO(cosmolike):

    _data_name = 'BAO'

    def mu(self,pars): # concrete class
        self.cosmo = self.get_model(pars)
        return np.array([self.cosmo.d_z(z) for z in self.x])


class cosmo_BAO_DESI_forecast(cosmolike):

    _data_name = 'BAO (DESI forecast)'

    def mu(self,pars): # concrete class
        self.cosmo = self.get_model(pars)
        z_unique = self.x[::2] # since every redshift repeats
        s = self.cosmo.dH * np.array([self.cosmo.r_s_comoving(z) for z in z_unique])
        Hs = np.array([self.cosmo.Hub(z) for z in z_unique]) * s
        dA_s = self.cosmo.dH * np.array([self.cosmo.dA(z) for z in z_unique]) / s
        y = np.empty(2*Hs.size)
        y[0::2] = Hs
        y[1::2] = dA_s
        return y


class cosmo_SNIa_conventional(cosmolike):

    _data_name = 'SNIa (conventional)'

    def mu(self, pars):
        self.cosmo = self.get_model(pars)
        return np.array([(self.cosmo.dist_mod(z)-19.3) for z in self.x]) # mu+M

    def dL(self, pars, N=100):
        self.z_arr = np.linspace(np.amin(self.x)-1e-8, np.amax(self.x), N)
        self.cosmo = self.get_model(pars)
        return np.array([self.cosmo.dL(z) for z in self.z_arr])

    def ddistmod(self, pars, scale=1e-3):
        """
        Derivatives of distance modulus computed for a uniform
        distribution of 200 redshifts between 0 and max(self.x)
        for each parameter. Returns 2d array.
        This function should be used for forecasting or when there are
        too many points that numerical derivatives takes too long.

        Method 1: Finite forward difference (scipy method; inaccurate; avoid)
        Method 2: Finite central difference (accurate but requires special library)
        """
        dL_arr = self.dL(pars)
        N = dL_arr.size
        ddL_arr = np.zeros((N, pars.size))
        ddistmod_arr = np.zeros((N, pars.size))
        step_size = scale*pars
        # step_size = 1e-4 # for testing only
        for i in range(N):
            fdL = lambda x: self.dL(x)[i] #scalar
            ddL_arr[i,:] = optimize.approx_fprime(pars, fdL, step_size) # Method 1
            # ddL_arr[i,:] = nd.Gradient(fdL, step=step_size)(pars) # Method 2
            # ddL_arr[i,:] = nd.Gradient(fdL)(pars)
        for i in range(pars.size):
            ddistmod_arr[:,i] = 5./np.log(10.)*(ddL_arr[:,i]/dL_arr)
        return ddistmod_arr

    def dmu_2pars(self, pars, scale=1e-3):
        dmu_arr = np.zeros((self.N, 2))
        ddistmod_arr = self.ddistmod(pars, scale)
        if self.cosmo_name in ['FlatIVCDM', 'IVCDM']:
            Om0_ind = self.ind['Om0']
            q_ind = self.ind['q']
            dmu_dOm0_spl = interp1d(self.z_arr, ddistmod_arr[:,Om0_ind])
            dmu_dq_spl = interp1d(self.z_arr, ddistmod_arr[:,q_ind])
            dmu_arr[:,0] = dmu_dOm0_spl(self.x)
            dmu_arr[:,1] = dmu_dq_spl(self.x)
            return dmu_arr
        else:
            raise ValueError('model {} cannot be used'.format(self.cosmo_name))

    def Fisher_2pars(self, pars, scale=1e-3, cov_is_diag=False, **kwargs):
        Npars = 2
        C = self.total_cov(pars)
        if cov_is_diag:
            C_diag = np.diag(C)
            inv_C = np.diag(1./C_diag)
        else:
            inv_C = linalg.inv(C)
        dmu = self.dmu_2pars(pars, scale)
        F = np.zeros([Npars,Npars])
        for i in range(Npars):
            for j in range(i,Npars):
                F[i,j] = np.dot(dmu[:,i], np.dot(inv_C, dmu[:,j]))
                F[j,i] = F[i,j] # because Fisher is symmetric
        return F

    def dmu(self, pars, scale=1e-3):
        dmu_arr = np.zeros((self.N, pars.size))
        ddistmod_arr = self.ddistmod(pars, scale)
        if self.cosmo_name in ['FlatIVCDM', 'IVCDM']:
            spl = []
            for i in range(pars.size):
                spl = interp1d(self.z_arr, ddistmod_arr[:,i])
                dmu_arr[:,i] = spl(self.x)
            return dmu_arr
        else:
            raise ValueError('model {} cannot be used'.format(self.cosmo_name))

    def Fisher(self, pars, scale=1e-3, cov_is_diag=False, **kwargs):
        Npars = pars.size
        C = self.total_cov(pars)
        if cov_is_diag:
            C_diag = np.diag(C)
            inv_C = np.diag(1./C_diag)
        else:
            inv_C = linalg.inv(C)
        dmu = self.dmu(pars, scale)
        F = np.zeros([Npars,Npars])
        for i in range(Npars):
            for j in range(i,Npars):
                F[i,j] = np.dot(dmu[:,i], np.dot(inv_C, dmu[:,j]))
                F[j,i] = F[i,j] # because Fisher is symmetric
        return F


class cosmo_SNIa(cosmolike):

    _data_name = 'SNIa'

    def mu(self,pars):
        # note covariance is updated here
        self.cosmo = self.get_model(pars)
        return self._mu(pars)

    def _mu(self,pars):
        """
        BHM mu vector predicted by cosmological model
        Returns the mean 1d array (length = 3*Nsnia) appearing in
        the residual y-mu:

            mu := <x> = {dm_1,0,0,dm_2,0,0,...,dm_N,0,0} + Y_0 * A

        dm_i is the ith distance modulus (m-M).

        Notes
        -----
        mu is NOT the distance modulus although this is calculated
        in this routine.
        'distmod' is the distance modulus (m-M) := 5 * log10(dL/Mpc) + 25.
        """
        dm = [self.cosmo.dist_mod(z) for z in self.x]

        # Insert zeros for every 3n-1 th and 3nth entry
        Nsnia = self.x.size
        mu_3N = np.hstack([dm[i],0.,0.] for i in range(Nsnia))
        Y0A_ = self.Y0A(**self.get_dict(pars))
        Y0A_3N = np.array(Nsnia*Y0A_)
        return mu_3N + Y0A_3N # shape=(3*Nsnia,)

    @staticmethod
    def Y0A(A=None, X0=None, B=None, C0=None, M0=None, **kwargs):
        return [M0-A*X0+B*C0, X0, C0] # length=3

    @staticmethod
    def _A3(A=None, B=None, **kwargs):
        A3 = np.identity(3)
        A3[1,0] = -A
        A3[2,0] = +B
        return A3 #3x3

    @staticmethod
    def _COVl3(VM=None, VX=None, VC=None, **kwargs):
        return np.diag([VM, VX, VC]) #3x3

    def _ATCOVlA3(self, A=None, B=None, VM=None,
                  VX=None, VC=None, **kwargs):
        A3 = self._A3(A,B)
        COVl3 = self._COVl3(VM,VX,VC)
        return np.dot(A3.T, np.dot(COVl3, A3)) #3x3

    # replaces generic method
    def total_cov(self,pars):
        Nsnia = self.x.size
        ATCOVlA3 = self._ATCOVlA3(**self.get_dict(pars))
        Xl = Nsnia*[ATCOVlA3]
        ATCOVlA = linalg.block_diag(*Xl)
        return self.cov + ATCOVlA # self.cov := Sigma_d

# auxiliary derivatives methods for SNIa Fisher

    def dmu(self, pars, scale=1e-3, fix_SN_nuisance=False, **kwargs):
        """
        Construct the 2d-array

            d/dtheta_j mu_i := mu_{i,j}

        where theta_j is the jth parameter and mu_i is the ith
        entry corresponding to the ith redshift 
        """
        Nsnia = self.x.size
        dmu_ = np.zeros([self.N,self.Npars]) #self.N=3*740=2220

        pars_names_nuisance = ['A', 'X0', 'VX', 'B', 'C0', 'VC', 'M0', 'VM']
        pars_names_cosmo = [s for s in self.pars_names if s not in pars_names_nuisance]

        # get the cosmological parameters only and keep their order
        pars_cosmo_only = [val for par,val in zip(self.pars_names,pars) \
                           if par in pars_names_cosmo]
        pars_cosmo_only = np.asarray(pars_cosmo_only)

        # Get derivatives wrt cosmological parameters
        # step_size = scale*pars_cosmo_only #only works if param is nonzero!
        step_size = 1e-3 # for testing only
        for j,z in enumerate(self.x):
            # dist mod as a function of pars (fixed z)
            fd = lambda x: self.get_model(x).dist_mod(z)

            # Uncomment one only (returns 1d array):
            # Method 1: Finite forward difference (scipy method; inaccurate; avoid)
            # grad_distmod = optimize.approx_fprime(pars_cosmo_only, fd, step_size)

            # Method 2: Finite central difference (accurate but requires special library)
            grad_distmod = nd.Gradient(fd, step=step_size)(pars_cosmo_only)

            for df,par in zip(grad_distmod, pars_names_cosmo):
                dmu_[3*j,self.ind[par]] = df

        # derivatives wrt nuisance parameters
        if fix_SN_nuisance:
            for par in pars_names_nuisance:
                dmu_[:,self.ind[par]] = np.array(Nsnia*[0.,0.,0.])
        else: # default
            p = self.get_dict(pars)
            d = {
                 'A':   [-p['X0'],0.,0.],
                 'X0':  [-p['A'], 1.,0.],
                 'B':   [+p['C0'],0.,0.],
                 'C0':  [+p['B'], 0.,1.],
                 'M0':  [1.,0.,0.],
                 'VM':  [0.,0.,0.],
                 'VX':  [0.,0.,0.],
                 'VC':  [0.,0.,0.]
                }

            for par,val in d.iteritems():
                dmu_[:,self.ind[par]] = np.array(Nsnia*val)

        return dmu_ # (3*Nsnia,Npars)

    def _dA(self): # derivative of _A3 for each par
        dA_ = np.zeros((self.Npars,3,3))
        dA_[self.ind['A'],1,0] = -1. #alpha
        dA_[self.ind['B'],2,0] = +1. #beta
        return dA_

    def _dCOVl(self):
        dCOVl_ = np.zeros((self.Npars,3,3))
        dCOVl_[self.ind['VM'],0,0] = 1.
        dCOVl_[self.ind['VX'],1,1] = 1.
        dCOVl_[self.ind['VC'],2,2] = 1.
        return dCOVl_

    # replaces generic method
    def dcov(self, pars, fix_SN_nuisance=False, **kwargs):
        """
        3d array of derivatives of total covariance matrix
        Since computed for each parameter array 
        is of shape (Npars, 3*Nsnia, 3*Nsnia)

        dC only depends on nuisance parameters so if
        we fix nuisance then derivs are zero.
        """
        # shorthands
        Npars = pars.size
        Nsnia = self.x.size
        dC = np.zeros([Npars, 3*Nsnia, 3*Nsnia])

        if fix_SN_nuisance:
            return dC
        else:
            # 3x3 arrays
            A3 = self._A3(**self.get_dict(pars))
            COVl3 = self._COVl3(**self.get_dict(pars))

            # table of derivatives (Npars x 3x3 arrays)
            dA = self._dA()
            dCOVl = self._dCOVl()

            for i in range(Npars):
                dblock3 = np.dot(dA[i].T, np.dot(COVl3, A3)) \
                          + np.dot(A3.T, np.dot(dCOVl[i], A3)) \
                          + np.dot(A3.T, np.dot(COVl3, dA[i])) # 3x3
                assert dblock3.shape == (3,3)
                list_dblock3 = Nsnia*[dblock3]
                dC[i] = linalg.block_diag(*list_dblock3)
            return dC


class joint_loglike:

    def __init__(self, cosmo_name, pars_names, data=(None,), **kwargs):
        self.data = data # list of strings 'RSD', 'BAO' etc
        self.loglike_dict = {}
        for name in self.data:
            llike = getattr(thismodule, 'cosmo_{}'.format(name))
            self.loglike_dict[name] = llike(*get_data(name, **kwargs), \
                                            cosmo_name=cosmo_name, \
                                            pars_names=pars_names, \
                                            **kwargs
                                            )

    def __call__(self,pars):
        joint_ll = 0.0
        for name in self.data:
            joint_ll += self.loglike_dict[name](pars)
        return joint_ll

    def m2loglike(self,pars):
        return -2.0*self.__call__(pars)

    def Fisher(self, pars, scale=1e-3, fix_SN_nuisance=False):
        """
        The combined Fisher matrix is gotten by summing
        over individual Fisher matrices of each experiment
        """
        F = np.zeros([pars.size,pars.size])
        for name in self.data:
            F += self.loglike_dict[name].Fisher(pars, scale, fix_SN_nuisance=fix_SN_nuisance)
        return F


def joint_logprior(Om0=None, Ob0=None, Ol0=None, q=None, sigma80=None,
                   A=None, X0=None, B=None, C0=None,
                   M0=None, VX=None, VC=None, VM=None,
                   Om_a=0.01, Om_b=0.99,
                   Ob_a=0.001, Ob_b=0.4,
                   Ol_a=0.01, Ol_b=0.99,
                   q_a=-1.5, q_b=4.5,
                   Nq=None, zbins=None, Nscale=4, bin_type='zunif',
                   xi0_lims=[5e-2, 2.0], xi0=0.1, **kwargs):

    lp = 0.0

    # Cosmological parameters

    if Om0 is not None:
        lp += uniform.logpdf(Om0, loc=Om_a, scale=Om_b-Om_a)
    if Ob0 is not None:
        lp += uniform.logpdf(Ob0, loc=Ob_a, scale=Ob_b-Ob_a)
    if Ol0 is not None:
        lp += uniform.logpdf(Ol0, loc=Ol_a, scale=Ol_b-Ol_a)
    if sigma80 is not None:
        lp += reciprocal.logpdf(sigma80, 1e-5, 1e+02)

    # Nuisance parameters

    if A is not None:
        lp += uniform.logpdf(A, loc=0.0, scale=1.0)
    if B is not None:
        lp += uniform.logpdf(B, loc=0.0, scale=4.0)
    if X0 is not None:
        lp += norm.logpdf(X0, loc=0.0, scale=10.0)
    if C0 is not None:
        lp += norm.logpdf(C0, loc=0.0, scale=1.0)
    if M0 is not None:
        lp += norm.logpdf(M0, loc=-19.3, scale=2.0)
    if VX is not None:
        lp += reciprocal.logpdf(VX, 1e-10, 1e+04)
    if VC is not None:
        lp += reciprocal.logpdf(VC, 1e-10, 1e+04)
    if VM is not None:
        lp += reciprocal.logpdf(VM, 1e-10, 1e+04)

    if q is not None: # one q par only
        lp += uniform.logpdf(q, loc=q_a, scale=q_b-q_a)
        # lp += norm.logpdf(q, loc=0.0, scale=0.3)

    # Multiple correlated q pars from multivariate gaussian

    if Nq is not None: # means multiple q pars
        if Nq == 1:
            lp += uniform.logpdf(kwargs['q1'], loc=q_a, scale=q_b-q_a)
            return lp

        qs = np.array([kwargs['q{}'.format(i+1)] for i in range(Nq)]) # extract q pars

        # Pick one:
        qmeans = smooth(qs, Nw=5) # fiducial
        # qmeans = 0.0*qs

        # Option 1: Hyperparameter dependent prior (unmarginalised)
        # cov = cov_q(Nq, zbins, Nscale=Nscale, bin_type=bin_type, xi0=xi0_lims[0])
        # lp += multivariate_normal.logpdf(qs, mean=qmeans, cov=cov)

        # Option 2: Hyperparameter-marginalised prior
        lp += np.log(prior_q(qs, zbins, Nscale=Nscale, bin_type=bin_type, xi0_lims=xi0_lims, qmeans=qmeans))

    return lp

def tilde_cov_q(Nq, zbins, Nscale=4, bin_type='zunif'):
    if bin_type == 'aunif': # uniform in a
        xbins = 1./(1.+zbins[::-1])
    elif bin_type == 'zunif': # uniform in z
        xbins = zbins

    deltax = xbins[1] - xbins[0]
    # check xbins is uniformly spaced
    assert np.isclose(deltax, xbins[-1]-xbins[-2])

    xs = Nscale * deltax
    xtilde = np.zeros((Nq+1, Nq+1))
    for i in range(Nq+1):
        for j in range(i, Nq+1):
            xtilde[i,j] = np.abs(xbins[i]-xbins[j])
            xtilde[j,i] = xtilde[i,j]

    xbar = xtilde/xs
    xplus = (xtilde + deltax)/xs
    xminus = (xtilde - deltax)/xs

    # below is only valid if uniform width
    tilde_C = \
              + xminus * np.arctan(xminus) \
              + xplus * np.arctan(xplus) \
              - 2.*xbar * np.arctan(xbar) \
              + np.log(1.+xbar**2) \
              - 0.5*np.log(1.+xplus**2) \
              - 0.5*np.log(1.+xminus**2)
    tilde_C *= (xs/deltax)**2
    return tilde_C[1:,1:]

def cov_q(Nq, zbins, Nscale=4, bin_type='zunif', xi0=0.1, **kwargs):
    return xi0 * tilde_cov_q(Nq, zbins, Nscale, bin_type)

def prior_q(qs, zbins, Nscale=4, bin_type='zunif', xi0_lims=[5e-2, 2.0], qmeans=0.0):
    """
    Compute the normalised pdf for the q parameters.

    Note
    ----
    The difference term ~ [gamma/x] becomes numerically
    unstable as x tends to zero. In that case we just
    return the well-defined limit x -> 0

    """
    Nq = qs.size
    Nq_2 = Nq/2.0
    xi_a = xi0_lims[0]
    xi_b = xi0_lims[1]

    Ctilde = tilde_cov_q(Nq, zbins, Nscale, bin_type)
    Ctilde_inv = linalg.inv(Ctilde)
    dq = qs - qmeans
    tilde_chi2 = np.dot(dq, np.dot(Ctilde_inv, dq))
    det = linalg.det(2.*np.pi*Ctilde)

    if tilde_chi2 > 1e-3:
        inc_gam_a = gammaincc(Nq_2, tilde_chi2/(2.0*xi_a)) * gamma(Nq_2)
        inc_gam_b = gammaincc(Nq_2, tilde_chi2/(2.0*xi_b)) * gamma(Nq_2)
        gamma_x = 1.0/pow(tilde_chi2/2.0, Nq_2) * (inc_gam_b - inc_gam_a)
    else: # return the qs -> 0 limit
        gamma_x = 1./(Nq_2) * (1./pow(xi_a, Nq_2) - 1./pow(xi_b, Nq_2))

    return 1.0/np.log(xi_b/xi_a) * 1.0/np.sqrt(det) * gamma_x

def get_data(data_name='', **kwargs): # returns eg get_SNIa_data()
    return globals()['get_{}_data'.format(data_name)](**kwargs)

def get_CC_data(**kwargs): # cosmic chronometers
    ndtypes = [('z',float),('H',float),('err',float)]
    data = np.genfromtxt(dirname+'/data/cosmic_chronometers.dat',dtype=ndtypes, skip_header=1)
    z = data['z']
    H = data['H']
    err = data['err']
    covariance = np.diag(err**2)
    return z,H,covariance # np.diag(np.random.normal(loc=2.0, scale=1.0, size=z.size)**2)

def get_RSD_data(get_survey=False, **kwargs):
    ndtypes = [('zeff',float),('fsigma8',float),('err',float),('survey','S20'),('year','S4')]
    data = np.genfromtxt(dirname+'/data/RSD.csv',delimiter=',',dtype=ndtypes, skip_header=1)
    survey = data['survey']
    zeff = data['zeff'] # effective redshift
    fsigma8 = data['fsigma8']
    err = data['err']
    covariance = np.diag(err**2)
    if get_survey:
        return zeff,fsigma8,covariance,survey
    else:
        return zeff,fsigma8,covariance

def get_CMB_data(use_omegab=True, **kwargs):
    """
    Data vector is y = (R, l_A, omega_b), where 

        R:               shift parameter (Efstathiou & Bond 1999)
        l_A=pi/theta_*:  angular position of first acoustic peak 
        omega_b:         physical baryon density

    where theta_* is the angular size at last scattering.
    The covariance matrix is given by

        C_ij = sigma_i * sigma_j * D_ij

    where D is the correlation matrix, sigma_i the errors of y
    """
    data = np.loadtxt(dirname+'/data/clik.dat', dtype=[('y', float), ('err', float)])
    D_corr = np.loadtxt(dirname+'/data/clik_corr.dat') # correlation matrix
    sig = data['err']
    covariance = np.outer(sig,sig)*D_corr # outer product
    if use_omegab:
        return None, data['y'], covariance
    else:
        return None, data['y'][:2], covariance[:2,:2]

def get_BAO_data(**kwargs):
    ndtypes = [('zeff',float),('d_z',float),('err_d_z',float),('survey','S20'),('year','S4')]
    data = np.genfromtxt(dirname+'/data/BAO.dat', dtype=ndtypes, skip_header=1)
    survey = data['survey']
    zeff = data['zeff'] # effective redshift
    d_z = data['d_z']
    err = data['err_d_z']
    err = err[~np.isnan(err)] # remove wiggle z nans
    inv_cov_wigz = np.loadtxt(dirname+'/data/wigglez_inv_cov.dat')
    cov_diag = np.diag(err**2)
    cov_wigz = linalg.inv(inv_cov_wigz)
    covariance = linalg.block_diag(cov_diag,cov_wigz)
    return zeff,d_z,covariance#,survey

def eta_cov():
    """
    NB. C_eta for us will include the covariance matrices
    C_coh, C_lens, C_z. While these three matrices are
    really NxN we enlarge to 3Nx3N filling in with zeros
    in order to conform with C_eta (3Nx3N) as it appears
    the literature
    """
    Ceta = np.zeros((2220,2220))
    for Cname in glob.glob(dirname+'/data/SNIa/covmat/fits/C*.npy'):
        mat = np.load(Cname)
        assert mat.shape == (2220,2220)
        Ceta += mat
    assert Ceta.shape == (2220,2220)
    return Ceta
    # return sum([np.load(m) for m in glob.glob(dirname+'/data/SNIa/covmat/fits/C*.npy')]) #3Nx3N

def get_SNIa_data(dset=None, method=1, **kwargs):
    """Returns the data vector and the covariance matrix.
    The JLA covariance matrix ('cov') consists of the systematic
    matrix and several other statistical covariance matrices.
    See eqn 11 in Betoule et al 2014.

    """
    if dset == None: #default to JLA
        ndtypes = [('SNIa','S12'), \
                   ('zcmb',float), \
                   ('zhel',float), \
                   ('e_z',float), \
                   ('mb',float), \
                   ('e_mb',float), \
                   ('x1',float), \
                   ('e_x1',float), \
                   ('c',float), \
                   ('e_c',float), \
                   ('logMst',float), \
                   ('e_logMst',float), \
                   ('tmax',float), \
                   ('e_tmax',float), \
                   ('cov(mb,s)',float), \
                   ('cov(mb,c)',float), \
                   ('cov(s,c)',float), \
                   ('set',int), \
                   ('RAdeg',float), \
                   ('DEdeg',float), \
                   ('bias',float)]
        # width of each column
        delim = (12, 9, 9, 1, 10, 9, 10, 9, 10, 9, 10, 10, 13, 9, 10, 10, 10, 1, 11, 11, 10)
        # load the data
        data = np.genfromtxt(dirname+'/data/SNIa/tablef3.dat', delimiter=delim, dtype=ndtypes, autostrip=True)
        # data = np.loadtxt(dirname+'/data/SNIa/JLA.tsv')
        if method == 1:
            cov = np.load(dirname+'/data/SNIa/covmat/stat.npy')
            for i in ['cal', 'model', 'bias', 'dust', 'sigmaz', 'sigmalens', 'nonia']:#, 'host', 'pecvel']:
                cov += np.load(dirname+'/data/SNIa/covmat/'+i+'.npy')
        elif method == 2:
            cov = np.load(dirname+'/data/SNIa/covmat/fits/C_stat.npy')
            for i in ['cal', 'model', 'bias', 'host', 'dust', 'pecvel','nonia', 'coh', 'lens', 'z']:
                cov += np.load(dirname+'/data/SNIa/covmat/fits/C_'+i+'.npy')
            # cov = eta_cov()
        z = data['zcmb'] # redshifts measured in CMB frame
        tri = np.column_stack([data['mb'],data['x1'],data['c']]) # {(mb,x1,c)}
        Nsnia = z.shape[0] # Number of SNIa
        D = np.hstack(tri[i] for i in range(Nsnia)) # shape=(3*Nsnia,) 1d array!
        return z,D,cov
    elif dset == 'Pantheon':
        with open(dirname+'/data/SNIa/pantheon/lcparam_full_long.txt') as f:
            names = f.readline()
        names = names[1:].split() # delete hash then split string up at whitespace
        names = names[:-1] # remove last name (more names than columns for some reason)
        d = np.genfromtxt(dirname+'/data/SNIa/pantheon/lcparam_full_long.txt', names=names, skip_header=1)
        z = d['zcmb'].astype(float)
        mu_plus_M = d['mb'].astype(float)
        err_mu_plus_M = d['dmb'].astype(float)
        cov = np.diag(err_mu_plus_M**2)
        return z,mu_plus_M,cov
    else:
        raise ValueError('SN dataset may be JLA or Pantheon only')


def get_DESI_RSD_forecast():
    """
    The fiducial cosmology used by DESI is given in table 2.2 in
    arxiv:1611.00036, which is in turn taken from Planck 2013
    [table 5, column 5 (Planck+WP+highL+BAO)]
    Here we fix the fiducial cosmology with sigma_80=0.815.
    This does not matter if we fix the parameter sigma_80 as
    the Fisher matrix of all other parameters will be independent
    of it. We only need to fix it to calculate fsigma_{80}.
    """
    data = np.loadtxt(dirname+'/data/DESI-RSD.dat', skiprows=1)
    z = data[:,0]

    # need to remove the dependence DESI fiducial cosmology
    fs8 = np.array([DESI_FID.fsigma_8(zi) for zi in z])
    sig_fs8 = fs8 * (data[:,1]/100.) # sigma_{fsigma_8}
    covariance = np.diag(sig_fs8)

    # any y will do here
    return z, np.zeros_like(z), covariance

def get_DESI_BAO_forecast():
    data = np.loadtxt(dirname+'/data/DESI-BAO.dat', skiprows=1)
    z = data[:,0]
    N = z.size

    # need to remove the dependence DESI fiducial cosmology
    s = DESI_FID.dH * np.array([DESI_FID.r_s_comoving(zi) for zi in z])
    Hs = np.array([DESI_FID.Hub(zi) for zi in z]) * s
    dA_s = DESI_FID.dH * np.array([DESI_FID.dA(zi) for zi in z]) / s
    sig_Hs = Hs * (data[:,1]/100.)
    sig_dA_s = dA_s * (data[:,2]/100.)

    # construct array of alternating Hs and dA_s errors
    sig = np.empty(2*N)
    z2 = np.empty_like(sig)
    sig[0::2] = sig_Hs
    sig[1::2] = sig_dA_s
    z2[0::2] = z
    z2[1::2] = z

    corr_coeff = 0.4
    corr_block = [[1.0, corr_coeff],[corr_coeff,1.0]]
    corr_mat = linalg.block_diag(*N*[corr_block])
    covariance = np.outer(sig,sig)*corr_mat
    return z2, np.zeros_like(z2), covariance

def get_LSST_z(NSNe=5e4, z0=0.0429, alpha=0.7):
    from scipy.special import gammaincinv # returns x such that gammainc(a, x) = y
    u = np.random.rand(int(NSNe))
    X = gammaincinv(3./alpha, u) # X:=(z/z0)^alpha
    return z0*pow(X, 1./alpha)

def get_SNIa_simple_cov(z, sigma_int=0.12, sigma_z0=0.05):
    sigma_z = sigma_z0*(1.0+z)
    sigma2 = sigma_int**2 + (5.*sigma_z/(z*np.log(10)))**2
    return np.diag(sigma2)

def get_LSST_SNIa_forecast(NSNe=5e4, sigma2_int=0.12, sigma_z0=0.05, z0=0.0429, alpha=0.7):
    z = get_LSST_z(NSNe, z0, alpha)
    covariance = get_SNIa_simple_cov(z, sigma2_int, sigma_z0)
    # any y will do here
    return z, np.zeros_like(z), covariance

def running_mean(x, Nw):
    """
    This method produces exactly the same output
    as np.convolve(x, np.ones(Nw)/Nw, 'valid') but is
    faster because it does not use FFT. All it is doing
    is taking the average but in an efficient way.

    E.g. for Nw=5 

        (x_{i-2} + x_{i-1} + x_i + x_{i+1} + x_{i+2})/5

    Returns array of length x.size - (Nw-1)
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[Nw:]-cumsum[:-Nw])/float(Nw)

def smooth(x, Nw):
    """
    Smooths a 1d array of size M and returns an
    array of the same size M.
    Interior points we use running_mean which
    works as we expect (arithmetic average). The
    Boundaries need to be handled separately and
    we take a weighted mean:

    The endpoints are computed as e.g. for
    Nw=5 (window size, must be odd):

        s[0] = x[0]
        s[1] = (x[0] + x[1] + x[2])/3
        s[2] = (x[0] + x[1] + x[2] + x[3] + x[4])/5
        s[3] = (x[1] + x[2] + x[3] + x[4] + x[5])/5 etc

    """
    assert Nw % 2 != 0 # check Nw is odd
    mid = running_mean(x, Nw)
    r = np.arange(1, Nw-1, 2)
    start = np.cumsum(x[:Nw-1])[::2]/r
    stop = (np.cumsum(x[:-Nw:-1])[::2]/r)[::-1]
    return np.concatenate((start,mid,stop))

def matrix_running_average(N, Nw=5):
    A = np.zeros((N,N))

    # boundaries
    num = Nw//2 + 1 # number of rows associated with bdy
    inds = [1 + 2*i for i in range(num)]
    weights = 1./np.array(inds)
    for i,(ind,w) in enumerate(zip(inds,weights)):
        A[i,:ind] = w
        A[-(i+1),-ind:] = w

    # interior
    for i,j in zip(range(num, N-num), range(N-2*num)):
        A[i,(1+j):(Nw+1+j)] = 1.0/Nw

    return A

def test_matrix_running_average():
    N = 20
    Nw = 5
    A = np.zeros((N,N))

    A[0,0] = 1.0
    A[-1,-1] = 1.0
    A[1,:3] = 1.0/3
    A[-2,-3:] = 1.0/3
    A[2,:5] = 1.0/5
    A[-3,-5:] = 1.0/5
    for i,j in zip(range(3, N-3), range(N-6)):
        A[i,(1+j):(6+j)] = 1.0/5

    p0 = np.random.randn(N)
    L1 = np.allclose(np.dot(A,p0), smooth(p0, Nw=5))
    L2 = np.allclose(matrix_running_average(N, Nw), A)

    if L1 and L2:
        print 'Test matrix_running_average: success'
    else:
        print 'Test matrix_running_average: failed'

def main():
    pass

if __name__ == '__main__':
    main()
