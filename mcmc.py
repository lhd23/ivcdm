from __future__ import division
import os
import sys
import argparse

import numpy as np
import emcee
from emcee.utils import MPIPool
# from schwimmbad import MPIPool

from likelihood import *
from models import cosmo


dirname = os.path.dirname(os.path.abspath(__file__))

# Notes: the name 'q' on its own always refers to (Flat)IVCDM(_H0) models
# and not the binned or smooth models. For those they are numbered q1,q2, etc


# Names of parameter models e.g. the only difference between LCDM
# LCDM_H0 is that H0 is fixed in one and not in the other.

multi_q_models = ['FlatIVCDM_binned', 'IVCDM_binned',
                  'FlatIVCDM_smooth', 'IVCDM_smooth',
                  'FlatIVCDM_H0_binned', 'IVCDM_H0_binned']

valid_models = ['FlatLCDM', 'LCDM', 'FlatIVCDM', 'IVCDM',
                'FlatLCDM_H0', 'LCDM_H0', 'FlatIVCDM_H0', 'IVCDM_H0'] + multi_q_models


def logpost(pars, pars_names, prior_only=False, **kwargs):
    lprior = logprior(pars, pars_names, **kwargs)
    if not np.isfinite(lprior):
        return -np.inf
    else:
        if prior_only:
            return lprior
        else:
            return loglike(pars) + lprior

def loglike(pars):
    return loglike_joint(pars)

def logprior(pars, pars_names, **kwargs):
    # [precompute q covariance matrix]
    pars_dict = dict(zip(pars_names,pars))
    for kw in ['Nq', 'zbins', 'Nscale', 'bin_type', 'xi0_lims', 'xi0']:
        pars_dict.update({kw: kwargs[kw]})
    return joint_logprior(**pars_dict)

def get_p0(pars, nwalkers, amp=0.1,
            Nq=None, zbins=None, Nscale=4,
                bin_type='aunif', xi0_lims=None, q_ind=None, **kwargs):
    """
    Compute Gaussian ball around centre point 'pars' with
    width 'amp'. If mean is zero then add a perturbation
    of amplitude amp

    Returns
    -------
     2d array of shape (nwalkers,ndim)
    """
    ndim = pars.size
    p0 = np.empty((nwalkers,ndim))

    for j in range(ndim):
        if np.isclose(pars[j], 0.0):
            dp = np.random.normal(loc=0.0, scale=amp) # add a perturbation
            pars[j] = dp
    for i in range(nwalkers):
        p0[i,:] = pars[:]*(1.0 + amp*np.random.randn(ndim))

        # add perturbation according to q covariance matrix
        if Nq > 1:
            xi0 = (xi0_lims[1]-xi0_lims[0])/2.0
            cov = cov_q(Nq, zbins, Nscale=Nscale, bin_type=bin_type, xi0=xi0)
            q0 = np.random.multivariate_normal(np.zeros(Nq), cov)
            p0[i,q_ind] = q0

    return p0

def get_resume_state(nwalkers, chain_name, nsamps, nsamps_add):
    """
    Note that even if the file exists other hardcoded parameters
    might have changed slightly so use with care when appending to
    existing chains
    update file name for later when saving chains

    """
    chain = np.loadtxt(chain_name)
    p_last = chain[-nwalkers:, :] #shape is (nwalkers,ndim)
    lnprob = None # will compute lnprob for above p
    # lnprob = [logpost(p0, pars_names) for p0 in list(p)]
    rootname = '{}_{}_{:d}{:d}'.format(cosmo_name, dstr, nwalkers, nsamps + nsamps_add)
    chain_name = '{}/chains/{}_samples.txt'.format(dirname, rootname)
    nsamples = nsamps_add
    return p_last, lnprob, chain_name, nsamples, chain

def check_model(Nq, cosmo_name):
    # do some basic checks
    if cosmo_name not in valid_models:
        raise ValueError('cosmology specified {} not found'.format(cosmo_name))
    if Nq is not None and cosmo_name not in multi_q_models:
        raise ValueError('cosmology model {} not compatible with {} q parameters'.format(cosmo_name, Nq))

def get_zbins(Nq, zmax, bin_type):
    if Nq is not None:
        # set zbins
        if bin_type == 'aunif': # means uniform in a
            abins = np.linspace(1./(1.0+zmax), 1.0, Nq+1)
            zbins = 1.0/abins[::-1] - 1.0
        if bin_type == 'zunif':
            zbins = np.linspace(0.0, zmax, Nq+1)
    else:
        zbins = None
    return zbins

def get_a_knots(Nq, zmax):
    if Nq is not None:
        a_knots = np.linspace(1./(1.0+zmax), 0.99, Nq)
    else:
        a_knots = None
    return a_knots

# parse command line arguments
parser = argparse.ArgumentParser()

# positional args
parser.add_argument('model', type=str, help='the parameter model, e.g. FlatIVCDM_H0')
parser.add_argument('nwalkers', type=int)
# parser.add_argument('burnin', type=int)
parser.add_argument('nsamples', type=int)
parser.add_argument('data_list', type=str, nargs='*', help='choose from BAO, CMB, RSD, SNIa, CC')

# optional args (e.g. if resume not specified in cmd line then is assigned value of False)
parser.add_argument('--resume', action='store_true')
parser.add_argument('-add', '--nsamps_add', type=int, default=0, help='number of samples after resuming')
parser.add_argument('--amp', type=float, default=0.1, help='parameter perturbation')
parser.add_argument('--prior_only', action='store_true')
parser.add_argument('--no_omegab', action='store_true')


# optional args related to binned model (if e.g. zmax not specified then assigned a value of 1.5)
parser.add_argument('--Nq', type=int, default=None, help='number of q parameters/bins')
parser.add_argument('--zmax', type=float, default=1.5, help='maximum redshift binning range')
parser.add_argument('--Nscale', type=int, default=4, help='smoothing scale given as multiple of bin width')
parser.add_argument('--xi0', type=float, default=0.1, help='amplitude of q covariance')
parser.add_argument('--xi0_lims', type=float, nargs=2, default=[0.05,2.0], help='xi0 prior limits')
parser.add_argument('--bin_type', type=str, default='zunif', help='either uniform in a (aunif) or z (zunif)')

# optional args related to smooth model
parser.add_argument('--basis', type=str, default=None, help='name of basis for smooth models')

args = parser.parse_args()

cosmo_name = args.model
nwalkers = args.nwalkers
# burnin = args.burnin
nsamples = args.nsamples
data_list = args.data_list # list of strings

resume = args.resume # false by default
nsamps_add = args.nsamps_add
amp = args.amp
prior_only = args.prior_only # false by default
use_omegab = not args.no_omegab # true by default


Nq = args.Nq
zmax = args.zmax
Nscale = args.Nscale
xi0 = args.xi0
xi0_lims = args.xi0_lims # list
bin_type = args.bin_type
basis = args.basis

check_model(Nq, cosmo_name)
if Nq > 1:
    q_ind = cosmo(Nq=Nq).get_q_indices(cosmo_name)
else:
    q_ind = None

# output files will begin with the following string
dstr = '_'.join(data_list) if not prior_only else 'PRIOR_ONLY'
rootname = '{}_{}_{:d}{:d}'.format(cosmo_name, dstr, nwalkers, nsamples)
chain_name = '{}/chains/{}_samples.txt'.format(dirname, rootname)

M = cosmo(Nq=Nq).get_model(cosmo_name,data_list)

if basis == 'cubic_spline':
    free_knots = get_a_knots(Nq, zmax)
else:
    free_knots = None

# instantiate the likelihood object for the model and chosen data
# cosmology optional arguments are to be passed through the joint_loglike class:
loglike_joint = joint_loglike(cosmo_name=M['model'], pars_names=M['names'], data=data_list, \
                              zmax=zmax, basis=basis, free_knots=free_knots, # optional cosmology arguments
                              use_omegab=use_omegab)

pars = M['x0']
emcee_args = [M['names']]

zbins = get_zbins(Nq, zmax, bin_type)

multi_q_kwargs = {'Nq': Nq,
                  'zbins': zbins,
                  'Nscale': Nscale,
                  'bin_type': bin_type,
                  'xi0_lims': xi0_lims,
                  'xi0': xi0,
                  'q_ind': q_ind
                  }

emcee_kwargs = {'prior_only': prior_only}
emcee_kwargs.update(multi_q_kwargs)


# Begin MCMC with the above parameters/specifications

# need: nwalkers, nsamples, burnin, pars, loglike object (and args, kwargs)
# and chain_name

# Everything below this line is generic
# ---------------------------------------------------

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
# pool = None


# display job arguments
print '-------------------------------------'
for arg in vars(args):
    print('{:12}\t{}'.format(arg, getattr(args, arg)))
print '-------------------------------------\n'


ndim = pars.size

# Initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=emcee_args, kwargs=emcee_kwargs, pool=pool)


if resume and os.path.exists(chain_name): # skip burnin
    print 'Resuming from existing chain'
    p, lnprob, chain_name, nsamples, samps_old = get_resume_state(nwalkers, chain_name, nsamples, nsamps_add)
# else:
#     print 'begin burn-in'
#     p0 = get_p0(pars, nwalkers, amp)
#     for i, (p, lnprob, lnlike) in enumerate(sampler.sample(p0, iterations=burnin)):
#         if (i+1) % 50 == 0:
#             print("{0:5.1%}".format(float(i) / burnin))
#     print 'burn-in complete'

# Reset the chain / clear the burn-in samples
# sampler.reset()


# Start the sampling from the final position in the burn-in chain
print '\nbegin sampling'
p = get_p0(pars, nwalkers, amp, **multi_q_kwargs)
for i, (p, lnprob, lnlike) in enumerate(sampler.sample(p, iterations=nsamples)):
    if (i+1) % 100 == 0:
        print("{0:5.1%}".format(float(i)/nsamples))
        np.savetxt(chain_name, sampler.flatchain)

# MCMC sampling complete
print '\nmean acceptance fraction: ', np.mean(sampler.acceptance_fraction)

try:
    pool.close()
except:
    pass


print '\nSaving chain to file'
samps = sampler.flatchain

if resume:
    samps = np.vstack((samps_old,samps))

# overwrites file if already exists
np.savetxt(chain_name, samps)


# tau = sampler.get_autocorr_time()
# print tau

