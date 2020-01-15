import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 22

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from getdist import plots, MCSamples

from scipy import linalg

from ivcdm import FlatIVCDM_binned
from models import cosmo
from likelihood import cov_q, matrix_running_average,joint_loglike

dir=os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, dir)
from mycorner import *


# a helper class to store all the relevant information for binned model
class binned_model:
    
    def __init__(self, file_name, cosmo_name, nwalkers, nsamples, 
                 skip_steps=0, kill=False, Nq=0, zmax=None, xi0=0.2, Nscale=4, qfid=None,
                 data_list = ['BAO','CC','RSD','SNIa','CMB']):

        self.file_name = file_name
        self.data_list = data_list
        self.samples = np.loadtxt(dir+'chains/5/{}.txt'.format(self.file_name))
        self.cosmo_name = cosmo_name
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.skip_steps = skip_steps
        
        # bin specifications
        self.Nq = Nq
        self.zmax = zmax
        self.xi0 = xi0
        self.Nscale = Nscale
        self.amin = 1./(1.+self.zmax)
        
        self.qfid = qfid # None means qfid=0

        cos = cosmo(Nq=self.Nq)
        self.M = cos.model[self.cosmo_name]

        self.pars_names = self.M['names']
        self.labels = get_labels(self.pars_names, Nq=self.Nq)
        self.npars = len(self.pars_names)
        self.ind = {par: i for i,par in enumerate(self.pars_names)}
        self.names_q = cos.names_q
        self.q_ind = cos.get_pars_indices(self.names_q, self.cosmo_name)
        self.labels_q = [self.labels[i] for i in self.q_ind]
        self.abins = np.linspace(1./(1.0+self.zmax), 1.0, self.Nq+1)
        self.zbins = 1.0/self.abins[::-1] - 1.0
        self.diffz = np.diff(self.zbins)
        self.zcentres = self.zbins[:self.Nq] + 0.5*self.diffz

        # create the getdist object
        self.MCsamps = get_MCSamples(self.samples, self.cosmo_name, self.Nq,
                                     self.nwalkers, self.nsamples, skip_steps=self.skip_steps)

        # for some chains need to do a cleaning step
        if kill:
            self.search_and_destroy()

    def search_and_destroy(self):
        # remove worst performing walker
        # note need to remove burnin before can compute stats on chain
        self.samples, self.nwalkers = clean_samples(self.samples, self.nwalkers, self.nsamples, 
                                                    self.npars, self.pars_names, self.labels, self.skip_steps)
        self.MCsamps = get_MCSamples(self.samples, self.cosmo_name, self.Nq, self.nwalkers, self.nsamples,
                                    skip_steps=self.skip_steps)

    @property
    def pars_means(self):
        return self.MCsamps.getMeans()

    @property
    def q_mean(self):
        return self.pars_means[self.q_ind]

    @property
    def Om0_mean(self):
        return self.pars_means[self.ind.get('Om0')]

    @property
    def Ob0_mean(self):
        return self.pars_means[self.ind.get('Ob0')]

    @property
    def q_68CL_lims(self):
        qlo_arr = np.zeros(self.Nq)
        qhi_arr = np.zeros(self.Nq)
        for i in range(self.Nq):
            iq = self.q_ind[i]
            qlo_arr[i], qhi_arr[i] = self.MCsamps.twoTailLimits(iq, 0.683)
        return qlo_arr, qhi_arr

    @property
    def q_95CL_lims(self):
        qlo_arr = np.zeros(self.Nq)
        qhi_arr = np.zeros(self.Nq)
        for i in range(self.Nq):
            iq = self.q_ind[i]
            qlo_arr[i], qhi_arr[i] = self.MCsamps.twoTailLimits(iq, 0.954)
        return qlo_arr, qhi_arr
 
    def q_prob_cloud(self, nbands=50):
        confs = np.linspace(0.03, 0.98, nbands)
        qlo = np.zeros((confs.size, self.Nq))
        qhi = np.zeros((confs.size, self.Nq))
        for j,conf in enumerate(confs):
            for i in range(self.Nq):
                iq = self.q_ind[i]
                qlo[j,i], qhi[j,i] = self.MCsamps.twoTailLimits(iq, conf)
        return qlo, qhi

    def get_alpha(self, q):
        return np.dot(self.inv_A.T, q)

    def project_onto_q(self, alpha, M):
        q_arr = np.zeros(self.Nq)
        for i in range(M):
            q_arr += alpha[i] * self.KL_basis[i]
        return q_arr

    def q_KL_projected(self, q, M):
        alpha = self.get_alpha(q)
        q_arr = np.zeros(self.Nq)
        for i in range(M):
            q_arr += alpha[i] * self.KL_basis[i]
        return q_arr

# posterior methods
    @property
    def cov_q(self): # posterior covariance
        return self.MCsamps.cov()[np.ix_(self.q_ind, self.q_ind)]

    @property
    def corr_q(self): # posterior correlation
        return self.MCsamps.corr()[np.ix_(self.q_ind, self.q_ind)]

    @property
    def Fisher_q(self): # inverse posterior covariance
        return linalg.inv(self.cov_q)

# priors methods
    @property
    def corr_q_prior(self):
        self.corr_q_prior = self.cov_q_prior
        for i in range(self.Nq):
            for j in range(self.Nq):
                self.corr_q_prior[i,j] /= np.sqrt(self.cov_q_prior[i,i])
                self.corr_q_prior[i,j] /= np.sqrt(self.cov_q_prior[j,j])
        return self.corr_q_prior

    @property
    def Fisher_q_prior(self): # inverse posterior covariance
        if self.qfid is None:
            cov_q_prior = cov_q(self.Nq, self.zbins, Nscale=self.Nscale, bin_type='aunif', xi0=self.xi0)
            return linalg.inv(cov_q_prior)
        elif self.qfid == 'running': # NB. hardcode for running 5 bin average
            cov_q_prior = cov_q(self.Nq, self.zbins, Nscale=self.Nscale, bin_type='aunif', xi0=self.xi0)
            A = matrix_running_average(self.Nq, Nw=5)
            I_A = np.eye(self.Nq) - A
            inv_cov_prior = linalg.inv(cov_q_prior)
            return np.dot(I_A.T, np.dot(inv_cov_prior, I_A))
        else:
            raise ValueError('fid: {} not valid'.format(fid))

    @property
    def cov_q_prior(self):
        if self.qfid is None:
            return cov_q(self.Nq, self.zbins, Nscale=self.Nscale, bin_type='aunif', xi0=self.xi0)
        elif self.qfid == 'running': # NB. hardcode for running 5 bin average
            return linalg.inv(self.Fisher_q_prior)
        else:
            raise ValueError('fid: {} not valid'.format(fid))

    @property
    def neff(self): # effective number of q
        if self.qfid == 'marge':
            raise ValueError('not valid for marginalised xi0 model')
        return self.Nq - np.trace(np.dot(self.Fisher_q_prior, self.cov_q))

    def get_PCA(self):
        U, s, _ = linalg.svd(self.Fisher_q)
        # svd orders from best to worst (largest to smallest eigenvalues)
        assert s[0] > s[-1] # check order is from best to worst

        self.PCA_W = U.T # decorrelation matrix
        self.PCA_eigval = s
        self.PCA_alpha = np.dot(self.PCA_W, self.q_mean) # coefficients of PC basis (uncorrelated)
        self.PCA_sigma_alpha = 1./np.sqrt(s) # errors
        
    def get_KL(self): # Karhunen-Loeve decomposition
        F_post = self.Fisher_q
        if self.qfid is None:
            F_prior = self.Fisher_q_prior
            w, v = linalg.eigh(F_prior, F_post) # eigh orders highest to lowest S/N
        elif self.qfid == 'running':
            F_prior = self.Fisher_q_prior
            w, v = linalg.eigh(F_prior, F_post)
        else:
            raise ValueError('fid: {} not valid'.format(fid))

        self.KL_SN = 1.0/np.abs(w)
        # assert self.KL_SN[0] > self.KL_SN[-1] # check order is from best to worst

        self.KL_A = v.T # rows of A are eigenvectors
        self.KL_basis = [self.KL_A[i,:] for i in range(self.Nq)] # first is best mode
        self.inv_A = linalg.inv(self.KL_A)
        self.KL_alpha = np.dot(self.inv_A.T, self.q_mean)
        self.KL_varalpha = np.ones(self.Nq)

    def bias2(self, M):
        self.get_KL()
        q_sum = 0.0
        for i in range(M):
            q_sum += self.KL_alpha[i] * self.KL_basis[i]
        bias2_ = 0.0
        for j in range(self.Nq):
            bias2_ += (q_sum[j] - self.q_mean[j])**2
        return bias2_

    def bin_var(self, M): # return array containing var in each bin
        self.get_KL()
        vars = np.zeros(self.Nq) # variance in each bin
        for j in range(self.Nq): # sum over bins
            for i in range(M): # sum over modes (eq 5 HS03)
                mode = self.KL_basis[i]
                vars[j] += mode[j]**2 * self.KL_varalpha[j]
        return vars

    def variance(self, M): # total variance over all bins
        return np.sum(self.bin_var(M))

    def MSE(self, M):
        return self.bias2(M) + self.variance(M)

    def plot_KL_basis(self, add_modes=0):
        self.get_KL()

        N = int(np.ceil(self.neff))
        print 'number of effective parameters: ', self.neff
        print 'S/N of the best {} eigenmodes:'.format(N), self.KL_SN[:N]

        fig, ax = plt.subplots(N+1, 1, sharex=True, sharey=False, figsize=(10,2*N))
        for i in range(N):
            ax[i].plot(self.zcentres, self.KL_basis[i], 'k')
            ax[i].axhline(y=0, c='k', ls=':')
            ax[i].annotate(r'$i={0:}$, $S/N={{{1:.1e}}}$'.format(i+1,self.KL_SN[i]),
                            xy=(0.7, 0.8), xycoords='axes fraction', fontsize=12)

        # truncate basis at first neff modes (rounded to nearest integer)
        num_modes = int(np.rint(self.neff)) + add_modes
        mode_sum = 0.0
        for i in range(num_modes):
            mode_sum += self.KL_alpha[i] * self.KL_basis[i]
        ax[-1].plot(self.zcentres, mode_sum, 'k')
        ax[-1].plot(self.zcentres, self.q_mean, 'k--')
        ax[-1].axhline(y=0, c='k', ls=':')
        ax[-1].annotate('Sum of first {} modes'.format(num_modes),
                        xy=(0.75, 0.8), xycoords='axes fraction', fontsize=12)

        plt.rc('font', size=12)
        plt.xlabel(r'redshift $z$') # of bin centres
        plt.xlim(0, self.zcentres[-1])
        plt.show()

    def plot_cov(self, kind='posterior'):
        if kind == 'posterior':
            C, Corr, F = self.cov_q, self.corr_q, self.Fisher_q
        elif kind == 'prior':
            C, Corr, F = self.cov_q_prior, self.corr_q_prior, self.Fisher_q_prior

        fig = plt.figure(figsize=(19,4))

        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(C, cmap='viridis', interpolation='nearest')
        ax1.set_title('Covariance')
        fig.colorbar(im1)

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(Corr, cmap='viridis', interpolation='nearest')
        ax2.set_title('Correlation matrix')
        fig.colorbar(im2)

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(F, cmap='viridis', interpolation='nearest')
        ax3.set_title('Fisher matrix')
        fig.colorbar(im3)

        plt.rc('font', size=14)
        plt.show()
        
    def get_DM_DE_derived(self, pts=50):
        if self.Nq != 20 or self.cosmo_name != 'FlatIVCDM_binned':
            raise ValueError('this method is hardcoded for Nq=20 and flat models only')

        p = self.MCsamps.getParams()
        a_arr = np.linspace(self.amin+1e-4, 1.0, pts)
        for i,ai in enumerate(a_arr):
            Omv = np.zeros(p.Om0.size)
            Omc = np.zeros(p.Om0.size)
            for j, pars in enumerate(zip(p.Om0, p.Ob0, \
                                     p.q1, p.q2, p.q3, p.q4, p.q5, p.q6, p.q7, p.q8, p.q9, p.q10, \
                                     p.q11, p.q12, p.q13, p.q14, p.q15, p.q16, p.q17, p.q18, p.q19, p.q20)):
                Om0 = pars[0]
                Ob0 = pars[1]
                qs = np.array(pars[2:])
                m = FlatIVCDM_binned(Om0=Om0, Ob0=Ob0, qs=qs, zbins=self.zbins)
                Omv[j] = m.Omega_vac(ai)
                Omc[j] = m.Omega_cdm(ai)

            try: # because MCsamps throws error if there is parameter with same name already
                self.MCsamps.addDerived(Omv, name='Omv_{}'.format(i), label='Omv_{}'.format(i))
                self.MCsamps.addDerived(Omc, name='Omc_{}'.format(i), label='Omc_{}'.format(i))
            except:
                pass
        self.MCsamps.updateBaseStatistics()

    def plot_qrec(self, nbands=50, inset=False):
        qlo_68CL, qhi_68CL = self.q_68CL_lims
        qlo_95CL, qhi_95CL = self.q_95CL_lims

        fig, ax = plt.subplots(figsize=[8, 6])
        ax.plot(self.zcentres, self.q_mean, c='white', ls='-', marker='',)
        ax.plot(self.zcentres, qhi_68CL, c='grey', ls='--', marker='',)
        ax.plot(self.zcentres, qlo_68CL, c='grey', ls='--', marker='',)
        ax.plot(self.zcentres, qhi_95CL, c='grey', ls='--', marker='',)
        ax.plot(self.zcentres, qlo_95CL, c='grey', ls='--', marker='',)
        ax.axhline(y=0.0, c='k', ls=':', lw=2, alpha=3)

        qlo, qhi = self.q_prob_cloud(nbands=nbands)
        colors = plt.cm.magma(np.linspace(0,1,nbands))
        alphas = np.linspace(0.9,0.07,nbands)
        confs = np.linspace(0.03,0.98,nbands)
        for j,conf in enumerate(confs):
            ax.fill_between(self.zcentres, qlo[j], qhi[j], alpha=0.02, facecolor='b', edgecolor='None')

        plt.rc('font', size=22)
        ax.set_xlim(self.zcentres[0], self.zcentres[-1])
        ax.set_ylim(-1, 1)
        ax.set_xlabel('redshift $z$')
        ax.set_ylabel('interaction history $q(z)$')
        # plt.yticks(np.linspace(-2,2,5))

        if inset:
            axins = zoomed_inset_axes(ax, zoom=2, loc=1)
            axins.plot(self.zcentres, self.q_mean, 'k-', lw=2)
            axins.plot(self.zcentres, qhi_68CL, c='grey', ls='--', marker='',)
            axins.plot(self.zcentres, qlo_68CL, c='grey', ls='--', marker='',)
            axins.plot(self.zcentres, qhi_95CL, c='grey', ls='--', marker='',)
            axins.plot(self.zcentres, qlo_95CL, c='grey', ls='--', marker='',)
            axins.axhline(y=0.0, c='k', ls=':', alpha=3)

            for j,conf in enumerate(confs):
                axins.fill_between(self.zcentres, qlo[j], qhi[j], alpha=0.02, facecolor='b', edgecolor='None')

            axins.set_xlim(self.zcentres[0], 0.2)
            axins.set_ylim((-0.2, 0.3))
            # axins.set_xticklabels([])

            # fix the number of ticks on the inset axes
            axins.yaxis.get_major_locator().set_params(nbins=3)
            axins.xaxis.get_major_locator().set_params(nbins=1)

            # mark_inset(ax, axins, loc1=2, loc2=2, fc='None', ec="0.85")
            # mark_inset(ax, axins, loc1=1, loc2=4, fc='None', ec="0.85")
            axins.tick_params(axis='both', which='major', labelsize=12)

        plt.show()
        # plt.savefig('qrec-20-prob-AB.pdf', bbox_inches='tight')

    def get_alpha_derived(self):
        if self.Nq != 20 or self.cosmo_name != 'FlatIVCDM_binned':
            raise ValueError('this method is hardcoded for Nq=20 and flat models only')

        self.get_KL()
        p = self.MCsamps.getParams()

        alpha = np.zeros((p.Om0.size, self.Nq))
        for j, pars in enumerate(zip(p.q1, p.q2, p.q3, p.q4, p.q5, p.q6, p.q7, p.q8, p.q9, p.q10, \
                                 p.q11, p.q12, p.q13, p.q14, p.q15, p.q16, p.q17, p.q18, p.q19, p.q20)):

            qs = np.array(pars)
            alpha[j,:] = np.dot(self.inv_A.T, qs)

        for i in range(self.Nq):
            try: # because MCsamps throws error if there is parameter with same name already
                self.MCsamps.addDerived(alpha[:,i], name='alpha_{}'.format(i+1), label='alpha_{}'.format(i+1))
            except:
                pass
        self.MCsamps.updateBaseStatistics()

    def chisquared(self):
        loglike_joint = joint_loglike(cosmo_name=self.cosmo_name, pars_names=self.pars_names,
                                    data=self.data_list, zmax=self.zmax)
        return loglike_joint.m2loglike(self.pars_means)

    def get_qrec_proj_samples(self, M, Ndraws=2000, conf=0.683):
        self.get_KL()
        self.get_alpha_derived()

        alpha_los = np.zeros(self.Nq)
        alpha_his = np.zeros(self.Nq)
        i0 = self.MCsamps.paramNames.numberOfName('alpha_1')
        for i in range(self.Nq):
            alpha_los[i], alpha_his[i] = self.MCsamps.twoTailLimits(i0+i, conf)

        qsamples = self.samples[:,self.q_ind]

        isamples = np.random.choice(qsamples.shape[0], size=Ndraws, replace=False)

        alpha_samples = np.zeros((Ndraws,self.Nq))
        for j in range(Ndraws):
            i = isamples[j]
            qs = qsamples[i]
            alpha_samples[j,:] = self.get_alpha(qs)


        alphas_in = np.empty((Ndraws,M), dtype=bool)
        for j in range(Ndraws):    
            alphas = alpha_samples[j]
            for i in range(M):
                if alpha_los[i] <= alphas[i] <= alpha_his[i]:
                    alphas_in[j,i] = True
                else:
                    alphas_in[j,i] = False

        mask = np.all(alphas_in, axis=1)

        # for i in range(Ndraws):
        #     print alphas_in[i,:], mask[i]

        alpha_samples_new = alpha_samples[mask]
        self.N_new = alpha_samples_new.shape[0]
        print 'N_new: ', self.N_new

        self.alpha_samples_padded = np.zeros((alpha_samples_new.shape[0], self.Nq))
        for i in range(self.N_new):
            self.alpha_samples_padded[i,:M] = alpha_samples_new[i,:M]

        qs_rec = np.zeros((self.N_new,self.Nq))
        for i,alp in enumerate(self.alpha_samples_padded):
            qs_rec[i] = self.project_onto_q(alp, M)

        self.q_68lo = np.zeros(self.Nq)
        self.q_68hi = np.zeros(self.Nq)
        for i in range(self.Nq):
            self.q_68lo[i] = np.amin(qs_rec[:,i])
            self.q_68hi[i] = np.amax(qs_rec[:,i])

        return qs_rec


# ===============================================

class binned_model_marge: # marginalised xi0

    def __init__(self, file_name, nwalkers, nsamples, skip_steps=0):
        self.file_name = file_name
        self.samples = np.loadtxt(dir+'chains/5/{}.txt'.format(self.file_name))
        self.cosmo_name = 'FlatIVCDM_binned'
        self.nwalkers = nwalkers
        self.nsamples = nsamples
        self.skip_steps = skip_steps
        
        # bin specifications
        self.Nq = 20
        self.zmax = 1.5
        self.xi0_lims = (0.2,2.0)
        self.Nscale = 4
        self.amin = 1./(1.+self.zmax)
        
        self.npars = self.Nq
        cos = cosmo(Nq=self.Nq)
        M = cos.model[self.cosmo_name]

        self.names_q = ['q{}'.format(i+1) for i in range(self.Nq)]
        self.labels_q = [r'q_{{{}}}'.format(i+1) for i in range(self.Nq)]

        samples_skipped = samples_skip(self.samples, self.nwalkers, self.nsamples, self.npars, self.skip_steps)
        self.MCsamps = MCSamples(samples=samples_skipped, names=self.names_q, labels=self.labels_q)

    @property
    def cov_q(self): # covariance
        return self.MCsamps.cov()

    @property
    def corr_q(self): # correlation
        return self.MCsamps.corr()

    @property
    def Fisher_q(self): # inverse covariance ('Fisher')
        return linalg.inv(self.cov_q)

    def plot_cov(self, kind='posterior'):
        C, Corr, F = self.cov_q, self.corr_q, self.Fisher_q

        fig = plt.figure(figsize=(19,4))

        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(C, cmap='viridis', interpolation='nearest')
        ax1.set_title('Covariance')
        fig.colorbar(im1)

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(Corr, cmap='viridis', interpolation='nearest')
        ax2.set_title('Correlation matrix')
        fig.colorbar(im2)

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(F, cmap='viridis', interpolation='nearest')
        ax3.set_title('Fisher matrix')
        fig.colorbar(im3)

        plt.rc('font', size=14)
        plt.show()

    def MSE(self, M):
        self.get_KL()
        q_sum = 0.0
        bias2 = 0.0
        variance = 0.0
        for i in range(M):
            q_sum += self.KL_alpha[i] * self.KL_basis[i]
            bias2 += (q_sum[i] - self.q_mean[i])**2
            variance += (self.KL_basis[i] * self.KL_varalpha[i])**2
        return bias2 + variance
