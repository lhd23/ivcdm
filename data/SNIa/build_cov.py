#!/usr/bin/python

import numpy as np
import pyfits
import tarfile
SPEEDOFLIGHT = 2.99792458e5 #km/s

def get_sigma_z(z): # cf. Betoule et al 2014 eqn (13)   
    return (5. * 150. / SPEEDOFLIGHT) / (np.log(10.) * z)

# once downloaded extract tarball
tar = tarfile.open('covmat_v6.tgz')
tar.extractall()
tar.close()

C_stat = pyfits.getdata('covmat/C_stat.fits')
C_cal = pyfits.getdata('covmat/C_cal.fits')
C_model = pyfits.getdata('covmat/C_model.fits')
C_bias = pyfits.getdata('covmat/C_bias.fits')
C_host = pyfits.getdata('covmat/C_host.fits')
C_dust = pyfits.getdata('covmat/C_dust.fits')
C_pecvel = pyfits.getdata('covmat/C_pecvel.fits')
C_nonia = pyfits.getdata('covmat/C_nonia.fits')



C_lens = np.zeros_like(C_stat) # shape=(2220,2220)
C_z = np.zeros_like(C_stat)
C_coh = np.zeros_like(C_stat)


sigma_coh, sigma_lens, z = np.genfromtxt('covmat/sigma_mu.txt', skip_header=4, unpack=True)

# \(C_{\rm pecvel}\): uncertainty on the peculiar velocity correction
#    (Systematic only: does not include the \(\sigma_z\) term of Eq. 13.)

sig_z = get_sigma_z(z)
for i in range(z.size):
    C_lens[3*i,3*i] = sigma_lens[i]**2
    C_coh[3*i,3*i] = sigma_coh[i]**2
    C_z[3*i,3*i] = sig_z[i]**2


# These are 2220x2220 matrices
pyfits.writeto('covmat/C_lens.fits', C_lens)
pyfits.writeto('covmat/C_coh.fits', C_coh)
pyfits.writeto('covmat/C_z.fits', C_z)


# uncomment to convert to .npy
# np.save('covmat/C_stat', C_stat)
# np.save('covmat/C_cal', C_cal)
# np.save('covmat/C_model', C_model)
# np.save('covmat/C_bias', C_bias)
# np.save('covmat/C_host', C_host)
# np.save('covmat/C_dust', C_dust)
# np.save('covmat/C_pecvel', C_pecvel)
# np.save('covmat/C_nonia', C_nonia)
# np.save('covmat/C_lens', C_lens)
# np.save('covmat/C_coh', C_coh)
# np.save('covmat/C_z', C_z)




