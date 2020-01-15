JLA dataset can be obtained from

    http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/A+A/568/A22

Only tablef3.f is required.

The covariance matrices can be downloaded from

    http://supernovae.in2p3.fr/sdss_snls_jla/covmat_v6.tgz

Redshift zcmb in JLA.tsv is based on my own calculations which is
slightly different to zcmb in tablef3.dat. Don't understand why
this leads to such large differences in parameters but to be
sure just use tablef3.dat if in doubt.

The file lcparams_full_long.txt is the Pantheon catalogue but only
contains data on name,zcmb,zhel,dz,mb,dmb the rest are all zero
mb refers to mu+M (see table 17 Scolnic et al. 2018)