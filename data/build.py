import numpy as np

def boostz(z,vel,RA0,DEC0,RAdeg,DECdeg):
    # Angular coords should be in degrees and velocity in km/s
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) \
        + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vel/C)*costheta*(1.+z)

C = 2.99792458e5 # km/s

# Tully et al 2008
vcmb = 371.0 # km/s
l_cmb = 264.14
b_cmb = 48.26
# converts to
ra_cmb = 168.0118667
dec_cmb = -6.98303424


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
data = np.genfromtxt('tablef3.dat', delimiter=delim, dtype=ndtypes, autostrip=True)

zcmb = data['zcmb']
mb = data['mb']
x1 = data['x1']
c = data['c']
logMass = data['logMst'] # log_10_ host stellar mass (in units=M_sun)
survey = data['set']
zhel = data['zhel']
ra = data['RAdeg']
dec = data['DEdeg']

# Survey values key:
#   1 = SNLS (Supernova Legacy Survey)
#   2 = SDSS (Sloan Digital Sky Survey: SDSS-II SN Ia sample)
#   3 = lowz (from CfA; Hicken et al. 2009, J/ApJ/700/331
#   4 = Riess HST (2007ApJ...659...98R)

zcmb1 = boostz(zhel,vcmb,ra_cmb,dec_cmb,ra,dec)

JLA = np.column_stack((zcmb1,mb,x1,c,logMass,survey,zhel,ra,dec))
np.savetxt('jla.tsv', JLA, delimiter='\t', \
           fmt=('%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%i','%9.7f','%11.7f','%11.7f'))
