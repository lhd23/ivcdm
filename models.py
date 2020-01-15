import numpy as np

class cosmo:

    def __init__(self, Nq=10):
        self.bds_nuisance = ((None,None),(None,None),(0,None),(None,None),
                           (None,None),(0,None),(None,None),(0,None),)
        self.names_nuisance = ['A', 'X0', 'VX', 'B', 'C0', 'VC', 'M0', 'VM']
        self.N_nuisance = 8

        # number of interaction parameters
        self.Nq = Nq # 10 # 15 # 30
        if Nq is None:
            self.Nq = 0
            self.names_q = []
        else:
            self.Nq = Nq
            self.names_q = ['q{}'.format(i+1) for i in range(self.Nq)]

        # dictionary of dictionaries
        self.model = {

            'FlatLCDM': 
                    {
                        'model':   'FlatLCDM',
                        'names':   ['Om0', 'Ob0', 'sigma80'] + self.names_nuisance,
                           'x0':   np.array([ 3.18791800e-01,  4.89667367e-02,  7.77884776e-01,  1.34609116e-01,
                                              4.20636967e-02,  8.70386375e-01,  3.07112420e+00, -1.91274179e-02,
                                              5.08485379e-03, -1.91639094e+01,  1.14865136e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + self.bds_nuisance
                    },

            # FlatLCDM+H0
            'FlatLCDM_H0':
                    {
                        'model':   'FlatLCDM',
                        'names':   ['Om0', 'Ob0', 'sigma80', 'H0'] + self.names_nuisance,
                           'x0':   np.array([ 2.99753483e-01,  4.76399715e-02,  7.89916331e-01,  6.87921412e+01,
                                            1.35046431e-01,  4.31534202e-02,  8.70923258e-01,  3.07711238e+00,
                                           -2.03226497e-02,  5.09950109e-03, -1.91248634e+01,  1.15018495e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + ((60,80),) + self.bds_nuisance
                    },

            'LCDM':
                    {
                        'model':   'LCDM',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'sigma80'] + self.names_nuisance,
                           'x0':   np.array([ 3.13207132e-01,  6.89471964e-01,  4.95519216e-02,  7.80597912e-01,
                                            1.34764942e-01,  4.24535387e-02,  8.70567367e-01,  3.07318130e+00,
                                           -1.95414157e-02,  5.08969342e-03, -1.91670795e+01,  1.14909124e-02]),
                       'bounds':   3*((0.01,0.99),) + ((0.3,1.3),) + self.bds_nuisance
                    },


            'LCDM_H0':
                    {
                        'model':   'LCDM',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'sigma80', 'H0'] + self.names_nuisance,
                           'x0':   np.array([ 2.99978124e-01,  6.99028522e-01,  4.73377987e-02,  7.89978117e-01,
                                            6.89211995e+01,  1.35031675e-01,  4.31161217e-02,  8.70902381e-01,
                                            3.07688296e+00, -2.02839856e-02,  5.09906110e-03, -1.91205816e+01,
                                            1.15017009e-02]),
                       'bounds':   3*((0.01,0.99),) + ((0.3,1.3),) + ((60,80),) + self.bds_nuisance
                    },


            'FlatIVCDM':
                    {
                        'model':   'FlatIVCDM',
                        'names':   ['Om0', 'Ob0', 'q', 'sigma80'] + self.names_nuisance,
                           'x0':   np.array([ 3.11370442e-01,  4.97572676e-02,  4.17874734e-02,  7.68854595e-01,
                                            1.34751053e-01,  4.24250978e-02,  8.70540584e-01,  3.07296559e+00,
                                           -1.94909839e-02,  5.08903425e-03, -1.91667145e+01,  1.14905470e-02]),
                       'bounds':   2*((0.01,0.99),) + ((-1,1),) + ((0.3,1.3),) + self.bds_nuisance
                    },


            'FlatIVCDM_H0':
                    {
                        'model':   'FlatIVCDM',
                        'names':   ['Om0', 'Ob0', 'q', 'sigma80', 'H0'] + self.names_nuisance,
                           'x0':   np.array([ 2.99742705e-01,  4.76664388e-02,  8.69202764e-04,  7.89622498e-01,
                                            6.87805208e+01,  1.35046067e-01,  4.31537879e-02,  8.70917393e-01,
                                            3.07710348e+00, -2.03213250e-02,  5.09948137e-03, -1.91252246e+01,
                                            1.15019126e-02]),
                       'bounds':   2*((0.01,0.99),) + ((-1,1),) + ((0.3,1.3),) + ((60,80),) + self.bds_nuisance
                    },


            'IVCDM':
                    {
                        'model':   'IVCDM',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'q', 'sigma80'] + self.names_nuisance,
                           'x0':   np.array([ 3.12389197e-01,  6.74010665e-01,  4.96837533e-02,  1.89493437e-01,
                                            7.27767449e-01,  1.34490125e-01,  4.17743497e-02,  8.70207632e-01,
                                            3.06946133e+00, -1.87531933e-02,  5.08025098e-03, -1.91617074e+01,
                                            1.14841389e-02]),
                       'bounds':   3*((0.01,0.99),) + ((-1,1),) + ((0.3,1.3),) + self.bds_nuisance
                    },


            'IVCDM_H0':
                    {
                        'model':   'IVCDM',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'q', 'sigma80', 'H0'] + self.names_nuisance,
                           'x0':   np.array([ 3.01276060e-01,  6.87752245e-01,  4.77271415e-02,  1.20482767e-01,
                                            7.52394118e-01,  6.86910027e+01,  1.34816761e-01,  4.25882553e-02,
                                            8.70619605e-01,  3.07384300e+00, -1.96704895e-02,  5.09121647e-03,
                                           -1.91236660e+01,  1.14924249e-02]),
                       'bounds':   3*((0.01,0.99),) + ((-1,1),) + ((0.3,1.3),) + ((60,80),) + self.bds_nuisance
                    },

            'FlatIVCDM_binned':
                    {
                        'model':   'FlatIVCDM_binned',
                        'names':   ['Om0', 'Ob0', 'sigma80'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.11370442e-01,  4.97572676e-02,  7.68854595e-01]
                                              + self.Nq*[0.0]
                                              + [1.34751053e-01,  4.24250978e-02,  8.70540584e-01,  3.07296559e+00,
                                             -1.94909839e-02,  5.08903425e-03, -1.91667145e+01,  1.14905470e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + self.Nq*((-1,1),) + self.bds_nuisance
                    },

            'FlatIVCDM_H0_binned':
                    {
                        'model':   'FlatIVCDM_binned',
                        'names':   ['Om0', 'Ob0', 'sigma80', 'H0'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.11370442e-01,  4.97572676e-02,  7.68854595e-01, 6.86910027e+01]
                                              + self.Nq*[0.0]
                                              + [1.34751053e-01,  4.24250978e-02,  8.70540584e-01,  3.07296559e+00,
                                             -1.94909839e-02,  5.08903425e-03, -1.91667145e+01,  1.14905470e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + ((60,80),) + self.Nq*((-1,1),) + self.bds_nuisance
                    },

            'FlatIVCDM_10bins': #optimize.minimize
                    {
                        'model':   'FlatIVCDM_binned',
                        'names':   ['Om0', 'Ob0', 'sigma80'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.18791800e-01,  4.89667367e-02,  7.77884776e-01]
                                          + [ 0.01605483, -0.00855738,  0.02230746, -0.02149422, -0.02002012,
                                              0.00321994, -0.00463525, -0.0038039 ,  0.00488536,  0.0058373 ]
                                          + [1.34609116e-01,
                                              4.20636967e-02,  8.70386375e-01,  3.07112420e+00, -1.91274179e-02,
                                              5.08485379e-03, -1.91639094e+01,  1.14865136e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + 10*((-1,1),) + self.bds_nuisance
                    },

            # 'FlatIVCDM_10bins': # mcmc mean values
            #         {
            #             'model':   'FlatIVCDM_binned',
            #             'names':   ['Om0', 'Ob0', 'sigma80'] + self.names_q + self.names_nuisance,
            #                'x0':   np.array([ 3.18791800e-01,  4.89667367e-02,  7.77884776e-01]
            #                               + [ 0.02, -0.01,  0.11, 0.18, -0.05,
            #                                   0.07, 0.14, 0.01,  2.0,  0.98]
            #                               + [1.34609116e-01,
            #                                   4.20636967e-02,  8.70386375e-01,  3.07112420e+00, -1.91274179e-02,
            #                                   5.08485379e-03, -1.91639094e+01,  1.14865136e-02]),
            #            'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + self.Nq*((-1,1),) + self.bds_nuisance
            #         },

            'IVCDM_binned':
                    {
                        'model':   'IVCDM_binned',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'sigma80'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.12389197e-01,  6.74010665e-01,  4.96837533e-02,  7.27767449e-01]
                                              + self.Nq*[0.0]
                                              + [1.34490125e-01,  4.17743497e-02,  8.70207632e-01,
                                                3.06946133e+00, -1.87531933e-02,  5.08025098e-03, -1.91617074e+01,
                                                1.14841389e-02]),
                       'bounds':   3*((0.01,0.99),) + ((0.3,1.3),) + self.Nq*((-1,1),) + self.bds_nuisance
                    },

            'IVCDM_H0_binned':
                    {
                        'model':   'IVCDM_binned',
                        'names':   ['Om0', 'Ol0', 'Ob0', 'sigma80', 'H0'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.12389197e-01,  6.74010665e-01,  4.96837533e-02,  7.27767449e-01, \
                                              6.86910027e+01]
                                              + self.Nq*[0.0]
                                              + [1.34490125e-01,  4.17743497e-02,  8.70207632e-01,
                                                3.06946133e+00, -1.87531933e-02,  5.08025098e-03, -1.91617074e+01,
                                                1.14841389e-02]),
                       'bounds':   3*((0.01,0.99),) + ((0.3,1.3),) + ((60,80),) + self.Nq*((-1,1),) + self.bds_nuisance
                    },

            'FlatIVCDM_smooth':
                    {
                        'model':   'FlatIVCDM_smooth',
                        'names':   ['Om0', 'Ob0', 'sigma80'] + self.names_q + self.names_nuisance,
                           'x0':   np.array([ 3.11370442e-01,  4.97572676e-02,  7.68854595e-01]
                                              + self.Nq*[0.0]
                                              + [1.34751053e-01,  4.24250978e-02,  8.70540584e-01,  3.07296559e+00,
                                             -1.94909839e-02,  5.08903425e-03, -1.91667145e+01,  1.14905470e-02]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + self.Nq*((-0.999,0.999),) + self.bds_nuisance
                    },

            'FlatIVCDM_3cheb': # MINUIT # 3 pars
                    {
                        'model':   'FlatIVCDM_smooth',
                        'names':   ['Om0', 'Ob0', 'sigma80', 'q1', 'q2', 'q3'] + self.names_nuisance,
                           'x0':   np.array([ 0.31437,  0.0496883,  0.836731]
                                              + [0.00465978, -0.0956699, -0.0207225]
                                              + [0.133148,  0.0381829,  0.86983,  3.05612,
                                             -0.0165006,  0.00506287, -19.1331,  0.0115217]),
                       'bounds':   2*((0.01,0.99),) + ((0.3,1.3),) + 3*((-0.999,0.999),) + self.bds_nuisance
                    }
        }    

    def get_pars_indices(self, pars_names, cosmo_name):
        M = self.model[cosmo_name]
        return [M['names'].index(n) for n in pars_names]

    def get_q_indices(self, cosmo_name):
        return self.get_pars_indices(self.names_q, cosmo_name)

    def get_model(self, cosmo_name, data_list=['BAO','CMB','RSD','SNIa','CC'], cosmo_pars_only=False):
        # strike out unconstrained parameters if using subset of datasets
        # data list is tuple of strings
        M = self.model[cosmo_name]
        names_nuis = self.names_nuisance
        names = M['names'] # original

        if 'RSD' not in data_list:
            i = M['names'].index('sigma80')
            M['names'].remove('sigma80')
            M['x0'] = np.delete(M['x0'],i)

        if 'SNIa' not in data_list or cosmo_pars_only:
            M['names'] = list(n for n in names if n not in names_nuis)
            M['x0'] = np.array([M['x0'][i] for i,n in enumerate(names) if n not in names_nuis])
            M['bounds'] = tuple(M['bounds'][i] for i,n in enumerate(names) if n not in names_nuis)
        return M
