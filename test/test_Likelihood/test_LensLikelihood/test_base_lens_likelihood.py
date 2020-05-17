import numpy as np
import pytest
import unittest
from hierarc.Likelihood.LensLikelihood.base_lens_likelihood import LensLikelihoodBase


class TestLensLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_lens = 0.8
        self.z_source = 3.0
        num_samples = 10000
        ddt_samples = np.random.normal(1, 0.1, num_samples)
        dd_samples = np.random.normal(1, 0.1, num_samples)

        self.likelihood_type_list = ['DdtGaussian',
                                'DdtDdKDE',
                                'DdtDdGaussian',
                                'DsDdsGaussian',
                                'DdtLogNorm',
                                'IFUKinCov',
                                'DdtHist',
                                'DdtHistKDE',
                                'DdtHistKin',
                                'DdtGaussKin']

        self.kwargs_likelihood_list = [{'ddt_mean': 1, 'ddt_sigma': 0.1},
                                  {'dd_samples': dd_samples, 'ddt_samples': ddt_samples, 'kde_type': 'scipy_gaussian', 'bandwidth': 1},
                                  {'ddt_mean': 1, 'ddt_sigma': 0.1, 'dd_mean': 1, 'dd_sigma': 0.1},
                                  {'ds_dds_mean': 1, 'ds_dds_sigma': 0.1},
                                  {'ddt_mu': 1, 'ddt_sigma': 0.1},
                                  {'sigma_v_measurement': [1], 'j_model': [1], 'error_cov_measurement': [[1]], 'error_cov_j_sqrt': [[1]]},
                                  {'ddt_samples': ddt_samples},
                                  {'ddt_samples': ddt_samples},
                                  {'ddt_samples': ddt_samples, 'sigma_v_measurement': [1], 'j_model': [1], 'error_cov_measurement': [[1]], 'error_cov_j_sqrt': [[1]]},
                                  {'ddt_mean': 1, 'ddt_sigma': 0.1, 'sigma_v_measurement': [1], 'j_model': [1], 'error_cov_measurement': [[1]], 'error_cov_j_sqrt': [[1]]},
                                  ]

    def test_log_likelihood(self):
        for i, likelihood_type in enumerate(self.likelihood_type_list):
            likelihood = LensLikelihoodBase(z_lens=self.z_lens, z_source=self.z_source, likelihood_type=likelihood_type,
                               **self.kwargs_likelihood_list[i])
            print(likelihood_type)
            logl = likelihood.log_likelihood(ddt=1, dd=1, aniso_scaling=None, sigma_v_sys_error=1)
            assert logl > -np.inf


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            LensLikelihoodBase(z_lens=0.5, z_source=2, likelihood_type='BAD')
        with self.assertRaises(ValueError):
            likelihood = LensLikelihoodBase(z_lens=0.5, z_source=2, likelihood_type='DdtGaussian',
                                            **{'ddt_mean': 1, 'ddt_sigma': 0.1})
            likelihood.likelihood_type = 'BAD'
            likelihood.log_likelihood(ddt=1, dd=1)


if __name__ == '__main__':
    pytest.main()
