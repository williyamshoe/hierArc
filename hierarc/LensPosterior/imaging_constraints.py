import copy

import numpy as np
            
from scipy.special import gamma, gammainc
from scipy.optimize import brentq


class ImageModelPosterior(object):
    """Class to manage lens and light model posteriors inferred from imaging data."""

    def __init__(self, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error, lens_light_model_list=None, kwargs_lens_light=None, kwargs_lens_light_error=None):
        """

        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        
        :param lens_light_model_list:
        :param kwargs_lens_light:
        :param kwargs_lens_light_error:
        """
        self._theta_E, self._theta_E_error = theta_E, theta_E_error
        self._gamma, self._gamma_error = gamma, gamma_error
        self._r_eff, self._r_eff_error = r_eff, r_eff_error
        self._lens_light_model_list = lens_light_model_list
        self._kwargs_lens_light = kwargs_lens_light
        self._kwargs_lens_light_error = kwargs_lens_light_error
        
    def _flux(self, amp, r_s, n, r):
        """
        get flux within radius r for given sersic parameterization
        :param amp: Amplitude
        :param r_s: sersic radius (half light radius)
        :param n: n sersic parameter
        :param r: radius to intergrate over
        :return: flux
        """
        bn = 1.9992 * n - 0.3271
        x = bn * (r/r_s)**(1./n)
        return np.abs(amp) * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gammainc(2*n, x) * gamma(2*n)

    def _total_flux(self, amp, r_s, n):
        """
        get total flux within for given sersic parameterization
        :param amp: Amplitude
        :param r_s: sersic radius (half light radius)
        :param n: n sersic parameter
        :return: total flux
        """
        bn = 1.9992 * n - 0.3271
        return np.abs(amp) * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gamma(2*n)
        
    def get_r_eff(self, kwargs_lens_light_sample):
        """

        :param kwargs_lens_light_sample: array, keyword argument list of lens light model
        :return: r_eff
        """
        if len(self._lens_light_model_list) == 1 and "SERSIC" in self._lens_light_model_list[0]:
            return kwargs_lens_light_sample[0]['r_sersic']
        if len(self._lens_light_model_list) == 1 and "HERNQUIST" in self._lens_light_model_list[0]:
            return kwargs_lens_light_sample[0]['Rs'] / 0.551
        
        if np.sum(["SERSIC" != i for i in self._lens_light_model_list]) > 0:
            raise ValueError("light model %s not valid/implemented." % self._lens_light_model_list)
        
        tot_flux = np.sum([self._total_flux(klls['amp'], klls['R_sersic'], klls['n_sersic']) for klls in kwargs_lens_light_sample])

        def min_fnc(lim):
            val = np.sum([self._flux(klls['amp'], klls['R_sersic'], klls['n_sersic'], lim) for klls in kwargs_lens_light_sample])
            return (val - tot_flux/2)

        hlr = brentq(min_fnc, 0.01, 10)
        return hlr
    
    def draw_lens_original(self, no_error=False):
        """

        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: theta_E, gamma, r_eff, delta_r_eff
        """
        if no_error is True:
            return self._theta_E, self._gamma, self._r_eff, 1
        theta_E_draw = np.maximum(
            np.random.normal(loc=self._theta_E, scale=self._theta_E_error), 0
        )
        gamma_draw = np.random.normal(loc=self._gamma, scale=self._gamma_error)
        # distributions are drawn in the range [1, 3)
        # the power-law slope gamma=3 is divergent in mass in the center and values close close to =3 may be unstable
        # to compute the kinematics for.
        gamma_draw = np.maximum(gamma_draw, 1.0)
        gamma_draw = np.minimum(gamma_draw, 2.999)
        # we make sure no negative r_eff are being sampled
        delta_r_eff = np.maximum(
            np.random.normal(loc=1, scale=self._r_eff_error / self._r_eff), 0.001
        )
        r_eff_draw = delta_r_eff * self._r_eff
        return theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff
    
    def draw_lens_light(self, no_error=False):
        """
        Samples the light profile
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: sampled kwargs_lens_light
        """
        light_samples = []
        for i, kll in enumerate(self._kwargs_lens_light):
            kll_drawn = copy.deepcopy(kll)
            for model_param in kll:
                if no_error is True:
                    drawn_val = kll[model_param]
                else:
                    drawn_val = np.random.normal(kll[model_param], self._kwargs_lens_light_error[i][model_param])
                kll_drawn[model_param] = drawn_val
            light_samples += [kll_drawn]
        return np.array(light_samples)
                                                      
    def draw_lens(self, no_error=False, draw_light_profile=False):
        """

        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: theta_E, gamma, r_eff, delta_r_eff, kwargs_lens_light_draw
        """
        
        if self._kwargs_lens_light_error is None or draw_light_profile is False:
            return self.draw_lens_original(no_error=no_error)
        
        theta_E_draw, gamma_draw, _, _ = self.draw_lens_original(no_error=no_error)
        
        light_samples = self.draw_lens_light(no_error=no_error)

        r_eff_draw = self.get_r_eff(light_samples)
        delta_r_eff = r_eff_draw / self._r_eff
        
        return theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff, light_samples
    
########################################################################################################
    
    def draw_lens_nfw(self, no_error=False, draw_light_profile=False):
        """
        TODO: Given a the lens and lens light model distribution, get the NFW parameterization distribution (if necessary) and sample from that.
                    -Make use of priors to estimate NFW distribution, then cache the result to avoid that everytime
            
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        :return: theta_E, gamma, r_eff, delta_r_eff, kwargs_lens_light_draw
        """
        
        if self._kwargs_lens_light_error is None or draw_light_profile is False:
            return self.draw_lens_original(no_error=no_error)
        
        theta_E_draw, gamma_draw, _, _ = self.draw_lens_original(no_error=no_error)
        
        light_samples = self.draw_lens_light(no_error=no_error)

        r_eff_draw = self.get_r_eff(light_samples)
        delta_r_eff = r_eff_draw / self._r_eff
        
        return theta_E_draw, gamma_draw, r_eff_draw, delta_r_eff, light_samples
