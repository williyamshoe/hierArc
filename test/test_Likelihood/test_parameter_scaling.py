import pytest
import numpy as np
import unittest
from hierarc.Likelihood.parameter_scaling import (
    ParameterScalingSingleAperture,
    ParameterScalingIFU,
)


class TestParameterScalingSingleAperture(object):
    def setup_method(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        param_scaling_array = ani_param_array * 2
        self.scaling = ParameterScalingSingleAperture(
            ani_param_array, param_scaling_array
        )

        ani_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        self.scaling_2d = ParameterScalingSingleAperture(
            ani_param_array, param_scaling_array
        )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.outer(gamma_in_array, log_m2l_array),
        )
        self.scaling_nfw = ParameterScalingSingleAperture(
            param_arrays, param_scaling_array
        )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
            log_m2l_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                np.outer(gamma_in_array, log_m2l_array),
            ),
        )

        self.scaling_nfw_2d = ParameterScalingSingleAperture(
            param_arrays, param_scaling_array
        )

    def test_param_scaling(self):
        scaling = self.scaling.param_scaling(param_array=[1])
        assert scaling == np.array([2])

        scaling = self.scaling.param_scaling(param_array=None)
        assert scaling == 1

        scaling = self.scaling_2d.param_scaling(param_array=[1, 2])
        assert scaling == 2

        scaling = self.scaling_nfw.param_scaling(param_array=[1, 2.9, 0.5])
        assert scaling == 1 * 2.9 * 0.5

        scaling = self.scaling_nfw_2d.param_scaling(param_array=[1, 2, 2.9, 0.5])
        assert scaling == 1 * 2 * 2.9 * 0.5


class TestParameterScalingIFU(object):
    def setup_method(self):
        ani_param_array = np.linspace(start=0, stop=1, num=10)
        param_scaling_array = ani_param_array * 2
        self.scaling = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=ani_param_array,
            scaling_grid_list=[param_scaling_array],
        )

        ani_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        self.scaling_2d = ParameterScalingIFU(
            anisotropy_model="GOM",
            param_arrays=ani_param_array,
            scaling_grid_list=[param_scaling_array],
        )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.outer(gamma_in_array, log_m2l_array),
        )
        self.scaling_nfw = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
            log_m2l_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                np.outer(gamma_in_array, log_m2l_array),
            ),
        )
        self.scaling_nfw_2d = ParameterScalingIFU(
            anisotropy_model="GOM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )

        param_arrays = [ani_param_array, gamma_in_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            gamma_in_array,
        )
        self.scaling_nfw_no_m2l = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                gamma_in_array,
            ),
        )
        self.scaling_nfw_2d_no_m2l = ParameterScalingIFU(
            anisotropy_model="GOM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )

    def test_param_scaling(self):
        scaling = self.scaling.param_scaling(param_array=None)
        assert scaling[0] == 1

        scaling = self.scaling.param_scaling(param_array=[1])
        assert scaling[0] == 2

        scaling = self.scaling_2d.param_scaling(param_array=[1, 2])
        assert scaling[0] == 2

        scaling = self.scaling_nfw.param_scaling(param_array=[1, 2.9, 0.5])
        assert scaling[0] == 1 * 2.9 * 0.5

        scaling = self.scaling_nfw_2d.param_scaling(param_array=[1, 2, 2.9, 0.5])
        assert scaling[0] == 1 * 2 * 2.9 * 0.5

        scaling = self.scaling_nfw_no_m2l.param_scaling(param_array=[1, 2.9])
        assert scaling[0] == 1 * 2.9

        scaling = self.scaling_nfw_2d_no_m2l.param_scaling(param_array=[1, 2, 2.9])
        assert scaling[0] == 1 * 2 * 2.9

    def test_draw_anisotropy(self):
        a_ani = 1
        beta_inf = 1.5
        param_draw = self.scaling.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        for i in range(100):
            param_draw = self.scaling.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )
        self.scaling._anisotropy_model = "const"
        param_draw = self.scaling.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == 1

        param_draw = self.scaling_2d.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        assert param_draw[1] == beta_inf
        for i in range(100):
            param_draw = self.scaling_2d.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )

        param_draw = self.scaling_nfw.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        for i in range(100):
            param_draw = self.scaling_nfw.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )

        param_draw = self.scaling_nfw_2d.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        assert param_draw[1] == beta_inf
        for i in range(100):
            param_draw = self.scaling_nfw_2d.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )

        param_draw = self.scaling_nfw_no_m2l.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        for i in range(100):
            param_draw = self.scaling_nfw_no_m2l.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )

        param_draw = self.scaling_nfw_2d_no_m2l.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw[0] == a_ani
        assert param_draw[1] == beta_inf
        for i in range(100):
            param_draw = self.scaling_nfw_2d_no_m2l.draw_anisotropy(
                a_ani=1, a_ani_sigma=1, beta_inf=beta_inf, beta_inf_sigma=1
            )

        scaling = ParameterScalingIFU(anisotropy_model="NONE")
        param_draw = scaling.draw_anisotropy(
            a_ani=1, a_ani_sigma=0, beta_inf=beta_inf, beta_inf_sigma=0
        )
        assert param_draw is None

    def test_draw_lens_parameters(self):
        param_draw = self.scaling_nfw.draw_lens_parameters(
            gamma_in=1, gamma_in_sigma=0, log_m2l=0.5, log_m2l_sigma=0
        )
        assert param_draw[0] == 1
        assert param_draw[1] == 0.5
        for i in range(100):
            param_draw = self.scaling_nfw.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=1, log_m2l=0.5, log_m2l_sigma=3
            )

        param_draw = self.scaling_nfw_2d.draw_lens_parameters(
            gamma_in=1, gamma_in_sigma=0, log_m2l=0.5, log_m2l_sigma=0
        )
        assert param_draw[0] == 1
        assert param_draw[1] == 0.5

        for i in range(100):
            param_draw = self.scaling_nfw_2d.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=1, log_m2l=0.5, log_m2l_sigma=3
            )

        param_draw = self.scaling_nfw_no_m2l.draw_lens_parameters(
            gamma_in=1, gamma_in_sigma=0, log_m2l=0.5, log_m2l_sigma=0
        )
        assert param_draw == 1
        for i in range(100):
            param_draw = self.scaling_nfw_no_m2l.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=1, log_m2l=0.5, log_m2l_sigma=3
            )

        param_draw = self.scaling_nfw_2d_no_m2l.draw_lens_parameters(
            gamma_in=1, gamma_in_sigma=0, log_m2l=0.5, log_m2l_sigma=0
        )
        assert param_draw == 1
        for i in range(100):
            param_draw = self.scaling_nfw_2d_no_m2l.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=1, log_m2l=0.5, log_m2l_sigma=3
            )


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            ParameterScalingIFU(
                anisotropy_model="blabla",
                param_arrays=np.array([0, 1]),
                scaling_grid_list=[np.array([0, 1])],
            )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        ani_scaling_array = ani_param_array * 2
        scaling = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=ani_param_array,
            scaling_grid_list=[ani_scaling_array],
        )
        with self.assertRaises(ValueError):
            scaling.draw_anisotropy(
                a_ani=-1, a_ani_sigma=0, beta_inf=-1, beta_inf_sigma=0
            )

        ani_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]
        ani_scaling_array = np.outer(ani_param_array[0], ani_param_array[1])
        scaling = ParameterScalingIFU(
            anisotropy_model="GOM",
            param_arrays=ani_param_array,
            scaling_grid_list=[ani_scaling_array],
        )
        with self.assertRaises(ValueError):
            scaling.draw_anisotropy(
                a_ani=0.5, a_ani_sigma=0, beta_inf=-1, beta_inf_sigma=0
            )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.multiply.outer(gamma_in_array, log_m2l_array),
        )
        print(param_scaling_array.shape)
        scaling = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )
        with self.assertRaises(ValueError):
            scaling.draw_lens_parameters(
                gamma_in=-1, gamma_in_sigma=0.1, log_m2l=0.5, log_m2l_sigma=0.1
            )
        with self.assertRaises(ValueError):
            scaling.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=0.1, log_m2l=-0.5, log_m2l_sigma=0.1
            )

        ani_param_array = np.linspace(start=0, stop=1, num=10)
        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)
        log_m2l_array = np.linspace(start=0.1, stop=1, num=10)

        param_arrays = [ani_param_array, gamma_in_array, log_m2l_array]
        param_scaling_array = np.multiply.outer(
            ani_param_array,
            np.multiply.outer(gamma_in_array, log_m2l_array),
        )

        scaling = ParameterScalingIFU(
            anisotropy_model="OM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )
        with self.assertRaises(ValueError):
            scaling.draw_lens_parameters(
                gamma_in=1, gamma_in_sigma=0, log_m2l=0, log_m2l_sigma=0
            )

        gom_param_array = [
            np.linspace(start=0, stop=1, num=10),
            np.linspace(start=1, stop=2, num=5),
        ]

        gamma_in_array = np.linspace(start=0.1, stop=2.9, num=5)

        param_arrays = [
            gom_param_array[0],
            gom_param_array[1],
            gamma_in_array,
        ]
        param_scaling_array = np.multiply.outer(
            gom_param_array[0],
            np.multiply.outer(
                gom_param_array[1],
                gamma_in_array,
            ),
        )
        self.scaling_nfw_2d_no_m2l = ParameterScalingIFU(
            anisotropy_model="GOM",
            param_arrays=param_arrays,
            scaling_grid_list=[param_scaling_array],
        )

        with self.assertRaises(ValueError):
            param_draw = self.scaling_nfw_2d_no_m2l.draw_lens_parameters(
                gamma_in=-1, gamma_in_sigma=1, log_m2l=0.5, log_m2l_sigma=3
            )


if __name__ == "__main__":
    pytest.main()