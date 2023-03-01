# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
import math
import unittest

import numpy as np
from yolino.model.loss import get_actual_weight
from yolino.runner.trainer import TrainHandler
from yolino.utils.enums import Dataset, LossWeighting
from yolino.utils.test_utils import test_setup

norm = lambda x, y: x + y
one = lambda x, y: 1


class KendallTest(unittest.TestCase):

    def test_learn(self):
        args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
            "loss_weight_strategy": str(LossWeighting.LEARN),
            "activations": "linear,sigmoid",
        })

        # we expect to provide the actual weighting factor to be as specified not sigma
        # so --weights = 1 / (2 * e^(s)); s = log(sigma^2) will be the training variable
        # so s = log(1 / (2 * --weights))
        # for exponential activation functions (conf) the factor 2 is to be neglected, so s = log(1 / --weights)
        for init_weight in range(1, 10):
            ########## CONF ###############
            c_weights = TrainHandler.__init_conf_loss_weights__("cpu", args.loss_weight_strategy,
                                                                init_weights=[init_weight, init_weight],
                                                                is_exponential=[True, True])
            self.validate_weight([1, 1], [init_weight, init_weight], is_grad=[True, True],
                                 weights=c_weights)

            ########## LOSS ###############
            l_weights = TrainHandler.__init_loss_weights__(cuda="cpu", loss_weighting=args.loss_weight_strategy,
                                                           num_train_vars=2,
                                                           init_weights=[init_weight, init_weight],
                                                           is_exponential=[False, True])
            is_grad = [True, False]
            self.validate_weight([1, 1], [init_weight, init_weight], is_grad, l_weights)

    def test_learn_log(self):
        for lws, fct in zip([LossWeighting.LEARN_LOG_NORM, LossWeighting.LEARN_LOG], [norm, one]):
            args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
                "loss_weight_strategy": str(lws),
                "activations": "linear,sigmoid",
            })

            # we expect to provide the actual weighting factor to be as specified not sigma
            # so --weights = 1 / (2 * e^(s)); s = log(sigma^2) will be the training variable
            # so s = log(1 / (2 * --weights))
            # for exponential activation functions (conf) the factor 2 is to be neglected, so s = log(1 / --weights)
            for init_weight in range(1, 10):
                ########## CONF ###############
                c_weights = TrainHandler.__init_conf_loss_weights__("cpu", args.loss_weight_strategy,
                                                                    init_weights=[init_weight, init_weight],
                                                                    is_exponential=[True, True])
                expected = math.log(1 / (init_weight / fct(init_weight, init_weight)))
                self.validate_weight([expected, expected], [init_weight, init_weight], is_grad=[True, True],
                                     weights=c_weights)

                ########## LOSS ###############
                l_weights = TrainHandler.__init_loss_weights__(cuda="cpu", loss_weighting=args.loss_weight_strategy,
                                                               num_train_vars=2,
                                                               init_weights=[init_weight, init_weight],
                                                               is_exponential=[False, True])
                wi = init_weight / fct(init_weight, init_weight)
                expected = [self.get_kendall_log_weight_factor(wi), 1]
                is_grad = [True, False]
                self.validate_weight(expected, [init_weight, init_weight], is_grad, l_weights)

    def test_fixed(self):
        test_setup(self._testMethodName, str(Dataset.ARGOVERSE2))

        for lws, fct in zip([LossWeighting.FIXED_NORM, LossWeighting.FIXED], [norm, one]):
            for first in range(1, 10, 2):
                for snd in range(1, 10, 3):
                    init_weights = [first, snd]
                    c_weights = TrainHandler.__init_conf_loss_weights__("cpu", lws,
                                                                        init_weights=init_weights,
                                                                        is_exponential=[True, True])
                    self.validate_weight([first / fct(first, snd), snd / fct(first, snd)],
                                         init_weight=init_weights, is_grad=[False, False], weights=c_weights)

                    l_weights = TrainHandler.__init_loss_weights__(cuda="cpu", loss_weighting=lws,
                                                                   num_train_vars=2,
                                                                   init_weights=init_weights,
                                                                   is_exponential=[False, True])

                    self.validate_weight([first / fct(first, snd), snd / fct(first, snd)],
                                         init_weight=init_weights, is_grad=[False, False], weights=l_weights)

    def test_loss_usage_learned_log(self):

        for lws, fct in zip([LossWeighting.LEARN_LOG_NORM, LossWeighting.LEARN_LOG], [norm, one]):
            args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
                "loss_weight_strategy": str(lws),
                "loss": ["mse_mean", "mse_mean"],
                "anchors": "none",
                "activations": "linear,sigmoid",
                "weights": [8, 2]
            })

            expected_factor = [args.weights[0] / fct(*args.weights), 0.36787945]
            expected_add = [0.5 * self.get_kendall_log_weight_factor(expected_factor[0]), 0.5]
            weight_needs_grad = [True, False]
            act_is_exp = [False, True]
            self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                         expected_factor=expected_factor, init_weights=args.weights,
                                         lws=args.loss_weight_strategy, is_conf=False,
                                         weight_needs_grad=weight_needs_grad)

    def test_conf_loss_usage_learned_log(self):

        for lws, fct in zip([LossWeighting.LEARN_LOG_NORM, LossWeighting.LEARN_LOG],
                            [norm, one]):
            args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
                "loss_weight_strategy": str(lws),
                "loss": ["mse_mean", "mse_mean"],
                "anchors": "none",
                "activations": "linear,sigmoid",
                "conf_match_weight": [2, 5]

            })
            expected_factor = np.asarray(args.conf_match_weight) / fct(*args.conf_match_weight)
            expected_add = 0.5 * np.log(1 / expected_factor)
            weight_needs_grad = [True, True]
            act_is_exp = [True, True]
            self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                         expected_factor=expected_factor, init_weights=args.conf_match_weight,
                                         lws=args.loss_weight_strategy, is_conf=True,
                                         weight_needs_grad=weight_needs_grad)

    def test_loss_usage_learned(self):

        args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
            "loss_weight_strategy": str(LossWeighting.LEARN),
            "loss": ["mse_mean", "mse_mean"],
            "anchors": "none",
            "activations": "linear,sigmoid",
            "weights": [8, 2]
        })

        expected_factor = [0.5, 1]
        expected_add = [math.log(1), 0]
        weight_needs_grad = [True, False]
        act_is_exp = [False, True]
        self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                     expected_factor=expected_factor, init_weights=[1, 1],
                                     lws=args.loss_weight_strategy, is_conf=False,
                                     weight_needs_grad=weight_needs_grad)

    def test_conf_loss_usage_learned(self):
        args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
            "loss_weight_strategy": str(LossWeighting.LEARN),
            "loss": ["mse_mean", "mse_mean"],
            "anchors": "none",
            "activations": "linear,sigmoid",
            "conf_match_weight": [2, 5]

        })
        expected_factor = [1, 1]
        expected_add = [math.log(1), math.log(1)]
        weight_needs_grad = [True, True]
        act_is_exp = [True, True]
        self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                     expected_factor=expected_factor, init_weights=args.conf_match_weight,
                                     lws=args.loss_weight_strategy, is_conf=True,
                                     weight_needs_grad=weight_needs_grad)

    def test_loss_usage_fixed(self):

        for lws, fct in zip([LossWeighting.FIXED_NORM, LossWeighting.FIXED], [norm, one]):
            args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
                "loss_weight_strategy": str(lws),
                "loss": ["mse_mean", "mse_mean"],
                "anchors": "none",
                "activations": "linear,sigmoid",
                "weights": [10, 1],
                "conf_match_weight": [8, 1]
            })

            expected_add = [0, 0]
            expected_factor = np.asarray(args.weights) / fct(*args.weights)
            weight_needs_grad = [False, False]
            act_is_exp = [False, True]
            self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                         expected_factor=expected_factor, init_weights=args.weights,
                                         lws=args.loss_weight_strategy, is_conf=False,
                                         weight_needs_grad=weight_needs_grad)

    def test_conf_loss_usage_fixed(self):

        for lws, fct in zip([LossWeighting.FIXED_NORM, LossWeighting.FIXED], [norm, one]):
            args = test_setup(self._testMethodName, str(Dataset.ARGOVERSE2), additional_vals={
                "loss_weight_strategy": str(lws),
                "loss": ["mse_mean", "mse_mean"],
                "anchors": "none",
                "activations": "linear,sigmoid",
                "weights": [10, 1],
                "conf_match_weight": [8, 1]

            })

            expected_add = [0, 0]
            expected_factor = np.asarray(args.conf_match_weight) / fct(*args.conf_match_weight)
            weight_needs_grad = [False, False]
            act_is_exp = [True, True]
            self.validate_actual_weights(act_is_exp=act_is_exp, expected_add=expected_add,
                                         expected_factor=expected_factor, init_weights=args.conf_match_weight,
                                         lws=args.loss_weight_strategy, is_conf=True,
                                         weight_needs_grad=weight_needs_grad)

    def get_kendall_log_weight_factor(self, wi):
        return math.log(1 / (2 * wi))

    def validate_weight(self, expected, init_weight, is_grad, weights):
        for i in range(0, 2):
            self.assertAlmostEqual(float(weights[i]), expected[i], places=5,
                                   msg=f"{i}: Initialize loss weight with {init_weight[i]} should lead to "
                                       f"trainable weight w={expected[i]}, but w={weights[i]}")

            if type(weights[i]) == int:
                self.assertFalse(is_grad[i])
            else:
                self.assertEqual(weights[i].requires_grad, is_grad[i])

    def validate_actual_weights(self, act_is_exp, expected_add, expected_factor, init_weights, lws, weight_needs_grad,
                                is_conf=True):
        if is_conf:
            weights = TrainHandler.__init_conf_loss_weights__("cpu", lws, init_weights=init_weights,
                                                              is_exponential=act_is_exp)
        else:
            weights = TrainHandler.__init_loss_weights__(cuda="cpu", loss_weighting=lws, num_train_vars=2,
                                                         init_weights=init_weights, is_exponential=act_is_exp)
        # geom / conf0
        add_weight1, weight_factor1 = get_actual_weight(epoch=0, variable_str="bla", weight_strategy=lws,
                                                        weight=weights[0], activation_is_exponential=act_is_exp[0])
        # conf / conf1
        add_weight2, weight_factor2 = get_actual_weight(epoch=0, variable_str="bla", weight_strategy=lws,
                                                        weight=weights[1], activation_is_exponential=act_is_exp[1])
        self.validate_weight(expected_add, init_weight=weights,
                             is_grad=weight_needs_grad, weights=[add_weight1, add_weight2])
        self.validate_weight(expected_factor, init_weight=weights, is_grad=weight_needs_grad,
                             weights=[weight_factor1, weight_factor2])


if __name__ == '__main__':
    unittest.main()
