import numpy as np
import unittest
np.random.seed(1)

import Surrogates.DataExtraction.handle_configurations as handle_configurations
import Surrogates.DataExtraction.configuration_space as configuration_space

class RemoveInactive(unittest.TestCase):

    def test_encode(self):
        float_a = configuration_space.UniformFloatHyperparameter("float_a", -5.3, 10.5)
        int_a = configuration_space.UniformIntegerHyperparameter("int_a", -5, 10)
        cat_a = configuration_space.CategoricalHyperparameter("enum_a", ["22", "33"])
        crazy = configuration_space.CategoricalHyperparameter("crazy", ["999"])
        easy_space = {"float_a": float_a,
                      "int_a": int_a,
                      "enum_a": cat_a,
                      "constant": crazy,
                      }

        # No condition
        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "33", 'constant': '999'}
        cond = handle_configurations.get_cond_dict(easy_space)
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        truth = {"float_a": -2.3, "int_a": -3, "enum_a": "33", "constant": "999"}
        self.assertDictEqual(ret, truth)

        # One condition
        cond = {"float_a": [["enum_a", "33"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        cond = {"float_a": [["enum_a", "22"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": "33", "constant": "999"}
        self.assertDictEqual(ret, truth)

        easy = {"int_a": -3, "enum_a": "33", 'constant': '999'}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": "33", "constant": "999"}
        self.assertDictEqual(ret, truth)

        # Missing stuff
        easy = {"int_a": -3, "enum_a": "22", 'constant': '999'}
        self.assertRaises(KeyError, handle_configurations.remove_inactive, clean_dict=easy, cond=cond)
        easy = {"float_a": -2.3, "int_a": -3}
        self.assertRaises(ValueError, handle_configurations.remove_inactive, clean_dict=easy, cond=cond)

        # Two conditions
        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "33", 'constant': "999"}
        cond = {"float_a": [["enum_a", "22"], ["constant", "999"]]}
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": "33", "constant": "999"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "33", 'constant': "666"}
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": "33", "constant": "666"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "22", 'constant': "999"}
        truth = {"float_a": -2.3, "int_a": -3, "enum_a": "22", "constant": "999"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        # Nested conditions
        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "33", 'constant': "999"}
        cond = {"float_a": [["enum_a", "22"], ["constant", "999"]], "enum_a": [["constant", "999"]]}
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": "33", "constant": "999"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        cond = {"float_a": [["enum_a", "22"], ["constant", ["999", ]]], "enum_a": [["constant", "999"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        cond = {"float_a": [["enum_a", "22"], ["constant", ["999", "662"]]], "enum_a": [["constant", "999"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"float_a": -2.3, "int_a": -3, "enum_a": "33", 'constant': "662"}
        cond = {"float_a": [["enum_a", "22"], ["constant", "999"]], "enum_a": [["constant", "999"]]}
        truth = {"float_a": np.nan, "int_a": -3, "enum_a": np.nan, "constant": "662"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"enum_a": "33", 'constant': "662", "float_a": -2.3, "int_a": -3}
        cond = {"enum_a": [["constant", "999"]], "float_a": [["enum_a", "22"], ["constant", "999"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"enum_a": "33", 'constant': "662", "float_a": "-2.3", "int_a": -3}
        cond = {"enum_a": [["constant", "999"]], "float_a": [["enum_a", "22"], ["constant", "999"]]}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

        easy = {"enum_a": "22", 'constant': "999", "float_a": -2.3, "int_a": -3}
        cond = {"enum_a": [["constant", "999"]], "float_a": [["enum_a", "22"], ["constant", "999"]]}
        truth = {"float_a": -2.3, "int_a": -3, "enum_a": "22", "constant": "999"}
        ret = handle_configurations.remove_inactive(clean_dict=easy, cond=cond)
        self.assertDictEqual(ret, truth)

