import numpy as np
import unittest
np.random.seed(1)

import Surrogates.RegressionModels.model_util as model_util
import Surrogates.DataExtraction.configuration_space as configuration_space

class ModelUtilTest(unittest.TestCase):
    _checkpoint = None
    _data = None
    _para_header = None
    _sp = None

    def test_replace_cat(self):
        truth = np.array([0, 23], dtype="float64").reshape([1, 2])
        x = [['abc', '23']]
        x = np.array(x)
        catdict = {"a": {'abc': 0, 'abcd': 1}}
        param_names = ["a", "b"]
        ret = model_util.replace_cat_variables(x=x, catdict=catdict, param_names=param_names)
        self.assertEqual(ret.shape, truth.shape)
        self.assertEqual(ret.dtype, truth.dtype)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret[0, 0], truth[0, 0])
        self.assertEqual(ret[0, 1], truth[0, 1])

        x = [[1.7, 23]]
        x = np.array(x)
        catdict = {"a": {'1.7': 0, '2': 1}}
        param_names = ["a", "b", "c"]
        ret = model_util.replace_cat_variables(x=x, catdict=catdict, param_names=param_names)
        self.assertEqual(ret.shape, truth.shape)
        self.assertEqual(ret.dtype, truth.dtype)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret[0, 0], truth[0, 0])
        self.assertEqual(ret[0, 1], truth[0, 1])

        truth = np.array([[0, -2., -3., 1, 0], [1, -2., -4., 1, 0]], dtype="float64")
        catdict = {"enum_a": {"22": 0, "33": 1}, "constant": {"999": 0}}
        param_names = ["fold", "float_a", "int_a", "enum_a",  "constant"]
        x = [[0.0, -2, -3, "22", "999"], [1.0, -2, -4, "33", "999"]]
        ret = model_util.replace_cat_variables(x=x, catdict=catdict, param_names=param_names)
        self.assertEqual(ret.shape, truth.shape)
        self.assertEqual(ret.dtype, truth.dtype)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret[0, 0], truth[0, 0])
        self.assertEqual(ret[0, 1], truth[0, 1])

    def test_encode(self):
        float_a = configuration_space.UniformFloatHyperparameter("float_a", -5.3, 10.5)
        int_a = configuration_space.UniformIntegerHyperparameter("int_a", -5, 10)
        cat_a = configuration_space.CategoricalHyperparameter("enum_a", ["22", "33"])
        crazy = configuration_space.CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["999"])
        easy_space = {"float_a": float_a,
                      "int_a": int_a,
                      "enum_a": cat_a,
                      "constant": crazy,
                      }
        param_names = ["fold", "float_a", "int_a", "enum_a",  "constant"]
        catdict = {"enum_a": {"22": 0, "33": 1},
                   "constant": {"999": 0}}

        x = [[0.0, -2, -3, "22", "999"], [1.0, -2, -4, "33", "999"]]
        x = model_util.replace_cat_variables(x=x, catdict=catdict, param_names=param_names)
        ret, ret_names = model_util.encode(sp=easy_space, x=x, param_names=param_names, num_folds=2, catdict=catdict)
        thruth_names = ["fold_0", "fold_1", "float_a", "int_a", "enum_a_22", "enum_a_33", "constant_999"]
        thruth = np.array([[1.,  0., -2., -3.,  1.,  0.,  1.], [0.,  1., -2., -4.,  0.,  1.,  1.]], dtype="float64")
        self.assertListEqual(thruth_names, ret_names)
        self.assertEqual(ret.dtype, thruth.dtype)
        self.assertEqual(ret.shape, thruth.shape)
        self.assertIsInstance(ret, np.ndarray)
        for i in range(2):
            for j in range(7):
                self.assertEqual(ret[i, j], thruth[i, j])

    def test_encode_for_ArcGP(self):
        float_a = configuration_space.UniformFloatHyperparameter("float_a", -5.3, 10.5)
        int_a = configuration_space.UniformIntegerHyperparameter("int_a", -5, 10)
        cat_a = configuration_space.CategoricalHyperparameter("enum_a", ["22", "33"])
        crazy = configuration_space.CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["999"])
        easy_space = {"float_a": float_a,
                      "int_a": int_a,
                      "enum_a": cat_a,
                      "constant": crazy,
                      }
        param_names = ["fold", "float_a", "int_a", "enum_a",  "constant"]
        catdict = {"enum_a": {"22": 0, "33": 1},
                   "constant": {"999": 0}}
        x = [[0.0, -2, -3, "22", "999"], [1.0, -2, -4, "33", "999"]]
        x = model_util.replace_cat_variables(x=x, catdict=catdict, param_names=param_names)
        x_rel = np.array([[True, True, True, True, True], [False, True, True, False, False]])

        ret, ret_rel, ret_names = model_util.encode_for_ArcGP(sp=easy_space, x=x, rel_array=x_rel,
                                                                               param_names=param_names, num_folds=2,
                                                                               catdict=catdict)
        thruth_rel = np.array([[True, True, True, True, True, True, True],
                               [False, False, True, True, False, False, False]], dtype="bool")
        thruth_names = ["fold_0", "fold_1", "float_a", "int_a", "enum_a_22", "enum_a_33", "constant_999"]
        thruth = np.array([[1.,  0., -2., -3.,  1.,  0.,  1.], [0.,  1., -2., -4.,  0.,  1.,  1.]], dtype="float64")
        self.assertListEqual(thruth_names, ret_names)
        self.assertEqual(ret.dtype, thruth.dtype)
        self.assertEqual(ret.shape, thruth.shape)
        self.assertIsInstance(ret, np.ndarray)
        self.assertEqual(ret_rel.dtype, thruth_rel.dtype)
        self.assertEqual(ret_rel.shape, thruth_rel.shape)
        self.assertIsInstance(ret_rel, np.ndarray)
        print type(ret_rel)
        for i in range(2):
            for j in range(7):
                self.assertEqual(ret[i, j], thruth[i, j])
                self.assertEqual(ret_rel[i, j], thruth_rel[i, j])
