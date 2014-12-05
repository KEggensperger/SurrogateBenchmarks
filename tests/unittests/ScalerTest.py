import os
import unittest

import numpy
numpy.random.seed(1)

import Surrogates.RegressionModels.scaler
import Surrogates.RegressionModels.model_util
import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv
from Surrogates.DataExtraction.handle_configurations import get_cat_val_map


class ScalerTest(unittest.TestCase):
    _checkpoint = None
    _data = None
    _para_header = None
    _sp = None

    def setUp(self):
        self._sp = pcs_parser.read(file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Testdata/nips2011.pcs")))
        # Read data from csv
        header, self._data = read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Testdata/hpnnet_nocv_convex_all_fastrf_results.csv"),
                                      has_header=True, num_header_rows=3)
        self._para_header = header[0][:-2]
        self._data = self._data[:1000, :-2]
        self._data = Surrogates.RegressionModels.model_util.replace_cat_variables(catdict=get_cat_val_map(sp=self._sp),
                                                                                  x=self._data,
                                                                                  param_names=self._para_header)

    def test_scale(self):
        scale_info = Surrogates.RegressionModels.scaler.get_x_info_scaling_all(sp=self._sp, x=self._data,
                                                                               param_names=self._para_header,
                                                                               num_folds=1, encoded=False)

        should_be_lower = [None, -29.6210089736, 0.201346561323, 0, -20.6929600285, 0, 0, 0, 4.60517018599, 0,
                           2.77258872224, 0, 0, 0.502038871605, -17.2269829469]
        should_be_upper = [None, -7.33342451433, 1.99996215592, 1, -6.92778489957, 2, 1, 1, 9.20883924585, 1,
                           6.9314718056, 3, 1, 0.998243871085, 4.72337617503]

        for idx in range(self._data.shape[1]):
            self.assertEqual(scale_info[0][idx], should_be_lower[idx])
            self.assertEqual(scale_info[1][idx], should_be_upper[idx])
        print self._data[0:1, :]

        scaled = Surrogates.RegressionModels.scaler.scale(scale_info=scale_info, x=self._data[0:1, :])[0]
        should_be_scaled = [0., 0.70916157, 0.49963619, 1., 0.49962673, 0., 0., 0., 0.17997641, 0., 0.8548805,
                            0.33333333, 0., 0.49971509, 0.44197703]
        for idx in range(self._data.shape[1]):
            self.assertAlmostEqual(should_be_scaled[idx], scaled[idx])

    def test_simple_scale(self):
        float_a = Surrogates.DataExtraction.configuration_space.UniformFloatHyperparameter("float_a", -5.3, 10.5)
        int_a = Surrogates.DataExtraction.configuration_space.UniformIntegerHyperparameter("int_a", -5, 10)
        cat_a = Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter("enum_a", ["22", "33", "44"])
        crazy = Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["999"])
        easy_space = {"float_a": float_a,
                      "int_a": int_a,
                      "enum_a": cat_a,
                      "constant": crazy,
                      }
        param_names = ["fold", "float_a", "int_a", "enum_a",  "constant"]

        x = numpy.array([[0.0, -5, -3, 0, 0], [1.0, -2, -4, 1, 0]], dtype="float64")

        lower, upper = Surrogates.RegressionModels.scaler.get_x_info_scaling_all(sp=easy_space, x=x,
                                                                                 param_names=param_names,
                                                                                 num_folds=2, encoded=False)

        thruth_lower = [0, -5, -4, 0, 0]
        thruth_upper = [1, -2, -3, 2, 1]
        self.assertListEqual(lower, thruth_lower)
        self.assertListEqual(upper, thruth_upper)

        ret_scaled = Surrogates.RegressionModels.scaler.scale(scale_info=(lower, upper), x=x)
        thruth_scaled = numpy.array([[0, 0, 1, 0, 0], [1, 1, 0, 0.5, 0]], dtype="float64")

        self.assertEqual(thruth_scaled.dtype, ret_scaled.dtype)
        self.assertEqual(thruth_scaled.shape, ret_scaled.shape)
        self.assertIsInstance(ret_scaled, numpy.ndarray)
        for i in range(2):
            for j in range(5):
                self.assertEqual(ret_scaled[i, j], thruth_scaled[i, j])

    def test_complex_scale(self):
        float_a = Surrogates.DataExtraction.configuration_space.UniformFloatHyperparameter("float_a", -5.3, 10.5)
        int_a = Surrogates.DataExtraction.configuration_space.UniformIntegerHyperparameter("int_a", -5, 10)
        cat_a = Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter("enum_a", ["22", "33", "44"])
        crazy = Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["999"])
        easy_space = {"float_a": float_a,
                      "int_a": int_a,
                      "enum_a": cat_a,
                      "constant": crazy,
                      }
        param_names = ["fold", "float_a", "int_a", "enum_a",  "constant"]

        x = numpy.array([[0.0, -5, -3, "22", "999"], [1.0, -2, -4, "33", "999"]], dtype="float64")

        lower, upper = Surrogates.RegressionModels.scaler.get_x_info_scaling_no_categorical(sp=easy_space, x=x,
                                                                                            param_names=param_names,
                                                                                            num_folds=2)

        thruth_lower = [None, -5, -4, None, None]
        thruth_upper = [None, -2, -3, None, None]
        self.assertListEqual(lower, thruth_lower)
        self.assertListEqual(upper, thruth_upper)

        ret_scaled = Surrogates.RegressionModels.scaler.scale(scale_info=(lower, upper), x=x)
        thruth_scaled = numpy.array([[0, 0, 1, "22", "999"], [1, 1, 0, "33", "999"]], dtype="float64")

        self.assertEqual(thruth_scaled.dtype, ret_scaled.dtype)
        self.assertEqual(thruth_scaled.shape, ret_scaled.shape)
        self.assertIsInstance(ret_scaled, numpy.ndarray)
        for i in range(2):
            for j in range(5):
                self.assertEqual(ret_scaled[i, j], thruth_scaled[i, j])