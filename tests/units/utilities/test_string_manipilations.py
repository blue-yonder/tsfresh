# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from unittest import TestCase

from tsfresh.utilities.string_manipulation import convert_to_output_format


class StringUtilities(TestCase):
    def test_convert_to_output_format(self):
        out = convert_to_output_format({"p1": 1, "p2": "a"})
        expected_out = 'p1_1__p2_"a"'
        self.assertEqual(out, expected_out)

        out = convert_to_output_format({"list": [1, 2, 4]})
        expected_out = "list_[1, 2, 4]"
        self.assertEqual(out, expected_out)

        out = convert_to_output_format({"list": ["a", "b", "c"]})
        expected_out = "list_['a', 'b', 'c']"
        self.assertEqual(out, expected_out)

    def test_convert_to_output_format_wrong_order(self):
        out = convert_to_output_format({"width": 1, "coeff": "a"})
        expected_out = 'coeff_"a"__width_1'
        self.assertEqual(out, expected_out)

        out = convert_to_output_format({"c": 1, "b": 2, "a": 3})
        expected_out = "a_3__b_2__c_1"
        self.assertEqual(out, expected_out)
