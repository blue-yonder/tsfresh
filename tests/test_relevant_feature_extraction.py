# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

from __future__ import absolute_import, division
from tests.fixtures import DataTestCase
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.feature_extraction.settings import FeatureExtractionSettings
from tsfresh.utilities.dataframe_functions import impute


# todo: add more unit tests
class RelevantFeatureExtractionTestCase(DataTestCase):
    """
    Test case for the relevant_feature_extraction function
    """

    def test_functional_equality(self):
        """
        `extract_relevant_features` should be equivalent to running first `extract_features` with impute and
        `select_features` afterwards.
        Meaning it should produce the same relevant features and the values of these features should be identical.
        :return:
        """
        df, y = self.create_test_data_sample_with_target()

        relevant_features = extract_relevant_features(df, y, column_id='id', column_value='val', column_kind='kind',
                                                      column_sort='sort')

        extraction_settings = FeatureExtractionSettings()
        extraction_settings.IMPUTE = impute
        extracted_features = extract_features(df, feature_extraction_settings=extraction_settings, column_id='id',
                                              column_value='val', column_kind='kind', column_sort='sort')
        selected_features = select_features(extracted_features, y)

        self.assertEqual(set(relevant_features.columns), set(selected_features.columns),
                         "Should select the same columns:\n\t{}\n\nvs.\n\n\t{}".format(relevant_features.columns,
                                                                                       selected_features.columns))
        self.assertTrue((relevant_features.values == selected_features.values).all().all(),
                        "Should calculate the same feature values")
