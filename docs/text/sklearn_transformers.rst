.. _sklearn-transformers-label:

scikit-learn Transformers
=========================

tsfresh includes three scikit-learn compatible transformers, which allow you to easily incorporate feature extraction
and feature selection from time series into your existing machine learning pipelines.

The scikit-learn pipeline allows you to assemble several pre-processing steps that will be executed in sequence and
thus, can be cross-validated together while setting different parameters (for more details about the scikit-learn's
pipeline, take a look at the official documentation [1]_).
Our tsfresh transformers allow you to extract and filter the time series features during these pre-processing sequence.

The first two estimators in tsfresh are the :class:`~tsfresh.transformers.feature_augmenter.FeatureAugmenter`,
which extracts the features, and the :class:`~tsfresh.transformers.feature_selector.FeatureSelector`, which
performs the feature selection algorithm.
It is preferable to combine extracting and filtering of the features in a single step to avoid unnecessary feature
calculations.
Hence, the :class:`~tsfresh.transformers.feature_augmenter.RelevantFeatureAugmenter` combines both the
extraction and filtering of the features in a single step.

Example
-------

In the following example you see how we combine tsfresh's
:class:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter` and a
:class:`~sklearn.ensemble.RandomForestClassifier` into a single pipeline. This pipeline can then fit both our
transformer and the classifier in one step.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from tsfresh.examples import load_robot_execution_failures
    from tsfresh.transformers import RelevantFeatureAugmenter
    import pandas as pd

    # Download dataset
    from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
    download_robot_execution_failures()

    pipeline = Pipeline([
                ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                ('classifier', RandomForestClassifier()),
                ])

    df_ts, y = load_robot_execution_failures()
    X = pd.DataFrame(index=y.index)

    pipeline.set_params(augmenter__timeseries_container=df_ts)
    pipeline.fit(X, y)

The parameters of the :class:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter` correspond to
the parameters of the top-level convenience function
:func:`~tsfresh.convenience.relevant_extraction.extract_relevant_features`.
In the above example, we only set the names of two columns ``column_id='id'``, ``column_sort='time'``
(see :ref:`data-formats-label` for more details on those parameters).

Because we cannot pass the time series container directly as a parameter to the augmenter step when calling fit or
transform on a :class:`sklearn.pipeline.Pipeline`, we have to set it manually by calling
``pipeline.set_params(augmenter__timeseries_container=df_ts)``.
In general, you can change the time series container from which the features are extracted by calling either the
pipeline's :func:`~sklearn.pipeline.Pipeline.set_params` method or the transformers
:func:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter.set_timeseries_container` method.

For further examples, visit the Jupyter Notebook 02 sklearn Pipeline.ipynb in the notebooks folder of the tsfresh
github repository.


References
----------

    .. [1] http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
