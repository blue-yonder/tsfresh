.. _sklearn-transformers-label:

scikit-learn Transformers
=========================

tsfresh includes three scikit-learn compatible transformers.
You can easily add them to your existing data science pipeline.
If you are not familiar with scikit-learn's pipeline we recommend you take a look at the official documentation [1]_.

The purpose of such a pipeline is to assemble several preprocessing steps that can be cross-validated together while
setting different parameters.
Our tsfresh transformer allows you to extract and filter the time series features during such a preprocessing sequence.

The first two estimators contained in tsfresh are the :class:`~tsfresh.transformers.feature_augmenter.FeatureAugmenter`,
which extracts the features, and the :class:`~tsfresh.transformers.feature_selector.FeatureSelector`, which only
performs the feature selection algorithm.
It is preferable to combine extracting and filtering of the features in a single step to avoid unnecessary feature
calculations.
Hence, we have the :class:`~tsfresh.transformers.feature_augmenter.RelevantFeatureAugmenter`, which combines both the
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

    pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                ('classifier', RandomForestClassifier())])

    df_ts, y = load_robot_execution_failures()
    X = pd.DataFrame(index=y.index)

    pipeline.set_params(augmenter__timeseries_container=df_ts)
    pipeline.fit(X, y)

The parameters of the augment transformer correspond to the parameters of the top-level convenience function
:func:`~tsfresh.convenience.relevant_extraction.extract_relevant_features`.
In the example, we only set the names of two columns ``column_id='id'``, ``column_sort='time'``
(see :ref:`data-formats-label` for an explanation of those parameters).

Because we cannot pass the time series container directly as a parameter to the augmenter step when calling fit or
transform on a :class:`sklearn.pipeline.Pipeline` we have to set it manually by calling
``pipeline.set_params(augmenter__timeseries_container=df_ts)``.
In general, you can change the time series container from which the features are extracted by calling either the
pipeline's :func:`~sklearn.pipeline.Pipeline.set_params` method or the transformers
:func:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter.set_timeseries_container` method.

For further examples, see the Jupyter Notebook pipeline_example.ipynb in the notebooks folder of the tsfresh package.


References
----------

    .. [1] http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
