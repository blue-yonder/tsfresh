How to contribute
=================

We want tsfresh to become the biggest archive of feature extraction methods in python. To achieve this goal, we need
your help!

All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome. If you
want to add one or two interesting feature calculators, implement a new feature selection process or just fix 1-2 typos,
your help is appreciated.

If you want to help, just create a pull request on our github page. To the new user, working with Git can sometimes be
confusing and frustrating. If you are not familiar with Git you can also contact us by :ref:`email <authors>`.


Guidelines
''''''''''

There are three general coding paradigms that we believe in:

    1. **Keep it simple**. We believe that *"Programs should be written for people to read, and only incidentally for
       machines to execute."*.

    2. **Keep it documented** by at least including a docstring for each method and class. Do not describe what you are
       doing but why you are doing it.

    3. **Keep it tested**. We aim for a high test coverage.


There are two important copyright guidelines:

    4. Please do not include any data sets for which a licence is not available or commercial use is even prohibited.
       Those can undermine the licence of the whole projects.

    5. Do not use code snippets for which a licence is not available (e.g. from stackoverflow) or commercial use is
       even prohibited. Those can undermine the licence of the whole projects.

Further, there are some technical decisions we made:

    6. Clear the Output of iPython notebooks. This improves the readability of related Git diffs.


Installation
''''''''''''

Install all the relevant python packages with

.. code::

    cd /path/to/tsfresh
    pip install -r requirements.txt
    pip install -r rdocs-requirements.txt
    pip install -r test-requirements.txt
    pre-commit install
    pip install -e .


The last command will dynamically link the tsfresh package which means that changes to the code will directly show up
for example in your test run.


Test framework
''''''''''''''

After making your changes, you probably want to test your changes locally. To run our comprehensive suite of unit tests
you have to:


.. code::

    pytest tests/


Documentation
'''''''''''''

Build the documentation after installing as described above with


.. code::

    python setup.py docs


The finished documentation can be found in the docs/_build/html folder.


Styling
'''''''

We use black and isort for styling. They are automatically triggered on every commit after having installed pre-commit
(as described above).


We are looking forward to hear from you! =)
