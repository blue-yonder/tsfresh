How to contribute
=================

We want tsfresh to become the biggest archive of feature extraction methods in python. To achieve this goal, we need
your help!

So if you want to add one or two interesting feature calculators, implement a new feature selection process
or just fix 1-2 typos, your help is appreciated.
Testing setup
'''''''''''''

After making your changes, you probably want to test your changes locally. To run our comprehensive suit of unit tests
you have to install all the relevant python packages with


.. code::

    cd /path/to/tsfresh
    pip install -r requirements.txt
    pip install -r docs-requirements.txt
    pip install -r test-requirements.txt
    pip install -e .


The last command will dynamically link the tsfresh package which means that changes to the code will directly show up
for example in your test run.

Then, if you have everything installed, you can run the tests with


.. code::

    python setupy.py test


or build the documentation with


.. code::

    python setupy.py docs



The finished documentation can be found in the docs/_build/html folder.

On Github we use a Travis CI Folder that runs our test suite every time a commit or pull request is sent. The
configuration of Travi is controlled by the `.travis.yml <https://github.com/blue-yonder/tsfresh/blob/master/.travis.yml>`_
file.

If you want to help, just create a pull request on our github page. If you are not familiar with the versioning tool git
you can contact us by email.

We are looking forward to hear from you! =)