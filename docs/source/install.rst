.. _install:

====================================
Install
====================================

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

Preparing a virtualenv
===============================

We recommend using a virtualenv, because (A) you can control the version of each python package,
(B) installing and removing packages becomes easy as (C) you don't need sudo rights and
(D) it becomes harder to mess up up the python installation on your system.

#. Get `virtualenv <http://www.virtualenv.org/en/latest/virtualenv.html#installation>`_,
   then load a freshly created virtualenv. (If you are not familiar with virtualenv,
   you might want to read `more <http://www.virtualenv.org/en/latest/virtualenv.html)>`_ about it)

    .. code:: bash

        pip install virtualenv
        virtualenv virtualSurrogates
        source virtualSurrogates/bin/activate

#. Install :bash:`numpy`, :bash:`scipy`, :bash:`matplotlib`, as this doesn't
   work through setup.py.

    .. code:: bash

        easy_install -U distribute
        pip install numpy==1.8.1
        pip install scipy==0.14.0
        pip install matplotlib
        pip install scikit-learn==0.15.1
        pip install matplotlib

   This may take some time. Afterwards you can verify having those libs installed with:

    .. code:: bash

        pip freeze

            argparse==1.2.1
            mock==1.0.1
            nose==1.3.4
            numpy==1.8.1
            pyparsing==2.0.3
            python-dateutil==2.3
            pytz==2014.10
            scipy==0.14.0
            six==1.8.0
            wsgiref==0.1.2
            scikit-learn==0.15.1
            matplotlib==1.4.2

Install the Surrogate Benchmark Library
===============================

#. Clone the repository:
     .. code:: bash

        git clone https://github.com/KEggensperger/SurrogateBenchmarks.git
        cd SurrogateBenchmarks

#. Run setup.py

    .. code:: python

        python setup.py install

   This will install tools, scripts  and some requirements (:bash:`networkx`, :bash:`pyparsing`, and :bash:`python-daemon`).
   This might take a while. When your environment is ready it could/should look like this:

    .. code:: bash

        pip freeze
            Surrogates==Nan
            argparse==1.2.1
            decorator==3.4.0
            lockfile==0.10.2
            mock==1.0.1
            networkx==1.9.1
            numpy==1.8.1
            pyparsing==2.0.3
            python-daemon==1.6.1
            python-dateutil==2.3
            pytz==2014.10
            scikit-learn==0.15.1
            scipy==0.14.0
            six==1.8.0
            wsgiref==0.1.2

#. If the installation was successful you can run some tests.
   **NOTE**: Some tests will fail, if you are using different versions of `numpy`, `scipy`, and/or `scikit-learn`. This is not problematic
   as some of the tests only assert that you retrieve exactly the same results as me and as the numeric results only slightly differ.

    .. code:: python

        python setup.py test
