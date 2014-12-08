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

We recommend using a virtualenv, because (A) you can control the version of
each python package, (B) installing and removing packages becomes easy as (C)
you don't need sudo rights and (D) if you observe the rules, you cannot mess up
the python installation on your system.

1.  Get `virtualenv <http://www.virtualenv.org/en/latest/virtualenv.html#installation>`_,
    then load a freshly created virtualenv. (If you are not familiar with virtualenv,
    you might want to read `more <http://www.virtualenv.org/en/latest/virtualenv.html)>`_ about it)

    .. code:: bash

        pip install virtualenv
        virtualenv virtualSurrogates
        source virtualSurrogates/bin/activate

3.  Install :bash:`numpy`, :bash:`scipy`, :bash:`matplotlib`, as this doesn't
    work through setup.py.

    .. code:: bash

        easy_install -U distribute
        pip install numpy==1.8.1
        pip install scipy==0.14.0
        pip install matplotlib

    This may take some time. Afterwards you can verify having those libs installed with:

    .. code:: bash

        pip freeze

            argparse==1.2.1
            matplotlib==1.4.2
            mock==1.0.1
            nose==1.3.4
            numpy==1.8.1
            pyparsing==2.0.3
            python-dateutil==2.3
            pytz==2014.10
            scipy==0.14.0
            six==1.8.0
            wsgiref==0.1.2

Install the Surrogate Benchmark Library
===============================

1. Clone the repository:
     .. code:: bash

        git clone https://keggensperger@bitbucket.org/keggensperger/surrogatebenchmarks.git
        cd surrogatebenchmarks

2. Run setup.py

    .. code:: python

        python setup.py install

    This will install tools, scripts  and some requirements (:bash:`networkx`,
    :bash:`pyparsing`, :bash:`scikit-learn`, and :bash:`python-daemon`).
    This might take a while. When finished your environment now could/should look like this:

    .. code:: bash

        pip freeze
            Surrogates==Nan
            argparse==1.2.1
            decorator==3.4.0
            lockfile==0.10.2
            matplotlib==1.4.2
            mock==1.0.1
            networkx==1.9.1
            numpy==1.8.1
            pyparsing==2.0.3
            python-daemon==1.6.1
            python-dateutil==2.3
            pytz==2014.10
            scikit-learn==0.15.2
            scipy==0.14.0
            six==1.8.0
            wsgiref==0.1.2

3. If the installation was successful you can run test.
   **NOTE**: Some tests will fail, if you are using different versions of `numpy`, `scipy`, and/or `scikit-learn`. This is not problematic
   as some of the tests only assert that you retrieve exactly the same results as , as the numeric results only slightly differs.

    .. code:: python

        python setup.py test

Run a Surrogate Benchmarks
===============================

The surrogate benchmarks run as independent daemon processes and are connected to the optimization run via a local socket.
For debugging reasons it is also possible to start such a process without creating a daemon. To interact with the daemon we have two scripts:

``daemonize_benchmark.py``
    as the name says daemonizes the surrogate benchmark by starting a daemon, that listens to a local socket. A daemon runs till its timeout limit is
    reached, which is set to 1200 sec. This means if that the daemon shuts down after waiting for more than 1200secs without receiving a single request.
    ``daemonize_benchmark.py`` can also be used to stop running daemons before that timelimit is reached.

``daemon_whisperer.py``
    can talk to the daemon and is used to request performance predictions from the surrogate benchmark. This script implements the same interface as
    all the HPOlib benchmarks. It also implements a **fallback functionality*, which tries to resurrect the daemon process if it is not running.

Example 1 and 2 will show how to use these scripts


Example 1 - Starting a Surrogate Benchmark
----------

1. Download a trained regression models, e.g. a KNN model, which uses a one-hot
encoding and is trained on all data: `ENCODED_logreg_nocv_all_KNN`

2. Get the corresponding file describing the searchspace: `space.pcs`

3. Start a daemon benchmark:

    .. code:: bash

        wget FOLDERLOGREG
        cd LogReg
        mkdir socketdir
        daemonize_benchmark.py --surrogateData ENCODED_logreg_nocv_all_KNN --pcs space.pcs --socket ./socketdir/something --dry

    By adding :bash:`--dry` we do not start an actual daemon, but just print
    a python command, which we can run to see the surrogate benchmark work

    .. code:: bash

        <output of previous command, e.g. daemon_benchmark.py --socket /home/eggenspk/TEST/LogReg/socketdir/something --data ENCODED_logreg_nocv_all_KNN --pcs space.pcs>

    Now the surrogate benchmark is running and listens for requests on the local socket in /socketdir/something.
    If this works, the benchmark outputs its regression model and :bash:`I'm so lonesome, I could cry ... or die in X sec`.

4. In a different terminal, you can send a request:
    .. code:: bash

        cd ..
        <new terminal>
        source virtualSurrogate/bin/activate
        daemon_whisperer.py --socket LogReg/socketdir/something --fold 0 --folds 1 --params -lrate 5 -l2_reg 0.5 -batchsize 1010 -n_epochs 1003

    Which should give you the following output:
    ::

        Found a socket on LogReg/socketdir/something
        Requesting: --fold 0 --folds 1 --params -lrate 5 -l2_reg 0.5 -batchsize 1010 -n_epochs 1003..........
        Answer: 0.725523
        Result for ParamILS: SAT, 0.002759, 1, 0.725523, -1, I'm not a daemon

    If you are missing a parameter or confuse the script you might kill the surrogate process.
    In such a case, you need to manually delete the socket in :literal:`socketdir/something` and start over at step 3.

5. You can now play around with the surrogate and send different requests. You can follow the requests in both terminal windows.
   When you are finished you can either manually kill the benchmark process with :literal:`ctr+C` or send the request to stop the process:

    ..code:: bash

    daemonize_benchmark.py --socket LogReg/socketdir/bla --stop --pcs LogReg/space.pcs

Next you can run your surrogate benchmark as a daemon process.

Example 2 - Starting a daemon
---------
1. Again run the command from above, but without :bash:`--dry`

    .. code:: bash
        daemonize_benchmark.py --surrogateData ENCODED_logreg_nocv_all_KNN --pcs space.pcs --socket ./socketdir/something

    You won't see any output, but you can verify with :bash:`ps -ef | grep daemon_benchmark` that your daemon is running.
    If not you can check `socketdir/somethingdaemon_log.txt` for errors.

2. Now you can send the same request as before:

    .. code:: bash

        cd ..
        source virtualSurrogate/bin/activate
        daemon_whisperer.py --socket LogReg/socketdir/something --fold 0 --folds 1 --params -lrate 5 -l2_reg 0.5 -batchsize 1010 -n_epochs 1003

    Which should give you the very same output (except the runtime) as before.

3. The benchmark output can be found in `socketdir/somethingdaemon_log.txt`

4. To stop the daemon run :bash:`daemonize_benchmark.py --socket socketdir/something --pcs space.pcs --stop`