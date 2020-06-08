.. _run:

===============================
Run a Surrogate Benchmarks
===============================

The surrogate benchmark runs as an independent daemon process and is connected to the optimizer via a local socket.
For debugging reasons it is also possible to start this process without creating a daemon. To interact with the daemon we have two scripts:

``daemonize_benchmark.py``
    as the name says starting a daemon, that listens to a local socket. A daemon runs till it reaches it timeout limit, which is set to 1200 sec.
    This means that the daemon terminates after waiting for more than 1200secs without receiving a single request.
    ``daemonize_benchmark.py`` can also be used to stop a running daemon before the timelimit is reached.

``daemon_whisperer.py``
    can talk to the daemon and is used to request performance predictions from the surrogate benchmark. This script implements the same interface as
    all the HPOlib benchmarks. It also implements a **fallback functionality*, which tries to resurrect the daemon process if it is not running.

Example 1 and 2 will show how to use these scripts


Example 1 - Starting a Surrogate Benchmark
===============================


#. Download a trained regression models, e.g. a KNN model, which uses a one-hot
   encoding and is trained on all data. You can get it from automl.org

   .. code:: bash

        wget www.automl.org/downloads/surrogate/onlineLDA_surrogate.tar.gz
        tar -xf onlineLDA_surrogate.tar.gz
        file `pwd`/onlineLDA/models/ENCODED_onlineLDA_all_KNN
        MODELNAME=`pwd`/onlineLDA/models/ENCODED_onlineLDA_all_KNN


#. Get the corresponding file describing the searchspace: `params.pcs`

    .. code:: bash

        file `pwd`/onlineLDA/smac_2_06_01-dev/params.pcs
        PCSFILE=`pwd`/onlineLDA/smac_2_06_01-dev/params.pcs

#. Start a daemon benchmark:

    .. code:: bash

        mkdir ~/socketdir
        SOCKETDIR=~/socketdir
        daemonize_benchmark.py --surrogateData ${MODELNAME} --pcs ${PCSFILE} --socket ${SOCKETDIR}/something --dry

    By adding `--dry` we do not start an actual daemon, but just print
    a python command, which we can run to see the surrogate benchmark work

        `<output of previous command>`

    Now the surrogate benchmark is running and listens for requests on the local socket in /socketdir/something.
    When this works, the benchmark outputs its regression model (which is wrong for this only one model,
    saying it is an SVM, but it is indeed a KNN :-) ) and the time left till it terminates.

#. In a different terminal, you can send a request:

    .. code:: bash

        cd ..
        <new terminal>
        source virtualSurrogate/bin/activate
        daemon_whisperer.py --socket ~/socketdir/something --fold 0 --folds 1 --params -Kappa 0.75 -Tau 512 -S 8192

    Which should give you the following output:
    ::

        Found a socket on /home/eggenspk/socketdir/something
        Requesting: --fold 0 --folds 1 --params -Kappa 0.75 -Tau 512 -S 8192..........
        Answer: 1416.1125104
        Result for ParamILS: SAT, 0.002427, 1, 1416.112510, -1, I'm not a daemon

    Whereas in the benchmark's terminal you will find:

    ::

        Got a connection: <socket._socketobject object at 0x7f0c0ea8ff30> on
        Received data: --fold 0 --folds 1 --params -Kappa 0.75 -Tau 512 -S 8192
        CLEAN {'Tau': 512, 'S': 8192, 'Kappa': 0.75}
        AFTER Unlogging:
        {'Tau': 512, 'S': 8192, 'Kappa': 0.75}
        {}
        Requesting performance for: [0, 512, 8192, 0.75]
        Encoding categorical features using a one hot encoding scheme
        My answer: 1416.1125104

    If you are missing a parameter or otherwise kill the script you might kill the surrogate process.
    In such a case, you need to manually delete the socket in :literal:`socketdir/something` and start over at step 3.

#.  You can now play around with the surrogate and send different requests. You can follow the requests in both terminal windows.
    When you are finished you can either manually kill the benchmark process with :literal:`ctr+C` or send the request to stop the process:

    .. code:: bash

        daemonize_benchmark.py --socket ~/socketdir/something --stop --pcs `pwd`/onlineLDA/smac_2_06_01-dev/params.pcs

Next you can run your surrogate benchmark as a daemon process.

Example 2 - Starting a daemon
===============================

#. Again run the command from above, but without :bash:`--dry`. You won't see any output,
   but you can verify with :bash:`ps -ef | grep daemon_benchmark` that your daemon is running.
   If not you can check :literal:`${SOCKETDIR}/somethingdaemon_log.txt` for errors.

    .. code:: bash

        daemonize_benchmark.py --surrogateData ${MODELNAME} --pcs ${PCSFILE} --socket ${SOCKETDIR}/something


#. Now you can send the same request as before (in the same terminal):

    .. code:: bash

        daemon_whisperer.py --socket ${SOCKETDIR}/something --fold 0 --folds 1 --params -Kappa 0.75 -Tau 512 -S 8192

    Which should give you the same output as before.

#. The benchmark output can be found in :literal:`${SOCKETDIR}/somethingdaemon_log.txt`

#. To stop the daemon run

    .. code:: bash
    
        daemonize_benchmark.py --socket ${SOCKETDIR} --pcs space.pcs --stop
        
        
