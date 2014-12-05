#! /usr/bin/env python
from argparse import ArgumentParser
import cPickle

import os
import socket
import sys
import time
import traceback

import numpy

import daemon

RNG = 1
numpy.random.seed(RNG)

from Surrogates.RegressionModels import GradientBoosting, KNN, LassoRegression, \
     LinearRegression, NuSupportVectorRegression, RidgeRegression, SupportVectorRegression, RandomForest

# ArcGP, Fastrf, GaussianProcess

import Surrogates.DataExtraction.handle_configurations


def parse_cli(misc_args):
    """
    Provide a generic command line interface for benchmarks. It will just parse
    the command line according to simple rules and return two dictionaries, one
    containing all arguments for the benchmark algorithm like dataset,
    crossvalidation metadata etc. and the containing all learning algorithm
    hyperparameters.

    Parsing rules:
    - Arguments with two minus signs are treated as benchmark arguments, Xalues
     are not allowed to start with a minus. The last argument must --params,
     starting the hyperparameter arguments.
    - All arguments after --params are treated as hyperparameters to the
     learning algorithm. Every parameter name must start with one minus and must
     have exactly one value which has to be given in single quotes.

    Example:
    python neural_network.py --folds 10 --fold 1 --dataset convex  --params
        -depth '3' -n_hid_0 '1024' -n_hid_1 '1024' -n_hid_2 '1024' -lr '0.01'
    """
    args = {}
    parameters = {}

    cli_args = misc_args
    found_params = False
    skip = False
    iterator = enumerate(cli_args)
    for idx, arg in iterator:
        if skip:
            skip = False
            continue
        else:
            skip = True

        if arg == "--params":
            found_params = True
            skip = False

        elif arg[0:2] == "--" and not found_params:
            if cli_args[idx+1][0] == "-":
                raise ValueError("Argument name is not allowed to have a "
                                 "leading minus %s" % cli_args[idx + 1])
            args[cli_args[idx][2:]] = cli_args[idx+1]

        elif arg[0:2] == "--" and found_params:
            raise ValueError("You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        elif arg[0] == "-" and arg[0:2] != "--" and found_params:
            parameters[cli_args[idx][1:]] = cli_args[idx+1]

        elif arg[0] == "-" and arg[0:2] != "--" and not found_params:
            raise ValueError("You either try to use arguments with only one lea"
                             "ding minus or try to specify a hyperparameter bef"
                             "ore the --params argument. %s" %
                             " ".join(cli_args))

        elif not found_params:
            raise ValueError("Illegal command line string, expected an argument"
                             " starting with -- but found %s" % (arg,))

        else:
            raise ValueError("Illegal command line string, expected a hyperpara"
                             "meter starting with - but found %s" % (arg,))

    return args, parameters


def build_input_array(params, other, sp, unimap, logdict, cond_dict, param_names, dflt):
    clean_dict = dict()
    for i in params.keys():
        tmp_i = i
        if i[0] == "-":
            tmp_i = i[1:]
        if isinstance(sp[unimap[tmp_i]], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            clean_dict[tmp_i] = params[i].strip(" ").strip("'").strip('"')
        else:
            clean_dict[tmp_i] = Surrogates.DataExtraction.handle_configurations.convert_to_number(params[i])
    print "CLEAN", clean_dict
    
    # Unlog parameter
    clean_dict = Surrogates.DataExtraction.handle_configurations.put_on_uniform_scale(clean_dict, sp, unimap, logdict)

    print "AFTER Unlogging:"
    print clean_dict
    print
    print cond_dict
    clean_dict = Surrogates.DataExtraction.handle_configurations.remove_inactive(clean_dict, cond_dict)

    # Fold is always the first entry
    if "fold" in other:
        fold = int(other["fold"])
    else:
        err_str = "Don't know where to find fold: %s" % str(other)
        sys.stderr.write(err_str + "\n")
        return err_str
    row = list()
    row.append(fold)
    # Now fill the param row
    for p_idx, p in enumerate(param_names[1:]):
        if p == "duration" or p == "result":
            # we're finished
            break
        if type(clean_dict[p]) != str and numpy.isnan(clean_dict[p]):
            # Replace with default
            if p in dflt:
                row.append(dflt[p])
            else:
                raise ValueError("Don't know that param")
        else:
            # be sure this is a real para
            assert type(clean_dict[p]) == str or numpy.isfinite(clean_dict[p])
            row.append(clean_dict[p])
    return row


def format_return_string(res):
    ret_str = "Result for ParamILS: %s, %f, 1, %f, %d, %s"
    sat = "SAT"
    duration = 10
    result = res
    additional_info = "I'm not a daemon"

    if type(res) == str:
        # Something happened
        result = 100
        additional_info = res
    return ret_str % (sat, duration, result, -1, additional_info)


def shutdown_server(s, socket_name):
    # Shutdown server and remove socket file
    try:
        s.shutdown(socket.SHUT_RDWR)
    except Exception, e:
        import traceback
        print "Something went wrong when shutting down the server"
        print traceback.format_exc()
    finally:
        os.remove(socket_name)
        # os.remove(socket_name + "daemon_log.txt")
    return True


def run_loop(socket_name, data_name, pcs, timeout_time=20*60, buffer_size=50, end_str="."*10):
    # Read stuff we need to build input arrays
    fh = open(data_name, 'r')
    surrogate = cPickle.load(fh)
    fh.close()

    sp = surrogate._sp
    logmap = Surrogates.DataExtraction.handle_configurations.get_log_to_uniform_map(sp)
    unimap = Surrogates.DataExtraction.handle_configurations.get_uniform_to_log_map(sp)
    dflt = Surrogates.DataExtraction.handle_configurations.get_default_values(sp)
    cond_dict = Surrogates.DataExtraction.handle_configurations.get_cond_dict(sp)
    logdict = Surrogates.DataExtraction.handle_configurations.get_logparams(pcs)
    catdict = Surrogates.DataExtraction.handle_configurations.get_cat_val_map(sp)
    para_header = surrogate._param_names

    if os.path.exists(socket_name):
        sys.stderr.write("Could not build socket in %s, file already exists\n" % socket_name)
        return

    sys.stderr.write("Arrived in function, building socket on %s" % socket_name)
    timeout_blocking = 10
    timeout_ct_limit = int(timeout_time / timeout_blocking)

    # Create socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(socket_name)
    s.listen(1)
    s.settimeout(timeout_blocking)

    # Now listen
    timeout = False
    timeout_ct = timeout_ct_limit
    evaluate_on_last_request = False
    try:
        sys.stdout.write("I'm ready for some requests\n")
        sys.stdout.write("I'm holding a %s, %s on socket %s\n" % (surrogate, surrogate._name, socket_name))
        while not timeout:
            if timeout_ct == 0:
                timeout = True
                sys.stdout.write("I feel neglected..shutting down\n")
                continue

            try:
                conn, addr = s.accept()
            except socket.timeout:
                # No one wants to talk to me
                print "I'm so lonesome, I could cry ... or die in", timeout_blocking*timeout_ct, "sec"
                timeout_ct -= 1
                continue

            # Now we went somehow further and can reset the counter
            timeout_ct = timeout_ct_limit

            sys.stdout.write("Got a connection: " + str(conn) + " on " + str(addr) + "\n")
            data = ""
            while end_str not in data:
                data += conn.recv(buffer_size)
            data = data[:-len(end_str)]

            sys.stdout.write("Received data: %s\n" % data)

            ans = False
            if not data:
                evaluate_on_last_request = False
                continue

            if data == "STOP":
                # We received a stop signal
                timeout = True
                ans = "Closing"

            # Just greet
            if data == "SayHello":
                ans = "Hello =)"

            # Return surrogate type
            if data == "type":
                ans = surrogate._name

            if not ans:
                try:
                    other_args, params = parse_cli(data.split(" "))
                    input_array = build_input_array(params=params, other=other_args, sp=sp,
                                                    unimap=unimap, logdict=logdict,
                                                    cond_dict=cond_dict, param_names=para_header,
                                                    dflt=dflt)
                except Exception as e:
                    # Whatever happened we give it back
                    input_array = traceback.format_exc()
                if type(input_array) == list:
                    # Dirty hack to make stuff work
                    #if surrogate_type == "spear_gp":
                    #    param_arr = numpy.array(param_arr).reshape([1, len(param_arr)])
                    sys.stdout.write("Requesting performance for: %s\n" % str(input_array))
                    ans = surrogate.predict(input_array)
                    if isinstance(ans, list) or isinstance(ans, numpy.ndarray):
                        ans = ans[0]
                else:
                    ans = input_array
                evaluate_on_last_request = True

            # Whatever ans is, it needs to be a str
            ans = str(ans)
            sys.stdout.write("My answer: " + ans + "\n")
            sys.stdout.flush()
            conn.send(ans)

    except KeyboardInterrupt:
        sys.stderr.write("You tried to kill me..excuse me I need to shutdown\n")
        shutdown_server(s, socket_name)
        return
    shutdown_server(s, socket_name)
    return


def main(args, unknown):
    # Unpickle data
    data_name = args.data

    if not os.path.exists(data_name):
        sys.stderr.write("%s does not exist\n" % data_name)
        sys.exit(1)

    if args.daemon:
        fh = open(args.socket + "daemon_log.txt", 'w')
        with daemon.DaemonContext(stdout=fh, stderr=fh):
            run_loop(socket_name=args.socket, data_name=data_name, timeout_time=20*60,
                     pcs=args.pcs, buffer_size=50, end_str="."*10)
    else:
        run_loop(socket_name=args.socket, data_name=data_name, pcs=args.pcs,
                 timeout_time=20*60, buffer_size=50, end_str="."*10)

if __name__ == "__main__":
    prog = "python daemon_benchmark.py"
    parser = ArgumentParser(description="Only for internal use. Do not call explicitly.", prog=prog)

    # IPC infos
    parser.add_argument("--socket", dest="socket", default=None, required=True,
                        help="Where to create socket?")
    parser.add_argument("--data", dest="data", default=None, required=True,
                        help="Where is the pickled data for this surrogate?")
    parser.add_argument("--pcs", dest="pcs", required=True)
    parser.add_argument("--daemon", dest="daemon", default=False, action="store_true",
                        help="Run as daemon")

    outer_args, outer_unknown = parser.parse_known_args()
    main(outer_args, outer_unknown)

