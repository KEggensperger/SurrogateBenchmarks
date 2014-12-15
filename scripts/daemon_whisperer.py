#! /usr/bin/env python
from argparse import ArgumentParser

import os
import socket
import sys
import subprocess
import numpy
import time

RNG = 1
numpy.random.seed(RNG)


def format_return_string(res, duration):
    ret_str = "Result for ParamILS: %s, %f, 1, %f, %d, %s"
    sat = "SAT"
    result = res
    additional_info = "I'm not a daemon"

    if type(res) == str or not numpy.isfinite(res):
        # Something happened
        raise ValueError("Something happened: %s" % res)        
        #result = 100
        #additional_info = res
    return ret_str % (sat, duration, result, -1, additional_info)


def evaluate_config(socket_name, params, buffer_size=1024, end_str="."*10):
    if not os.path.exists(socket_name):
        return "Socket %s does not exist" % socket_name

    print params

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(socket_name)
        print "Found a socket on %s" % socket_name
        request = params + end_str
        print "Requesting: %s" % request
        s.send(request)
        data = s.recv(buffer_size)
        print "Answer: %s" % data
        s.close()
        try:
            res = float(data)
        except:
            raise Exception("Somethings wrong with that: >%s<" % str(data))
        return res
    except socket.error:
        return "Could not connect to socket"


def whisper(socket_name, message, end_str="."*10, buffer_size=1024):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(socket_name)
        request = message + end_str
        s.send(request)
        data = s.recv(buffer_size)
        s.close()
        return data
    except socket.error:
        return False


def main():
    prog = "python daemon_whisperer.py"
    parser = ArgumentParser(description="", prog=prog)

    # IPC infos
    parser.add_argument("--socket", dest="socket", default=None, required=True,
                        help="To which socket listens the daemon?")
    parser.add_argument("--data", dest="data", default=None,
                        help="In case of trouble, shall I "
                             "try to resurrect daemon?")
    parser.add_argument("--pcs", dest="pcs", default=None,
                        help="In case of trouble, shall I "
                             "try to resurrect daemon?")
    args, unknown = parser.parse_known_args()

    socket_dir = args.socket
    if not os.path.isabs(socket_dir):
        socket_dir = os.path.normpath(os.path.abspath(socket_dir))

    surrogate_data = args.data
    if surrogate_data and not os.path.isabs(surrogate_data):
            surrogate_data = os.path.normpath(os.path.abspath(surrogate_data))

    pcs_file = args.pcs
    if pcs_file and not os.path.isabs(pcs_file):
        pcs_file = os.path.normpath(os.path.abspath(pcs_file))

    # Talk to daemon
    print unknown

    start = time.time()
    res = evaluate_config(socket_name=socket_dir, params=" ".join(unknown))
    
    if surrogate_data is not None and pcs_file is not None and \
            type(res) == str and \
            (res == "Could not connect to socket" or "does not exist" in res):
        print "Daemon seems to be dead"
        if os.path.exists(socket_dir):
            os.remove(socket_dir)

        cmd = ["daemon_benchmark.py",
               "--socket", socket_dir, "--data", surrogate_data,
               "--pcs", pcs_file, "--daemon"]
        p = subprocess.Popen(cmd)
        sys.stdout.write(" ".join(cmd) + "\n")

        try_ct = 0
        works = False
        while not works and try_ct < 20:
            answer = whisper(socket_name=socket_dir, message="SayHello")
            print "Daemon answers: ", answer
            if answer == "Hello =)":
                works = True
                print "Deamon is resurrected!"
            time.sleep(5)
            try_ct += 1
        
        if not works:
            print "Could not bring daemon to life..."
            raise ValueError("Socket is broken and daemon not repairable: %s" % str(args))
        
        # Now try one more time
        res = evaluate_config(socket_name=socket_dir, params=" ".join(unknown))
        
    duration = time.time() - start
    res_str = format_return_string(res=res, duration=duration)
    sys.stdout.write(res_str + "\n")

if __name__ == "__main__":
    main()

