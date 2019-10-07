#!/usr/bin/python3
# -*- coding: utf-8 -*-
##===-----------------------------------------------------------------------------*- Python -*-===##
##                         _       _                   
##                        | |     | |                  
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/ 
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##
##
## Several system related utility functions.
##
## Source: https://github.com/llvm-mirror/llvm/blob/master/utils/lit/lit/util.py with modification
## by Fabian Thuering
##
##===------------------------------------------------------------------------------------------===##

import os
import platform
import signal
import subprocess
import threading
import time


def to_bytes(str):
    # Encode to UTF-8 to get binary data.
    return str.encode('utf-8')


def to_string(bytes):
    if isinstance(bytes, str):
        return bytes
    return to_bytes(bytes)


def convert_string(bytes):
    try:
        return to_string(bytes.decode('utf-8'))
    except AttributeError:  # 'str' object has no attribute 'decode'.
        return str(bytes)
    except UnicodeError:
        return str(bytes)


def levenshtein(source, target):
    """ From Wikipedia article; Iterative with two matrix rows. """
    if source == target:
        return 0
    elif len(source) == 0:
        return len(target)
    elif len(target) == 0:
        return len(source)
    v0 = [None] * (len(target) + 1)
    v1 = [None] * (len(target) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(source)):
        v1[0] = i + 1
        for j in range(len(target)):
            cost = 0 if source[i] == target[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(target)]


def detectCPUs():
    """
    Detects the number of CPUs on a system. Cribbed from pp.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(capture(['sysctl', '-n', 'hw.ncpu']))
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            # With more than 32 processes, process creation often fails with
            # "Too many open files".  FIXME: Check if there's a better fix.
            return min(ncpus, 32)
    return 1  # Default


def capture(args, env=None):
    """capture(command) - Run the given command (or argv list) in a shell and
    return the standard output. Raises a CalledProcessError if the command
    exits with a non-zero status."""
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env)
    out, err = p.communicate()
    out = convert_string(out)
    err = convert_string(err)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(cmd=args,
                                            returncode=p.returncode,
                                            output="{}\n{}".format(out, err))
    return out


class ExecuteCommandTimeoutException(Exception):
    def __init__(self, msg, out, err, exitCode):
        assert isinstance(msg, str)
        assert isinstance(out, str)
        assert isinstance(err, str)
        assert isinstance(exitCode, int)
        self.msg = msg
        self.out = out
        self.err = err
        self.exitCode = exitCode


# Close extra file handles on UNIX (on Windows this cannot be done while
# also redirecting input).
kUseCloseFDs = not (platform.system() == 'Windows')


def asyncExecuteCommand(commands, cwds, env=None):
    running_procs = [(subprocess.Popen(command, cwd=cwd,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env=env, close_fds=kUseCloseFDs), idx) for command, cwd, idx
                     in zip(commands, cwds, range(0, len(commands)))]

    print(running_procs)
    results = len(commands) * [None]

    while running_procs:
        for proc, idx in running_procs:
            print(proc, idx)
            retcode = proc.poll()
            if retcode is not None:  # Process finished.
                out, err = proc.communicate()
                results[idx] = (out, err, retcode)
                running_procs.remove((proc, idx))
                break
            else:  # No process is done, wait a bit and check again.
                time.sleep(.1)
                continue

    return results


def executeCommand(command, cwd=None, env=None, input=None, timeout=0):
    """
        Execute command ``command`` (list of arguments or string)
        with
        * working directory ``cwd`` (str), use None to use the current
          working directory
        * environment ``env`` (dict), use None for none
        * Input to the command ``input`` (str), use string to pass
          no input.
        * Max execution time ``timeout`` (int) seconds. Use 0 for no timeout.

        Returns a tuple (out, err, exitCode) where
        * ``out`` (str) is the standard output of running the command
        * ``err`` (str) is the standard error of running the command
        * ``exitCode`` (int) is the exitCode of running the command

        If the timeout is hit an ``ExecuteCommandTimeoutException``
        is raised.
    """
    p = subprocess.Popen(command, cwd=cwd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=env, close_fds=kUseCloseFDs)
    timerObject = None
    hitTimeOut = [False]
    try:
        if timeout > 0:
            def killProcess():
                # We may be invoking a shell so we need to kill the
                # process and all its children.
                hitTimeOut[0] = True
                killProcessAndChildren(p.pid)

            timerObject = threading.Timer(timeout, killProcess)
            timerObject.start()

        out, err = p.communicate(input=input)
        exitCode = p.wait()
    finally:
        if timerObject != None:
            timerObject.cancel()

    # Ensure the resulting output is always of string type.
    out = convert_string(out)
    err = convert_string(err)

    if hitTimeOut[0]:
        raise ExecuteCommandTimeoutException(
            msg='Reached timeout of {} seconds'.format(timeout),
            out=out,
            err=err,
            exitCode=exitCode
        )

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    return out, err, exitCode


def killProcessAndChildren(pid):
    """
    This function kills a process with ``pid`` and all its
    running children (recursively). It is currently implemented
    using the psutil module which provides a simple platform
    neutral implementation.
    """
    import psutil
    try:
        psutilProc = psutil.Process(pid)
        # Handle the different psutil API versions
        try:
            # psutil >= 2.x
            children_iterator = psutilProc.children(recursive=True)
        except AttributeError:
            # psutil 1.x
            children_iterator = psutilProc.get_children(recursive=True)
        for child in children_iterator:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutilProc.kill()
    except psutil.NoSuchProcess:
        pass
