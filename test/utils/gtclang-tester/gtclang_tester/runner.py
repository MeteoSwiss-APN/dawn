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

from collections import defaultdict
from difflib import unified_diff
from json import load, dumps, loads
from os import rename, remove
from signal import SIGSEGV
from time import time

from .config import Config
from .error import report_info
from .progressbar import TerminalController, ProgressBar, SimpleProgressBar, EmptyProgressbar
from .utility import executeCommand, levenshtein


def run(tests):
    """ Run the tests """
    tester = TestRunner(tests)
    return tester.run()


class TestRunner(object):
    """ Run the tests and report errors """

    def __init__(self, tests):
        self.__tests = tests
        self.__term = TerminalController()
        self.__failure_map = defaultdict(dict)

        self.__test_count = 0
        self.__test_count_passed = 0
        self.__test_time = 0

        try:
            if Config.no_progressbar:
                self.__progressbar = EmptyProgressbar()
            else:
                self.__progressbar = ProgressBar(self.__term, 'Tests')
        except ValueError:
            report_info("Failed to initialize advanced progressbar")
            self.__progressbar = SimpleProgressBar('Tests  ')

    def run(self):

        #
        # Run tests
        #
        cur_time = time()

        # commands = [test.get_run_command() for test in self.__tests]
        # cwds = [test.get_cwd() for test in self.__tests]
        # results = asyncExecuteCommand(commands, cwds)

        for i in range(len(self.__tests)):
            self.__test_count += 1

            test = self.__tests[i]
            file = test.get_file()

            self.__progressbar.update(i / len(self.__tests), str(file))

            # Execute test
            has_failure = False
            stdout = str()
            stderr = str()

            try:
                for cmd in test.get_run_command():

                    if Config.verbose:
                        self.__progressbar.clear()
                        report_info("Running TEST: %s" % ' '.join(cmd))

                    (out, err, exit_code) = executeCommand(cmd, cwd=test.get_cwd())

                    # Check return code
                    if exit_code != test.expected_exit_code():
                        if exit_code == -SIGSEGV:
                            err += "Segmentation fault"

                        self.add_failure(file, "EXECUTION",
                                         "expected error code '%i', got '%i':\n%s" % (
                                             test.expected_exit_code(), exit_code, err))

                        has_failure = True
                        break

                    # Generate reference files
                    if Config.generate_reference:
                        for files in test.get_expected_file():
                            for outputfile, referencefile in zip(files.get_output_files(),
                                                                 files.get_reference_files()):
                                rename(outputfile, referencefile)

                    stdout += out
                    stderr += err

            except FileNotFoundError as e:
                self.add_failure(file, "EXECUTION", "%s" % e)
                has_failure = True

            if has_failure:
                continue

            # Check expected files
            if test.get_expected_file() and not Config.generate_reference:
                for files in test.get_expected_file():
                    for outputfile, referencefile in zip(files.get_output_files(),
                                                         files.get_reference_files()):

                        try:
                            output_json = load(open(outputfile, 'r'))
                            reference_json = load(open(referencefile, 'r'))
                            # Json does not enforce sorting, so we sort the files by sorting here
                            output_json = loads(dumps(output_json, sort_keys=True))
                            reference_json = loads(dumps(reference_json, sort_keys=True))
                        except FileNotFoundError as e:
                            self.add_failure(file, "EXECUTION", "%s" % e)
                            break

                        # Filter ignored nodes
                        if files.get_ignored_nodes():
                            for node in files.get_ignored_nodes():
                                del output_json[node]
                                del reference_json[node]

                        output, reference = dumps(output_json, indent=2), dumps(reference_json, indent=2)

                        if output != reference:
                            output = list(map(lambda x : x + "\n", output.split("\n")))
                            reference = list(map(lambda x : x + "\n", reference.split("\n")))
                            msg = "\n"
                            for line in unified_diff(output,
                                                     reference,
                                                     fromfile=outputfile,
                                                     tofile=referencefile):
                                msg += line

                            self.add_failure(file, "EXPECTED_FILE", msg)
                            has_failure = True
                        else:
                            remove(outputfile)
                            for discarded_file in files.get_discarded_files():
                                remove(discarded_file)

            # Check expected output
            out = stdout.split('\n')

            if test.get_expected_output():
                for expected_output in test.get_expected_output():
                    if expected_output not in out:
                        msg = "expected:\n\"%s\"" % expected_output
                        msg += self.__get_closest_match(expected_output, out)

                        self.add_failure(file, "EXPECTED", msg)
                        has_failure = True
                        break

            if test.get_expected_accesses():
                for expected_accesses in test.get_expected_accesses():
                    prefix = expected_accesses.get_prefix()
                    line_idx = -1
                    for o in out:
                        if o.startswith(prefix):
                            line_idx = out.index(o)
                            break

                    if line_idx == -1:
                        msg = "expected: \"%s\"" % prefix
                        msg += self.__get_closest_match(expected_accesses.get_prefix(), out)
                        self.add_failure(file, "EXPECTED_ACCESSES", msg)
                        has_failure = True
                        break

                    line = out[line_idx]
                    line = line[line.find(prefix) + len(prefix):].strip().split(' ')
                    for access in expected_accesses.get_expected_output():
                        if access not in line:
                            msg = "expected in line %s \n\"%s\"" % (
                                expected_accesses.get_line_number(), access)
                            msg += self.__get_closest_match(access, line)
                            self.add_failure(file, "EXPECTED_ACCESSES", msg)
                            has_failure = True
                            break

            # Check expected error output
            if test.get_expected_error_output():
                for expected_error in test.get_expected_error_output():
                    if "error: " + expected_error not in stderr:
                        msg = "expected error message:\n\"%s\"\n\ngot:\n%s" % (
                            expected_error, stderr)

                        self.add_failure(file, "EXPECTED_ERROR", msg)
                        has_failure = True
                        break

            if not has_failure:
                self.__test_count_passed += 1

        self.__progressbar.clear()
        self.__test_time = (time() - cur_time) * 1000

        #
        # Report results
        #
        t = self.__term
        print(t.BOLD + t.GREEN + "[==========]" + t.NORMAL + " %i test%s ran. (%i ms total)" % (
            self.__test_count, "s" if self.__test_count > 1 else "", self.__test_time))

        # Report passed test
        if self.__test_count_passed > 0:
            print(t.BOLD + t.GREEN + "[  PASSED  ] " + t.NORMAL + "%i test%s." % (
                self.__test_count_passed, "s" if self.__test_count_passed > 1 else ""))

        # Report failed tests
        if self.__failure_map:
            self.report_failure()
            return 1
        else:
            return 0

    def add_failure(self, file, error_type, msg):
        self.__failure_map[file][error_type] = msg

    def report_failure(self):
        t = self.__term

        test_count_failed = self.__test_count - self.__test_count_passed
        print(t.BOLD + t.RED + "[  FAILED  ] " + t.NORMAL + "%i test%s." % (
            test_count_failed, "s" if test_count_failed > 1 else ""))

        for key, test_failaure_map in self.__failure_map.items():
            print(t.BOLD + t.RED + "[----------] " + t.NORMAL)
            print(t.BOLD + t.RED + "[  FAILED  ] " + t.NORMAL + key)

            report = lambda type, msg: print(
                self.__split_line(t.BOLD + type + ": " + t.NORMAL + value, 13, 160))

            for error_type, value in test_failaure_map.items():
                if error_type is "EXECUTION":
                    report("Error", value)
                if error_type is "EXPECTED":
                    report("Output mismatch", value)
                if error_type is "EXPECTED_ACCESSES":
                    report("Access output mismatch", value)
                if error_type is "EXPECTED_FILE":
                    report("File mismatch", value)
                if error_type is "EXPECTED_ERROR":
                    report("Error mismatch", value)

    def __split_line(self, line, indent, max_line):
        ws = indent * ' '
        new_line = str()

        for subline in line.split('\n'):

            new_sub_line = ws
            cur_len = len(new_sub_line)

            for token in subline.split(' '):
                if cur_len + 1 + len(token) <= max_line:
                    cur_len += len(token) + 1
                    new_sub_line += token + " "
                else:
                    line_to_append = "\n" + ws + token
                    cur_len = len(line_to_append)
                    new_sub_line += line_to_append

            new_line += new_sub_line + "\n"

        return new_line[:-1]

    def __get_closest_match(self, output, out):
        edit_distances = []
        for o in out:
            edit_distances += [levenshtein(output, o)]

        if min(edit_distances) < 10:
            return "\nclosest match:\n\"" + out[edit_distances.index(min(edit_distances))] + "\""
        else:
            return ""
