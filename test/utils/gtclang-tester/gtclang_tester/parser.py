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

from os import path, listdir
from re import compile
from re import split
from tempfile import mkdtemp

from .config import Config
from .error import report_warning, report_fatal_error, report_info


def parse(dirs):
    """ Parse files in the given directories and return a list of Test objects """
    tests = []
    for dir in dirs:
        if path.isdir(dir) and path.isabs(dir):
            report_info("Parsing directory %s ... " % dir)

            # Parse configuration file (gtclang-tester.cfg)
            config = ConfigParser(dir)

            for file in listdir(dir):
                if path.isdir(path.join(dir, file)):
                    continue

                # Files not ending with .cpp are skipped
                if not file.endswith(".cpp"):
                    continue

                if config.exclude(file):
                    report_info("Skipping file %s ... " % file)
                    continue

                report_info("Parsing file %s ... " % file)
                parser = Parser(dir, file)
                parser.parse()
                if parser.get_test_config().is_valid():
                    tests += [parser.get_test_config()]
                report_info("Done parsing file %s ... " % file)
            report_info("Done parsing directory %s ... " % dir)
        else:
            report_warning("'%s' is not a full path directory" % dir)

    return tests


class Parser(object):
    """ Parse a file and create the corresponding Test object """

    def __init__(self, dir, filename):
        self.__dir = dir
        self.__file = path.join(dir, filename)
        self.__test = Test(self.__file)

    def parse(self):
        with open(self.__file, 'r') as f:
            lines = f.readlines()

            for i in range(len(lines)):
                line = lines[i]
                self.__parse_line(line, i + 1)  # Clang line numbers start from 1

    def __parse_line(self, line, linenumber):
        run_idx = line.find("RUN:")
        if run_idx >= 0:
            line = line[run_idx + len("RUN:"):].rstrip()
            report_info("Found RUN at %i: \"%s\"" % (linenumber, line))

            if self.__test.is_valid():
                report_fatal_error("multiple \"RUN:\" arguments in file '%s'" % self.__file)

            run_command = self.__substitute_keywords(line, linenumber)
            report_info("Parsed RUN as: \"%s\"" % run_command)
            self.__test.add_run_command(run_command)

        #
        # EXPECTED
        #
        expected_idx = line.find("EXPECTED:")
        if expected_idx >= 0:
            line = line[expected_idx + len("EXPECTED:"):].rstrip()
            report_info("Found EXPECTED at %i: \"%s\"" % (linenumber, line))

            expected_output = self.__substitute_keywords(line, linenumber)
            report_info("Parsed EXPECTED as: \"%s\"" % expected_output)
            self.__test.add_expected_command(expected_output)

        #
        # EXPECTED_ERROR
        #
        expected_error_idx = line.find("EXPECTED_ERROR:")
        if expected_error_idx >= 0:
            line = line[expected_error_idx + len("EXPECTED_ERROR:"):].rstrip()
            report_info("Found EXPECTED_ERROR at %i: \"%s\"" % (linenumber, line))

            expected_error_output = self.__substitute_keywords(line, linenumber)
            report_info("Parsed EXPECTED_ERROR as: \"%s\"" % expected_error_output)
            self.__test.add_expected_error_command(expected_error_output)

        #
        # EXPECTED_ACCESSES
        #
        expected_accesses_idx = line.find("EXPECTED_ACCESSES:")
        if expected_accesses_idx >= 0:
            line = line[expected_accesses_idx + len("EXPECTED_ACCESSES:"):].rstrip()
            report_info("Found EXPECTED_ACCESSES at %i: \"%s\"" % (linenumber, line))

            expected_accesses = self.__substitute_keywords(line, linenumber)

            parsed_expected_accesses = self.__test.add_expected_accesses_command(linenumber,
                                                                                 expected_accesses)
            report_info("Parsed EXPECTED_ACCESSES as: \"%s %s\"" % (
                parsed_expected_accesses.get_prefix(),
                parsed_expected_accesses.get_expected_output()))

        #
        # EXPECTED_FILE
        #
        expected_file_idx = line.find("EXPECTED_FILE:")
        if expected_file_idx >= 0:
            line = line[expected_file_idx + len("EXPECTED_FILE:"):].rstrip()
            report_info("Found EXPECTED_FILE at %i: \"%s\"" % (linenumber, line))

            expected_file = self.__substitute_keywords(line, linenumber)

            report_info("Parsed EXPECTED_FILE as: \"%s\"" % expected_file)
            self.__test.add_expected_file_command(expected_file, self.__dir, self.__file,
                                                  linenumber)
    

    def __parse_line_with_number(self, line, linenumber):
        found = line.find("%line")
        if found == -1:
            return line
        char = -1
        words  = line.split()
        for word in words:
            char += len(word)
            if char >= found :
                match = word
                break
        splits = match.split("+")
        if len(splits) == 1:
            splits = match.split("-")
        else:
            number = splits[1][:-2]
            line = line.replace(match, str(linenumber+int(number))+":")
        if len(splits) == 1:
            found = line.find("%line%")
            if found == -1:
                report_fatal_error("Bad line-stmt in " +line)
            line = line.replace("%line%", str(linenumber))
        else:
            number = splits[1][:-2]
            line = line.replace(match, str(linenumber-int(number))+":")

        return line


    def __substitute_keywords(self, line, linenumber):
        """ Substitute keywords """

        # %gtclang%
        line = line.replace("%gtclang%", Config.gtclang)

        # %c++%
        line = line.replace("%c++%", Config.cxx)

        # %gridtools_flags%
        line = line.replace("%gridtools_flags%", Config.gridtools_flags)

        # %file%
        line = line.replace("%file%", self.__file)

        # %filename%
        line = line.replace("%filename%", path.splitext(path.basename(self.__file))[0])

        # %filedir%
        line = line.replace("%filedir%", path.dirname(self.__file))

        # %line[+-Val]%
        line = self.__parse_line_with_number(line, linenumber)

        # %tmpdir%
        line = line.replace("%tmpdir%", mkdtemp())

        # Remove \t \n
        line = line.replace("\n", "")
        line = line.replace("\t", "")

        return line.lstrip()

    def get_test_config(self):
        return self.__test



class ExpectedAccesses(object):
    """ Parsed access pattern """

    def __init__(self, line_number, line):
        self.__line_number = line_number
        self.__expected_output = []

        and_idx = line.find("%and%")
        while and_idx != -1:
            self.__expected_output.append(line[:and_idx].strip())
            line = line[and_idx + len("%and%"):]
            and_idx = line.find("%and%")
        self.__expected_output.append(line.strip())

    def get_prefix(self):
        return "ACCESSES: line %i:" % self.__line_number

    def get_line_number(self):
        return self.__line_number

    def get_expected_output(self):
        return self.__expected_output


class ExpectedFile(object):
    """ Parsed diff of file """

    def __init__(self, command, dir, file, linenumber):
        self.__file_output = []
        self.__file_reference = []
        self.__ignored_nodes = []
        self.__discarded_files = []

        commands = command.split(' ')
        for cmd in commands:
            output_idx = cmd.find("OUTPUT:")
            if output_idx >= 0:
                output_cmd = cmd[output_idx + len("OUTPUT:"):].rstrip()
                for output_file in output_cmd.split(','):
                    self.__file_output.append(path.join(dir, output_file))

            reference_idx = cmd.find("REFERENCE:")
            if reference_idx >= 0:
                reference_cmd = cmd[reference_idx + len("REFERENCE:"):].rstrip()
                for ref_file in reference_cmd.split(','):
                    self.__file_reference.append(path.join(dir, ref_file))

            ignore_idx = cmd.find("IGNORE:")
            if ignore_idx >= 0:
                ignored_nodes = cmd[ignore_idx + len("IGNORE:"):].rstrip()
                for node in ignored_nodes.split(','):
                    self.__ignored_nodes.append(node)

            delete_idx = cmd.find("DELETE:")
            if delete_idx >= 0:
                discarded_files = cmd[delete_idx + len("DELETE:"):].rstrip()
                for discarded_file in discarded_files.split(','):
                    self.__discarded_files.append(path.join(dir, discarded_file))

        if len(self.__file_output) != len(self.__file_reference):
            report_fatal_error(
                "%s:%i: mismatch of number of output and reference files" % (file, linenumber))

    def get_output_files(self):
        return self.__file_output

    def get_reference_files(self):
        return self.__file_reference

    def get_ignored_nodes(self):
        return self.__ignored_nodes

    def get_discarded_files(self):
        return self.__discarded_files


class Test(object):
    """ Parsed content of a test file """

    def __init__(self, file):
        self.__file = file

        self.__run_command = None
        self.__expected_output = []
        self.__expected_accesses = []
        self.__expected_file = []
        self.__expected_error_output = []

    def get_cwd(self):
        return path.dirname(self.__file)

    def get_filename(self):
        return path.basename(self.__file)

    def get_file(self):
        return self.__file

    def is_valid(self):
        """ Check if test is valid i.e has a run command """
        return self.__run_command is not None

    def add_run_command(self, command):
        self.__run_command = command

    def add_expected_command(self, command):
        self.__expected_output.append(command)

    def add_expected_accesses_command(self, line_number, line):
        expected_accesses = ExpectedAccesses(line_number, line)
        self.__expected_accesses.append(expected_accesses)
        return expected_accesses

    def add_expected_file_command(self, command, dir, file, linenumber):
        self.__expected_file.append(ExpectedFile(command, dir, file, linenumber))

    def add_expected_error_command(self, command):
        self.__expected_error_output.append(command)

    def get_run_command(self):
        """ Get the run command(s), pipes '|' will be split in sub commands """
        cmds = []
        for cmd in self.__run_command.split('|'):
            cmds += [cmd.strip().split(' ')]
        return cmds

    def expected_exit_code(self):
        """ 0 if expected_error_output is empty, 1 otherwise """
        return 0 if len(self.__expected_error_output) is 0 else 1

    def get_expected_output(self):
        return self.__expected_output

    def get_expected_accesses(self):
        return self.__expected_accesses

    def get_expected_file(self):
        return self.__expected_file

    def get_expected_error_output(self):
        return self.__expected_error_output


class ConfigParser(object):
    """ Parse configuration file in given directory """

    def __init__(self, dir):
        config_file = path.join(dir, "gtclang-tester.cfg")
        self.__exclude_pattern = []

        if path.isfile(config_file):
            with open(config_file, 'r') as cfg:
                for line in cfg.readlines():

                    # Parse "EXCLUDE=" pattern
                    exclude_idx = line.find("EXCLUDE=")
                    if exclude_idx >= 0:
                        line = line[exclude_idx + len("EXCLUDE="):].rstrip()

                        # Remove '"'
                        if line.startswith('"'):
                            line = line[1:]
                        if line.endswith('"'):
                            line = line[:-1]

                        # Create regex
                        self.__exclude_pattern.append(compile(line))

    def exclude(self, file):
        """ Check whether this file will be skipped """

        # We skip files matching *.~
        if file.endswith('~'):
            return True

        # We skip files which match the provided pattern
        for pattern in self.__exclude_pattern:
            if pattern.match(file):
                return True
        return False
