#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##===-----------------------------------------------------------------------------*- Python -*-===##
##                          _ 
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

import os
import re

from docutils.parsers.rst import Directive, directives
from docutils.transforms import Transform
from docutils.utils.error_reporting import SafeString, ErrorString
from docutils import io, nodes

from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.roles import XRefRole
from sphinx.util.nodes import make_refnode
from sphinx import addnodes

class CMakeModule(Directive):
    """ Declare the cmake-module directive
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'encoding': directives.encoding}

    def __init__(self, *args, **keys):
        self.re_start = re.compile(r'^#\[(?P<eq>=*)\[\.rst:$')
        Directive.__init__(self, *args, **keys)

    def run(self):
        settings = self.state.document.settings
        if not settings.file_insertion_enabled:
             raise self.warning('"%s" directive disabled.' % self.name)

        env = self.state.document.settings.env
        rel_path, path = env.relfn2path(self.arguments[0])
        path = os.path.normpath(path)
        encoding = self.options.get('encoding', settings.input_encoding)
        e_handler = settings.input_encoding_error_handler
        try:
            settings.record_dependencies.add(path)
            f = io.FileInput(source_path=path, encoding=encoding,
                             error_handler=e_handler)
        except UnicodeEncodeError as error:
            raise self.severe('Problems with "%s" directive path:\n'
                              'Cannot encode input file path "%s" '
                              '(wrong locale?).' %
                              (self.name, SafeString(path)))
        except IOError as error:
            raise self.severe('Problems with "%s" directive path:\n%s.' %
                      (self.name, ErrorString(error)))
        raw_lines = f.read().splitlines()
        f.close()
        rst = None
        lines = []

        for line in raw_lines:
            if rst is not None and rst != '#':
                # Bracket mode: check for end bracket
                pos = line.find(rst)
                if pos >= 0:
                    if line[0] == '#':
                        line = ''
                    else:
                        line = line[0:pos]
                    rst = None
            else:
                # Line mode: check for .rst start (bracket or line)
                m = self.re_start.match(line)
                if m:
                    rst = ']%s]' % m.group('eq')
                    line = ''
                elif line == '#.rst:':
                    rst = '#'
                    line = ''
                elif rst == '#':
                    if line == '#' or line[:2] == '# ':
                        line = line[2:]
                    else:
                        rst = None
                        line = ''
                elif rst is None:
                    line = ''
            lines.append(line)
        if rst is not None and rst != '#':
            raise self.warning('"%s" found unclosed bracket "#[%s[.rst:" in %s' %
                               (self.name, rst[1:-1], path))

        self.state_machine.insert_input(lines, path)
        return []

def setup(app):
    app.add_directive('cmake-module', CMakeModule)
    # app.add_transform(CMakeTransform)
    # app.add_transform(CMakeXRefTransform)
    # app.add_domain(CMakeDomain)