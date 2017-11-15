##
# Doxygen filter for Google Protocol Buffers .proto files.
# This script converts .proto files into C++ style ones
# and prints the output to standard output.
#
# version 0.6-beta
#
# How to enable this filter in Doxygen:
#   1. Generate Doxygen configuration file with command 'doxygen -g <filename>'
#        e.g.  doxygen -g doxyfile
#   2. In the Doxygen configuration file, find JAVADOC_AUTOBRIEF and set it enabled
#        JAVADOC_AUTOBRIEF      = YES
#   3. In the Doxygen configuration file, find FILE_PATTERNS and add *.proto
#        FILE_PATTERNS          = *.proto
#   4. In the Doxygen configuration file, find EXTENSION_MAPPING and add proto=C
#        EXTENSION_MAPPING      = proto=C
#   5. In the Doxygen configuration file, find INPUT_FILTER and add this script
#        INPUT_FILTER           = "python proto2cpp.py"
#   6. Run Doxygen with the modified configuration
#        doxygen doxyfile
#
#
# Copyright (C) 2012-2015 Timo Marjoniemi and Fabian Thuering
# All rights reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
##

import os
import sys
import re
import fnmatch
import inspect

class proto2cpp:
  logNone = 0
  logErrors = 1
  logAll= 2

  def __init__(self):
    self.logFile = "proto2cpp.log"
    self.errorLogFile = "proto2cpp.error.log"
    self.logLevel = self.logAll

  def handleFile(self, filename):
    if fnmatch.fnmatch(filename, '*.proto'):
      self.log('\n==========\n' + filename + '\n==========\n\n')

      try:
        with open(filename, 'r') as file:
          self.parseFile(file)
        pass
      except IOError as e:
        self.logError('the file ' + filename + ' could not be opened for reading')

    else:
      try:
        with open(filename, 'r') as theFile:
          for line in theFile:
            print(line)
            self.log(line + '\n')
        pass
      except IOError as e:
        self.logError('the file ' + filename + ' could not be opened for reading')

  ## Parser function.
  ## 
  ## The function takes a .proto file object as input
  ## parameter and modifies the contents into C++ style.
  ## The modified data is printed into standard output.
  ## 
  ## @param inputFile Input file object
  def parseFile(self, inputFile):
    isEnum = False

    # This variable is here as a workaround for not getting extra line breaks (each line
    # ends with a line separator and print() method will add another one).
    # We will be adding lines into this var and then print the var out at the end.
    theOutput = ''
    for line in inputFile:
      # Search for comment ("//")
      matchComment = re.search("//", line)

      # Search for semicolon and if one is found before comment.
      matchSemicolon = re.search(";", line)

      if matchSemicolon is not None and (matchComment is not None and matchSemicolon.start() < matchComment.start()):
         line = line[:matchComment.start()] + "///<" + line[matchComment.end():]
      elif matchComment is not None:
         line = line[:matchComment.start()] + "///" + line[matchComment.end():]

      # Replace "<type> <name> = <tag>;" with "<type> <name>;" i.e remove the tags
      matchTag = re.search("\s*=\s*\d*\s*;", line)
      if matchTag is not None:
        line = line[:matchTag.start()] + ";" + line[matchTag.end():]

      # Replace "oneof" with "union"
      matchOneof = re.search("oneof", line)
      if matchOneof is not None:
        line = line[:matchOneof.start()] + "union" + line[matchOneof.end():]

      # Search for "enum" and if one is found before comment,
      # start changing all semicolons (";") to commas (",").
      matchEnum = re.search("enum", line)
      if matchEnum is not None and (matchComment is None or matchEnum.start() < matchComment.start()):
        isEnum = True

      # Search again for semicolon if we have detected an enum, and replace semicolon with comma.
      if isEnum is True and re.search(";", line) is not None:
        matchSemicolon = re.search(";", line)
        line = line[:matchSemicolon.start()] + "," + line[matchSemicolon.end():]
      
      # Search for a closing brace.
      matchClosingBrace = re.search("}", line)
      if isEnum is True and matchClosingBrace is not None:
        line = line[:matchClosingBrace.start()] + "};" + line[matchClosingBrace.end():]
        isEnum = False
      elif isEnum is False and matchClosingBrace is not None:
        # Message (to be struct) ends => add semicolon so that it'll be a proper C(++) struct and 
        # Doxygen will handle it correctly. But don't fidle with braces in comments!
        if matchComment is None or matchComment.start() > matchClosingBrace.start():
          line = line[:matchClosingBrace.start()] + "};" + line[matchClosingBrace.end():]

      # Search for 'message' and replace it with 'struct' unless 'message' is behind a comment.
      matchMsg = re.search("message", line)
      if matchMsg is not None and (matchComment is None or matchMsg.start() < matchComment.start()):
        output = "struct" + line[:matchMsg.start()] + line[matchMsg.end():]
        theOutput += output
      else:
        theOutput += line

    lines = theOutput.splitlines()
    for line in lines:
      if len(line) > 0:
        print(line)
        self.log(line + '\n')
      else:
        self.log('\n')

  def log(self, string):
    if self.logLevel >= self.logAll:
      with open(self.logFile, 'a') as theFile:
        theFile.write(string)

  def logError(self, string):
    if self.logLevel >= self.logError:
      with open(self.errorLogFile, 'a') as theFile:
        theFile.write(string)

converter = proto2cpp()
for idx in range(1, len(sys.argv)):
  converter.handleFile(sys.argv[idx])
