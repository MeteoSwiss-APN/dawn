#!/usr/bin/python3
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

from sys import path as sys_path

from dawn.config import __dawn_install_protobuf_module__

from dawn.error import Error

sys_path.insert(1, __dawn_install_protobuf_module__)

#
# Export all SIR classes
#
from .SIR_pb2 import *
from google.protobuf import json_format


class ParseError(Error):
    """ Thrown in case of a parsing error.
    """
    pass


def to_json(msg):
    """ Converts protobuf message to JSON format.

    :param msg: The protocol buffers message instance to serialize.
    :returns: A string containing the JSON formatted protocol buffer message.
    """
    return json_format.MessageToJson(msg)


def from_json(text: str, message_type):
    """ Parses a JSON representation of a protocol message and returns the parsed message.

    :param text: Text JSON representation of the message.
    :param message_type: The *type* of message to parse.
    :returns: The parsed message.
    :raises ParseError: Failed to parse JSON
    """
    msg = message_type()
    try:
        json_format.Parse(text, msg)
    except json_format.ParseError as e:
        raise ParseError(str(e))
    return msg


def makeLiteralAccessExpr(value: str, type: BuiltinType.TypeID) -> LiteralAccessExpr:
    """ Create a LiteralAccessExpr.

    :param value:    Value of the literal (e.g "1.123123").
    :param type:     Builtin type id of the literal.
    """
    builtin_type = BuiltinType()
    builtin_type.type_id = type

    literal = LiteralAccessExpr()
    literal.value = value
    literal.type.CopyFrom(builtin_type)
    return literal


__all__ = [
    # Protobuf messages
    'SIR',
    'Stencil',
    'Stmt',
    'Expr',
    'StencilFunction',
    'GlobalVariableMap',
    'SourceLocation',
    'AST',
    'Field',
    'Attributes',
    'BlockStmt',
    'BuiltinType',
    'SourceLocation',
    'Type',
    'LiteralAccessExpr',

    # Convenience functions
    'to_json',
    'from_json',
    'ParseError',
    'makeLiteralAccessExpr'
]
