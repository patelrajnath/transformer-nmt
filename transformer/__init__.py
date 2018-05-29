#!/usr/bin/env 
# -*- coding: utf-8 -*-

"""
 Copyright (c) 2018 Raj Nath Patel
 Licensed under the GNU Public License.
 Author: Raj Nath Patel
 Email: patelrajnath (at) gmail (dot) com
 Created: 30/May/2018 03:25
 """
import importlib
import os

REGISTERED_CLASSES = []


def registered_class(cls):
    REGISTERED_CLASSES.append(cls)
    return cls


from os.path import dirname

pwd = dirname(__file__)
# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('transformer.' + module)

__all__ = [
    'REGISTERED_CLASSES'
]