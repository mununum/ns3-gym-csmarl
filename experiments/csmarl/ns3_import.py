import os
import importlib.util

def _import_module(name, path):
    cwd = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(cwd, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_graph_module():
    return _import_module("graph", "../graph.py")