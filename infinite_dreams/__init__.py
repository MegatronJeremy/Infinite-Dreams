# infinite_dreams/__init__.py
import importlib

import torch as _torch  # noqa: F401

__all__ = ["infinite_dreams_ext"]


def __getattr__(name: str):
    if name == "infinite_dreams_ext":
        return importlib.import_module(".infinite_dreams_ext", __name__)
    raise AttributeError(name)
