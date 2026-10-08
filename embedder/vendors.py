"""Lazy, optional imports of the torch/transformers/open_clip stack.

The server must import (and serve /healthz, /v1/models, API validation) on
machines without torch — heavy deps are only imported when a model is loaded
or an embedding is computed. ImportError from this module means the embedder
deployment is missing deps.
"""

from __future__ import annotations


def torch_mod():
    import torch

    return torch


def np_mod():
    import numpy

    return numpy


def Image_cls():
    from PIL import Image

    return Image


def open_clip_mod():
    import open_clip

    return open_clip


def transformers_mod():
    import transformers

    return transformers


def qwen_vision_process():
    from qwen_vl_utils.vision_process import process_vision_info

    return process_vision_info
