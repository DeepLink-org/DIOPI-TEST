# -*- coding: UTF-8 -*-
import math

from ctypes import c_float, c_double, c_int64, c_int32, c_bool, c_void_p, byref, pointer
from .diopi_runtime import Sizes, Scalar, Tensor, TensorHandle
from .utils import check_returncode, check_function, squeeze
from .utils import logger
from . import Dtype, raw_like
from collections import namedtuple
import numpy as np


def broadcast_out_size(size1, size2):
    sizeO = size1 if len(size1) > len(size2) else size2
    length = len(size2) if len(size1) > len(size2) else len(size1)
    idx = -1
    while length > 0:
        assert size1[idx] == size2[idx] or size1[idx] == 1 or size2[idx] == 1,\
            "size1 and size2 must be broadcastable"
        sizeO[idx] = size1[idx] if size2[idx] == 1 else size2[idx]
        idx -= 1
        length -= 1

    return sizeO


def reduce_op_process(input, dim=None, keepdim=False, dtype=None):
    sizeI = list(input.size())
    if dim is None:
        for i in range(0, len(sizeI)):
            sizeI[i] = 1
        dim = []
    elif isinstance(dim, list):
        for i in dim:
            sizeI[i] = 1
    else:
        sizeI[dim] = 1
        dim = [dim]

    if dtype is None:
        dtype = input.get_dtype()

    out = Tensor(sizeI, dtype)
    if not keepdim:
        squeeze(out)
    return dim, out


def fill(tensor, value):
    func = check_function("diopiFill")
    ret = func(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_returncode(ret)
    return tensor


def ones_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 1)
    return new_tensor


def zeros_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 0)
    return new_tensor


def unary_op(input, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle)

    check_returncode(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle,
                   other.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, other.tensor_handle)

    check_returncode(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None, dtype=None) -> Tensor:
    args = "input.context_handle, "
    if dtype is None:
        dtype = input.get_dtype()

    if inplace:
        out = input
    else:
        sizeI = input.size()
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, dtype)
        else:
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, dtype)
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(input.get_dtype(), other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"\

    if alpha is not None:
        alpha = Scalar(input.get_dtype(), alpha)
        args = args + ", byref(alpha)"

    func = check_function(call)
    ret = eval(f'func({args})')

    check_returncode(ret)
    return out


def relu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiRelu')
