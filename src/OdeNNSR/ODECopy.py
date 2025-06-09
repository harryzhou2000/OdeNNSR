from functools import singledispatch
import numpy as np
import torch
import copy

@singledispatch
def copy_data(data_object):
    """
    Generic copy function using singledispatch.
    Default implementation.
    """
    # print(f"Defaulting to copy.deepcopy() for {type(data_object)}")
    return copy.deepcopy(data_object)


@copy_data.register(np.ndarray)
def _(data_object: np.ndarray):
    # print(f"Using np.ndarray.copy() for {type(data_object)}")
    return data_object.copy()


@copy_data.register(torch.Tensor)
def _(data_object: torch.Tensor):
    # print(f"Using torch.Tensor.clone() for {type(data_object)}")
    return data_object.clone().detach()
