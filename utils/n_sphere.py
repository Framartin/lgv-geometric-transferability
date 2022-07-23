# N-sphere Convert to Spherical or Rectangular Coordination
# improve n-sphere package with numerical stability and basic vectorization: https://pypi.org/project/n-sphere/

import numpy as np
import math
import torch

SUPPORTED_TYPES = ['Tensor', 'ndarray', 'list']


def convert_spherical(input, digits=6, tol=1e-8):
    input_type = type(input).__name__
    if input_type not in SUPPORTED_TYPES:
        raise ValueError("Unsupported type")
    result = []
    if input_type == 'list':
        input = np.array(input)
    # over 2-dimension (current available 2-dimension)
    if (input.ndim == 1):
        r = 0
        for i in range(0, len(input)):
            r += input[i] * input[i]
        r = math.sqrt(r)
        convert = [r]
        for i in range(0, len(input) - 2):
            convert.append(round(math.acos(input[i] / (r + tol)), digits))
            r = math.sqrt(r * r - input[i] * input[i])
        if input[-2] >= 0:
            convert.append(round(math.acos(input[-2] / (r + tol)), digits))
        else:
            convert.append(round(2 * math.pi - math.acos(input[-2] / (r + tol)), digits))
        result = convert
        if input_type == 'ndarray':
            result = np.array(result)
        elif input_type == 'Tensor':
            result = torch.stack(result)
    else:
        result = np.zeros(input.shape)
        ssq_cum = np.sum(input[:,-2:] ** 2, axis=1)
        result[:, -1] = np.arccos(input[:,-2] / np.sqrt(ssq_cum + tol))
        mask = input[:, -1] < 0
        result[mask, -1] = 2 * np.pi - result[mask, -1]
        for i in range(2, input.shape[1]):
            ssq_cum = ssq_cum + input[:, -i-1]**2
            result[:, -i] = np.arccos(input[:, -i-1] / np.sqrt(ssq_cum + tol))
        result[:,0] = np.sqrt(ssq_cum)
        if input_type == 'Tensor':
            result = torch.from_numpy(result)
    return result


def convert_rectangular(input, digits=6):
    input_type = type(input).__name__
    if input_type not in SUPPORTED_TYPES:
        raise ValueError("Unsupported type")
    if input_type == 'list':
        input = np.array(input)
    if input.ndim == 1:
        result = []
        r = input[0]
        multi_sin = 1
        convert = []
        if input_type == 'Tensor':
            for i in range(1, len(input) - 1):
                convert.append(r * multi_sin * math.cos(input[i]))
                multi_sin *= math.sin(input[i])
            convert.append(r * multi_sin * math.cos(input[-1]))
            convert.append(r * multi_sin * math.sin(input[-1]))
            convert = np.array(convert)
            convert = torch.from_numpy(convert)
        else:
            for i in range(1, len(input) - 1):
                convert.append(round(r * multi_sin * math.cos(input[i]), digits))
                multi_sin *= math.sin(input[i])
            convert.append(round(r * multi_sin * math.cos(input[-1]), digits))
            convert.append(round(r * multi_sin * math.sin(input[-1]), digits))
            if input_type != 'list':
                convert = np.array(convert)
        result = convert
    else:
        # over 2-dimension
        result = np.zeros(input.shape)
        r = input[:, 0]
        multi_sin = np.zeros(input.shape[0]) + 1
        for i in range(1, input.shape[1] - 1):
            result[:, i - 1] = r * multi_sin * np.cos(input[:, i])
            multi_sin *= np.sin(input[:, i])
        result[:, i] = r * multi_sin * np.cos(input[:, i + 1])
        result[:, i + 1] = r * multi_sin * np.sin(input[:, i + 1])
        if input_type == 'Tensor':
            result = torch.from_numpy(result)
    return result
