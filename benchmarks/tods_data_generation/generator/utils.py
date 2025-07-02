from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Vector = np.ndarray
Matrix = np.ndarray
pi = np.pi


def series_segmentation(
    data: Matrix,
    stepsize: int = 1,
) -> Tuple[Matrix, Matrix]:
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def sine(
    length: int,
    freq: float = 0.04,
    coef: float = 1.5,
    offset: float = 0.0,
    noise_amp: float = 0.05,
) -> Vector:
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)

    if noise_amp != 0:
        noise = np.random.normal(loc=0, scale=1, size=length)
        value = value + noise_amp * noise
    value = coef * value + offset

    return value


def cosine(
    length: int,
    freq: float = 0.04,
    coef: float = 1.5,
    offset: float = 0.0,
    noise_amp: float = 0.05,
) -> Vector:
    timestamp = np.arange(length)
    value = np.cos(2 * np.pi * freq * timestamp)

    if noise_amp != 0:
        noise = np.random.normal(loc=0, scale=1, size=length)
        value = value + noise_amp * noise
    value = coef * value + offset

    return value


def square_sine(
    level: int = 5,
    length: int = 500,
    freq: float = 0.04,
    coef: float = 1.5,
    offset: float = 0.0,
    noise_amp: float = 0.05,
) -> Vector:
    value = np.zeros(length)

    for i in range(level):
        value += (1 / (2*i + 1)) * sine(
            length=length,
            freq=freq*(2*i + 1),
            coef=coef,
            offset=offset,
            noise_amp=noise_amp,
        )
    
    return value


def collective_global_synthetic(
    length: int,
    base: Vector,
    coef: float = 1.5,
    noise_amp: float = 0.005,
) -> Vector:
    value = []
    norm = np.linalg.norm(base)
    base = base / norm
    num = length // len(base)
    residual = length % len(value)

    for i in range(num):
        value.extend(base)
    
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(loc=0, scale=1, size=length)
    value = coef * value + noise_amp * noise

    return value
