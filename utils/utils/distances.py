import drjit as dr
from drjit.auto import ArrayXf, Float

def euclidean_squared_distance(x: ArrayXf, y: ArrayXf) -> Float:
    return dr.sum(dr.power(x - y, 2))

def euclidean_distance(x: ArrayXf, y: ArrayXf) -> Float:
    return dr.sqrt(euclidean_squared_distance(x, y))