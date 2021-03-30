import numpy as np


class Sphere:
    def __init__(self, pos : Tuple, radius : float):
        self.pos = pos
        self.radius = radius

class Plane:
    def __init__(self, normal : Tuple, offset : float):
        self.normal = normal
        self.offset = offset

class Box:
    def __init__(self, pos : Tuple, scale : Tuple, rotation : Tuple):
        self.pos = pos
        self.scale = scale
        self.rotation = rotation
