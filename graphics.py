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


class Material:
    def __init__(self, dif : float, spec : float,phong : float,ref : float,trans : float):
        self.dif = dif
        self.spec = spec
        self.phong = phong
        self.ref = ref
        self.trans = trans

class Light:
    def __init__(self, pos : Tuple, color : float,spec : float,shadow : float,factor : float,radius : float):
        self.pos = pos
        self.color = color
        self.spec = spec
        self.shadow = shadow
        self.factor = factor
        self.radius = radius

class Camera:
    def __init__(self, pos : Tuple, LAP : Tuple,up : Tuple,dist : float,width : float,bg : float,shadow_num : int,
                 rec_level : int,fish_eye : bool):
        self.pos = pos
        self.LAP = LAP
        self.up = up
        self.dist = dist
        self.width = width
        self.bg = bg
        self.shadow_num = shadow_num
        self.rec_level = rec_level
        self.fish_eye = fish_eye