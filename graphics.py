import numpy as np


class Sphere:
    def __init__(self, x : float, y : float, z : float, radius : float, mat_idx : int):
        self.pos = (x,y,z)
        self.radius = radius
        self.mat_idx = mat_idx


class Plane:
    def __init__(self, nx : float, ny : float, nz : float, offset : float, mat_idx : int):
        self.normal = (nx, ny, nz)
        self.offset = offset
        self.mat_idx = mat_idx


class Box:
    def __init__(self, x : float,y:float,z:float, sx : float,sy : float,sz : float, rotx : float,roty : float,rotz : float, mat_idx : int):
        self.pos = (x,y,z)
        self.x=x
        self.y=y
        self.z=z
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.rotx = rotx
        self.roty = roty
        self.rotz = rotz
        self.mat_idx = mat_idx


class Material:
    def __init__(self, dif : float, spec : float, phong : float,ref : float, trans : float):
        self.dif = dif
        self.spec = spec
        self.phong = phong
        self.ref = ref
        self.trans = trans


class Light:
    def __init__(self, x : float,y:float,z:float, color : float,spec : float,shadow : float,factor : float,radius : float):
        self.pos = (x, y, z)
        self.x = x
        self.y = y
        self.z = z
    def __init__(self, pos : Tuple, color : float, spec : float, shadow : float,factor : float, radius : float):
        self.pos = pos
        self.color = color
        self.spec = spec
        self.shadow = shadow
        self.factor = factor
        self.radius = radius


class Camera:
    def __init__(self, x : float,y:float,z:float, lx : float,ly:float,lz:float,ux : float,uy:float,
                 uz:float,dist : float,width : float):
        self.pos = (x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.lpos = (lx, ly, lz)
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.upos = (ux, uy, uz)
        self.ux = ux
        self.uy = uy
        self.uz = uz
    def __init__(self, pos : Tuple, LAP : Tuple,up : Tuple, dist : float, width : float):
        self.pos = pos
        self.LAP = LAP
        self.up = up
        self.dist = dist
        self.width = width


class Settings:
    def __init__(self, bg : float,shadow_num : int, rec_level : int, fish_eye : bool = False):
        self.bg = bg
        self.shadow_num = shadow_num
        self.rec_level = rec_level
        self.fish_eye = fish_eye
