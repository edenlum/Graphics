import numpy as np


class Sphere:
    def __init__(self, x : string, y : string, z : string, radius : string, mat_idx : string):
        self.pos = (float(x),float(y),float(z))
        self.radius = float(radius)
        self.mat_idx = int(mat_idx)


class Plane:
    def __init__(self, nx : float, ny : float, nz : float, offset : float, mat_idx : int):
        self.normal = (float(nx), float(ny), float(nz))
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)


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
    def __init__(self, dr : float, dg : float, db : float, sr : float, sg : float, sb : float, rr : float, rg : float, rb : float, phong : float, trans : float):
        self.dif = (dr, dg, db)
        self.spec = (sr, sg, sb)
        self.phong = phong
        self.ref = (rr, rg, rb)
        self.trans = trans


class Light:
    def __init__(self, x : float, y:float, z:float, r : float, g : float, b : float, spec : float, shadow : float,radius : float):
        self.pos = (x, y, z)
        self.x = x
        self.y = y
        self.z = z
        self.color = (r, g, b)
        self.r = r
        self.g = g
        self.b = b
        self.spec = spec
        self.shadow = shadow
        self.factor = 1-shadow
        self.radius = radius


class Camera:
    def __init__(self, x : string, y:string, z:string, lx : string, ly:string, lz:string, ux : string, uy:string,
                 uz:string, dist : string, width : string, fish_eye : string = False, k : string = 0.5):
        self.pos = (float(x), float(y), float(z))
        self.x = float(x)
        self.z = float(z)
        self.y = float(y)
        self.lpos = (float(lx), float(ly), float(lz))
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        self.upos = (float(ux), float(uy), float(uz))
        self.ux = float(ux)
        self.uy = float(uy)
        self.uz = float(uz)
        self.dist = float(dist)
        self.width = float(width)
        self.fish_eye = bool(fish_eye)
        self.k = float(k)


class Settings:
    def __init__(self, bgr : string, bgg : string, bgb : string, shadow_num : string, rec_level : string):
        self.bg = (float(bgr), float(bgg), float(bgb))
        self.shadow_num = int(shadow_num)
        self.rec_level = int(rec_level)
