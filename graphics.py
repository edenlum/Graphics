import numpy as np


class Sphere:
    def __init__(self, x : string, y : string, z : string, radius : string, mat_idx : string):
        self.pos = (float(x),float(y),float(z))
        self.radius = float(radius)
        self.mat_idx = int(mat_idx)


class Plane:
    def __init__(self, nx : string, ny : string, nz : string, offset : string, mat_idx : string):
        self.normal = (float(nx), float(ny), float(nz))
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)


class Box:
    def __init__(self, x : string,y:string,z:string, sx : string,sy : string,sz : string, rotx : string,roty : string,
                 rotz : string, mat_idx : string):
        self.pos = (float(x),float(y),float(z))
        self.x=float(x)
        self.y=float(y)
        self.z=float(z)
        self.sx = float(sx)
        self.sy = float(sy)
        self.sz = float(sz)
        self.rotx = float(rotx)
        self.roty = float(roty)
        self.rotz = float(rotz)
        self.mat_idx = int(mat_idx)


class Material:
    def __init__(self, dr : string, dg : string, db : string, sr : string, sg : string, sb : string, rr : string,
                 rg : string, rb : string, phong : string, trans : string):
        self.dif = (float(dr), float(dg), float(db))
        self.spec = (float(sr), float(sg), float(sb))
        self.phong = float(phong)
        self.ref = (float(rr), float(rg), float(rb))
        self.trans = float(trans)


class Light:
    def __init__(self, x : string, y:string, z:string, r : string, g : string, b : string, spec : string,
                 shadow : string,radius : string):
        self.pos = (float(x), float(y), float(z))
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.color = (float(r), float(g), float(b))
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        self.spec = float(spec)
        self.shadow = float(shadow)
        self.factor = 1-float(shadow)
        self.radius = float(radius)


class Camera:
    def __init__(self, x : float,y:float,z:float, lx : float,ly:float,lz:float,ux : float,uy:float,
                 uz:float,dist : float,width : float, fish_eye : bool = False, k : float = 0.5):
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
        self.dist = dist
        self.width = width
        self.fish_eye = fish_eye
        self.k = k


class Settings:
    def __init__(self, bgr : float, bgg : float, bgb : float, shadow_num : int, rec_level : int):
        self.bg = (bgr, bgg, bgb)
        self.shadow_num = shadow_num
        self.rec_level = rec_level
