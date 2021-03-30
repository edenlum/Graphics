import numpy as np


class Sphere:
    def __init__(self, x : str, y : str, z : str, radius : str, mat_idx : str):
        self.pos = (float(x),float(y),float(z))
        self.radius = float(radius)
        self.mat_idx = int(mat_idx)


class Plane:
    def __init__(self, nx : str, ny : str, nz : str, offset : str, mat_idx : str):
        self.normal = (float(nx), float(ny), float(nz))
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)


class Box:
    def __init__(self, x : str,y:str,z:str, sx : str,sy : str,sz : str, rotx : str,roty : str,
                 rotz : str, mat_idx : str):
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
    def __init__(self, dr : str, dg : str, db : str, sr : str, sg : str, sb : str, rr : str,
                 rg : str, rb : str, phong : str, trans : str):
        self.dif = (float(dr), float(dg), float(db))
        self.spec = (float(sr), float(sg), float(sb))
        self.phong = float(phong)
        self.ref = (float(rr), float(rg), float(rb))
        self.trans = float(trans)


class Light:
    def __init__(self, x : str, y:str, z:str, r : str, g : str, b : str, spec : str,
                 shadow : str,radius : str):
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
    def __init__(self, x : str, y:str, z:str, lx : str, ly:str, lz:str, ux : str, uy:str,
                 uz:str, dist : str, width : str, fish_eye : str = False, k : str = 0.5):
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
    def __init__(self, bgr : str, bgg : str, bgb : str, shadow_num : str, rec_level : str):
        self.bg = (float(bgr), float(bgg), float(bgb))
        self.shadow_num = int(shadow_num)
        self.rec_level = int(rec_level)
