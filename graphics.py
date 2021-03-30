import numpy as np
from numpy.linalg import norm


class Ray:
    def __init__(self, pos : np.array, vec : np.array):
        self.p0 = pos
        self.vec = vec
        self.vec /= norm(vec)


class Sphere:
    def __init__(self, x : str, y : str, z : str, radius : str, mat_idx : str):
        self.pos = np.array([float(x),float(y),float(z)])
        self.radius = float(radius)
        self.mat_idx = int(mat_idx)

    def intersect(self, ray : Ray):
        a = 1
        d = ray.p0 - self.pos
        b = 2*np.dot(ray.vec, d)
        c = np.sum(d**2) - self.radius **2
        sol = quad(a, b, c)
        if sol == None:
            return None
        t1, t2 =sol
        return min(t1, t2), self.mat_idx

    def intersect2(self, ray: Ray):
        l = self.pos - ray.p0
        t_ca = np.dot(l, ray.vec)
        if (t_ca < 0):
            return None
        d_sq = np.dot(l, l) - t_ca**2
        if (d_sq > self.radius**2):
            return None
        t_hc = np.sqrt(self.radius**2 - d_sq)
        return (t_ca - t_hc), self.mat_idx # there is also t_ca+t_hc

    def find_normal(self, p : np.array):
        n = (p-self.pos)/norm(p-self.pos)
        return n



class Plane:
    def __init__(self, nx : str, ny : str, nz : str, offset : str, mat_idx : str):
        self.normal = np.array([float(nx), float(ny), float(nz)])
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)

    def intersect(self, ray : Ray):
        t = (self.offset - np.dot(ray.p0, self.normal)) / (np.dot(ray.vec, self.normal))
        return t, self.mat_idx


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
        self.dif = np.array([float(dr), float(dg), float(db)])
        self.spec = np.array([float(sr), float(sg), float(sb)])
        self.ref = np.array([float(rr), float(rg), float(rb)])
        self.phong = float(phong)
        self.trans = float(trans)

    def get_color(self, bgc : np.array):
        self.color = bgc * self.trans + (self.dif + self.spec) * (1-self.trans) + self.ref
        return self.color



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
        self.pos = np.array([float(x), float(y), float(z)])
        self.x = float(x)
        self.z = float(z)
        self.y = float(y)
        self.lpos = np.array([float(lx), float(ly), float(lz)])
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        self.upos = np.array([float(ux), float(uy), float(uz)])
        self.ux = float(ux)
        self.uy = float(uy)
        self.uz = float(uz)
        self.dist = float(dist)
        self.width = float(width)
        self.fish_eye = bool(fish_eye)
        self.k = float(k)

    def calc_axis(self,):
        self.Vz = self.lpos - self.pos # direction the camera is looking at (viewing direciton)
        self.Vz /= norm(self.Vz)
        self.up = self.upos # up direction of the world
        self.Vx = np.cross(self.up, self.Vz) # left direction of the Camera
        self.Vx /= norm(self.Vx)
        self.Vy = np.cross(self.Vz, self.Vx) # up direction of the camera (Vx, Vy, Vz are orthogonal)
        self.Vy /= norm(self.Vy)


class Settings:
    def __init__(self, bgr : str, bgg : str, bgb : str, shadow_num : str, rec_level : str):
        self.bg = np.array([float(bgr), float(bgg), float(bgb)])
        self.shadow_num = int(shadow_num)
        self.rec_level = int(rec_level)


def reflection(normal : np.array, ray : Ray, point : np.array):
    direction = ray.vec - 2*normal * np.dot(ray.vec, normal)
    new_ray = Ray(point, direction)
    return new_ray


def quad(a,b,c):
    if(b**2>=4*a*c):
        return ((-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a))
    else:
        return None
