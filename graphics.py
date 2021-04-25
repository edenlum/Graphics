import numpy as np
from numpy.linalg import norm

DTYPE = np.float32

class Ray:
    def __init__(self, pos : np.array, vec : np.array):
        self.p0 = pos
        self.vec = vec
        self.vec /= norm(vec)


class Sphere:
    def __init__(self, x : str, y : str, z : str, radius : str, mat_idx : str):
        self.pos = np.array([float(x),float(y),float(z)], dtype=DTYPE)
        self.radius = float(radius)
        self.mat_idx = int(mat_idx)

    def intersect(self, p0 : np.array, v : np.array, calc_normal=True):
        # getting n rays, p0 and v are of shape (n,3)
        # returns t_min (array of shape n) and mat_idx
        n = p0.shape[0]
        t = np.full(n, np.inf, dtype=DTYPE)
        normals = None

        l = self.pos[np.newaxis, :] - p0
        t_ca = np.einsum("nd,nd->n", l, v)
        mask = (t_ca >= 0).astype(bool)
        d_sq = np.zeros(n, dtype=DTYPE)
        d_sq[mask] = np.einsum("nd,nd->n", l[mask], l[mask]) - t_ca[mask]**2
        mask = mask * (d_sq < self.radius**2)
        t_hc = np.sqrt(self.radius**2 - d_sq[mask])
        t[mask] = t_ca[mask] - t_hc

        if calc_normal:
            normals = np.zeros_like(v, dtype=DTYPE)
            hit_points = p0[mask] + v[mask]*t[mask, np.newaxis]
            normals[mask] = self.get_normal(hit_points)
        return t, self.mat_idx, normals


    def get_normal(self, p: np.array):
        d = p-self.pos
        n = normalize(d)
        return n



class Plane:
    def __init__(self, nx : str, ny : str, nz : str, offset : str, mat_idx : str):
        self.normal = np.array([float(nx), float(ny), float(nz)])
        self.normal /= norm(self.normal)
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)

    def intersect(self, p0: np.array, v: np.array, calc_normal=True):
        # getting n rays, p0 and v are of shape (n,3)
        # returns t_min (array of shape n) and mat_idx
        # according to equation (p0 + v*t) * Normal = offset -> t = (offset-N*p0)/N*v
        n = p0.shape[0]
        numerator = self.offset - np.sum(p0*self.normal[np.newaxis,:], axis=1)
        denom = np.sum(v*self.normal[np.newaxis,:], axis=1)
        t = np.divide(numerator, denom, out=np.full_like(numerator, -1), where=denom != 0)
        mask = (t > 0).astype(bool)
        t = np.where(t > 0, t, np.inf)
        return t, self.mat_idx, np.tile(self.get_normal(), (n, 1))

    def get_normal(self, *args):
        return self.normal


class Box:
    def __init__(self, x : str, y:str, z:str, size:str, mat_idx : str):
        self.pos = np.array([float(x),float(y),float(z)], dtype=DTYPE)
        self.x=float(x)
        self.y=float(y)
        self.z=float(z)
        self.size = float(size)
        self.mat_idx = int(mat_idx)

        # 6 planes, in order -x, x, -y, y, -z, z
        self.planes = [Plane('-1', '0', '0', str(-self.x + self.size/2), str(self.mat_idx)),
                       Plane('1', '0', '0', str(self.x + self.size/2), str(self.mat_idx)),
                       Plane('0', '-1', '0', str(-self.y + self.size/2), str(self.mat_idx)),
                       Plane('0', '1', '0', str(self.y + self.size/2), str(self.mat_idx)),
                       Plane('0', '0', '-1', str(-self.z + self.size/2), str(self.mat_idx)),
                       Plane('0', '0', '1', str(self.z + self.size/2), str(self.mat_idx))]

    def intersect(self, p0: np.array, v: np.array, calc_normal=True):
        # a box is 6 planes, we will calculate the hit point in each plane
        # and check if the hit point is inside the boundaries
        n = p0.shape[0]
        ts=[]
        normals = []

        for i, plane in enumerate(self.planes):
            t, _, normal = plane.intersect(p0, v)
            hit_points = p0 + v*t[:, np.newaxis]
            if(i<=1):
                condition = in_range(hit_points, [1,2], lowers=[self.y-self.size/2, self.z-self.size/2], uppers = [self.y+self.size/2, self.z+self.size/2])

            elif (i<=3):
                condition = in_range(hit_points, [0,2], lowers=[self.x-self.size/2, self.z-self.size/2], uppers = [self.x+self.size/2, self.z+self.size/2])

            else:
                condition = in_range(hit_points, [0,1], lowers=[self.x-self.size/2, self.y-self.size/2], uppers = [self.x+self.size/2, self.y+self.size/2])

            t = np.where(condition, t, np.inf)
            ts.append(t)
            normals.append(normal)

        ts=np.array(ts) # shape of (6, n)
        normals = np.array(normals) # shape of (6, n, 3)
        indices = np.argmin(ts, axis=0) # shape of n
        correct_t = ts[indices, np.arange(n)]
        correct_normals = normals[indices, np.arange(n)].reshape(n,3)

        return correct_t, self.mat_idx, correct_normals

    def get_normal(self, p:np.array):
        eps = 1e-3
        for plane in self.planes:
            normal = plane.get_normal()
            if abs(np.dot(normal, p) - plane.offset) < eps:
                return normal
        return np.zeros(3)


class Material:
    def __init__(self, dr: str, dg: str, db: str, sr: str, sg: str, sb: str, rr: str,
                 rg: str, rb: str, phong: str, trans: str, ar: str = 0, ag: str = 0, ab: str = 0):
        self.dif = np.array([float(dr), float(dg), float(db)])
        self.spec = np.array([float(sr), float(sg), float(sb)])
        self.ref = np.array([float(rr), float(rg), float(rb)])
        self.phong = float(phong)
        self.trans = float(trans)
        self.amb = np.array([float(ar), float(ag), float(ab)])


class Light:
    def __init__(self, x : str, y:str, z:str, r : str, g : str, b : str, spec : str,
                 shadow : str,radius : str):
        self.pos = np.array([float(x), float(y), float(z)])
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.color = np.array([float(r), float(g), float(b)])
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        self.spec = float(spec)
        self.shadow = float(shadow)
        self.factor = 1-float(shadow)
        self.radius = float(radius)


class Camera:
    def __init__(self, x : str, y:str, z:str, lx : str, ly:str, lz:str, ux : str, uy:str,
                 uz:str, dist : str, width : str, fish_eye: str = False, k : str = 0.5):
        self.pos = np.array([float(x), float(y), float(z)], dtype=DTYPE)
        self.x = float(x)
        self.z = float(z)
        self.y = float(y)
        self.lpos = np.array([float(lx), float(ly), float(lz)], dtype=DTYPE)
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        self.upos = np.array([float(ux), float(uy), float(uz)], dtype=DTYPE)
        self.ux = float(ux)
        self.uy = float(uy)
        self.uz = float(uz)
        self.dist = float(dist)
        self.width = float(width)
        self.fish_eye = True if fish_eye == "True" else False
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
    def __init__(self, bgr : str, bgg : str, bgb : str, shadow_num : str, rec_level : str, amb_intensity: str = '0'):
        self.bg = np.array([float(bgr), float(bgg), float(bgb)], dtype=DTYPE)
        self.shadow_num = int(shadow_num)
        self.rec_level = int(rec_level)
        self.amb_intensity = float(amb_intensity)


def reflection(normal : np.array, rays_v: np.array):
    direction = rays_v - 2*normal * np.sum(rays_v*normal,axis=1)[:,np.newaxis]
    #new_ray = direction
    return direction


def normalize(vec: np.array, norm_axis=1):
    # gets a tensor of arbitrary shape,
    # and returns the tensor with values normalized according to given axis
    norm = np.sqrt(np.sum(vec**2, axis=norm_axis, keepdims=True))
    return np.divide(vec, norm, out=np.zeros_like(vec), where=norm != 0)


def in_range(points: np.array, axises, lowers, uppers):
    condition = np.ones(points.shape[0])
    for i, axis in enumerate(axises):
        condition *= (points[:, axis] <= uppers[i])
        condition *= (points[:, axis] >= lowers[i])
    return condition


def distance(points1, points2):
    return np.sqrt(np.sum((points1 - points2)**2, axis=1))
