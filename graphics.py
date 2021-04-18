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

    def intersect(self, ray : Ray):
        a = 1
        d = ray.p0 - self.pos
        b = 2*np.dot(ray.vec, d)
        c = np.sum(d**2) - self.radius **2
        sol = quad(a, b, c)
        if sol == None:
            return None
        t1, t2 = sol
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

    def intersect_vec(self, p0 : np.array, v : np.array):
        # getting n rays, p0 and v are of shape (n,3)
        # returns t_min (array of shape n) and mat_idx
        n = p0.shape[0]
        t_min = np.full(n, np.inf)
        normals = np.zeros_like(v)

        l = self.pos[np.newaxis, :] - p0
        t_ca = np.einsum("nd,nd->n", l, v)
        mask = t_ca > 0
        d_sq = np.zeros(n)
        d_sq[mask] = np.einsum("nd,nd->n", l[mask], l[mask]) - t_ca[mask]**2
        mask *= (d_sq <= self.radius**2)
        t_hc = np.sqrt(self.radius**2 - d_sq[mask])
        t_min[mask] = (t_ca[mask] - t_hc)
        hit_points = p0[mask] + v[mask]*t_min[mask, np.newaxis]
        normals[mask] = self.get_normal_vec(hit_points)
        return t_min, self.mat_idx, normals


    def get_normal_vec(self, p: np.array):
        d = p-self.pos
        n = normalize(d)
        return n

    def get_normal(self, p : np.array):
        n = (p-self.pos)/norm(p-self.pos)
        return n



class Plane:
    def __init__(self, nx : str, ny : str, nz : str, offset : str, mat_idx : str):
        self.normal = np.array([float(nx), float(ny), float(nz)])
        self.normal /= norm(self.normal)
        self.offset = float(offset)
        self.mat_idx = int(mat_idx)

    def intersect(self, ray : Ray):
        t = (self.offset - np.dot(ray.p0, self.normal)) / (np.dot(ray.vec, self.normal))
        return t, self.mat_idx

    def intersect_vec(self, p0: np.array, v: np.array):
        # getting n rays, p0 and v are of shape (n,3)
        # returns t_min (array of shape n) and mat_idx
        n = p0.shape[0]
        t = (self.offset - np.sum(p0*self.normal[np.newaxis,:], axis=1)) / np.sum(v*self.normal[np.newaxis,:], axis=1)
        return t, self.mat_idx, np.tile(self.get_normal(), (n,1))

    def get_normal(self):
        return self.normal


class Box:
    def __init__(self, x : str, y:str, z:str, size:str, mat_idx : str):
        self.pos = np.array([float(x),float(y),float(z)], dtype=DTYPE)
        self.x=float(x)
        self.y=float(y)
        self.z=float(z)
        self.size = float(size)
        self.mat_idx = int(mat_idx)
        print(str(self.x-self.size/2))

        # 6 planes, in order -x, x, -y, y, -z, z
        self.planes = [Plane('-1', '0', '0', str(-self.x + self.size/2), str(self.mat_idx)),
            Plane('1', '0', '0', str(self.x + self.size/2), str(self.mat_idx)),

                        Plane('0', '-1', '0', str(-self.y + self.size/2), str(self.mat_idx)),
                        Plane('0', '1', '0', str(self.y + self.size/2), str(self.mat_idx)),
                        Plane('0', '0', '-1', str(-self.z + self.size/2), str(self.mat_idx)),
                        Plane('0', '0', '1', str(self.z + self.size/2), str(self.mat_idx))]

    def intersect_vec(self, p0: np.array, v: np.array):
        # a box is 6 planes, we will calculate the hit point in each plane
        # and check if the hit point is inside the boundaries
        n = p0.shape[0]
        ts=[]
        normals = []

        for i, plane in enumerate(self.planes):
            t, _, normal = plane.intersect_vec(p0, v)
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


class Material:
    def __init__(self, dr : str, dg : str, db : str, sr : str, sg : str, sb : str, rr : str,
                 rg : str, rb : str, phong : str, trans : str):
        self.dif = np.array([float(dr), float(dg), float(db)])
        self.spec = np.array([float(sr), float(sg), float(sb)])
        self.ref = np.array([float(rr), float(rg), float(rb)])
        self.phong = float(phong)
        self.trans = float(trans)

    def get_color(self, bgc : np.array):
        # self.color = bgc * self.trans + (self.dif + self.spec) * (1-self.trans) + self.ref
        return self.dif



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
                 uz:str, dist : str, width : str, fish_eye : str = False, k : str = 0.5):
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
        self.bg = np.array([float(bgr), float(bgg), float(bgb)], dtype=DTYPE)
        self.shadow_num = int(shadow_num)
        self.rec_level = int(rec_level)


def reflection(normal : np.array, rays_v: np.array):
    direction = rays_v - 2*normal * np.sum(rays_v*normal,axis=1)[:,np.newaxis]
    #new_ray = direction
    return direction


def quad(a,b,c):
    if(b**2>=4*a*c):
        return ((-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a))
    else:
        return None


def normalize(vec : np.array, norm_axis=1):
    # gets a tensor of arbitrary shape,
    # and returns the tensor with values normalized according to given axis
    return vec/np.sqrt(np.sum(vec**2, axis=norm_axis, keepdims=True))


def in_range(points: np.array, axises, lowers, uppers):
    condition = np.ones(points.shape[0])
    for i, axis in enumerate(axises):
        condition *= (points[:, axis] <= uppers[i])
        condition *= (points[:, axis] >= lowers[i])
    return condition
