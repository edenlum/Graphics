import numpy as np
import sys
from graphics import *
from numpy.linalg import norm
from matplotlib import pyplot as plt

class Scene:
    def __init__(self, width : int, height : int):
        self.width = width # number of pixels
        self.height = height # number of pixels
        self.materials = []
        self.spheres = []
        self.boxes = []
        self.planes = []
        self.lights = []

    def parse_scene(self, scene_name):
        with open(scene_name, 'r') as scene:
            for line in scene:
                if len(line)==0 or line[0] == "#":
                    continue # we skip comment lines
                line = line.split()
                if len(line)==0:
                    continue
                if line[0] == "cam":
                    self.camera = Camera(*line[1:])
                if line[0] == "set":
                    self.Settings = Settings(*line[1:])
                if line[0] == "mtl":
                    self.materials.append(Material(*line[1:]))
                if line[0] == "sph":
                    self.spheres.append(Sphere(*line[1:]))
                if line[0] == "pln":
                    self.planes.append(Plane(*line[1:]))
                if line[0] == "box":
                    self.boxes.append(Box(*line[1:]))
                if line[0] == "lgt":
                    self.lights.append(Light(*line[1:]))
        self.vectorize_materials()


    def vectorize_materials(self):
        # creates np.arrays of shape (m, 3) or (m) where m is the number of materials
        # each array contains the value of a field for all the materials by order
        dif, spec, ref, phong, trans = [], [], [], [], []
        for mat in self.materials:
            dif.append(mat.dif)
            spec.append(mat.spec)
            ref.append(mat.ref)
            phong.append(mat.phong)
            trans.append(mat.trans)
        self.mat_dif = np.stack(dif)
        self.mat_spec = np.stack(spec)
        self.mat_ref = np.stack(ref)
        self.mat_phong = np.array(phong)
        self.mat_trans = np.array(trans)


    def ray_cast_vec(self,):
        image = np.zeros((self.width, self.height, 3))
        pixel_size = self.camera.width / self.width
        height = self.height * pixel_size
        self.camera.calc_axis()
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        corner_screen = center_screen - self.camera.Vx * (self.camera.width+pixel_size)/2 - self.camera.Vy * (height+pixel_size)/2
        rays_p0 = np.tile(self.camera.pos, (self.height*self.width,1)) # shape of  n,3
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)
        x = x[:, :, np.newaxis]*self.camera.Vx[np.newaxis, np.newaxis, :]*pixel_size
        y = y[:, :, np.newaxis]*self.camera.Vy[np.newaxis, np.newaxis, :]*pixel_size
        rays_v = x + y + corner_screen
        rays_v = rays_v.reshape((self.height*self.width,3)) - rays_p0
        # normalization
        rays_v = normalize(rays_v)

        mat_idxs, normals, t = self.find_intersection_vec(rays_p0, rays_v)
        hit_points = rays_p0 + rays_v*t[:, np.newaxis]
        # mat_idxs = mat_idxs.reshape(self.height, self.width)
        # normals = normals.reshape(self.height, self.width,3)
        # hit_points = hit_points.reshape(self.height, self.width,3)
        colors = self.get_color_vec(hit_points, mat_idxs, normals)
        colors = colors.reshape(self.height, self.width, 3)
        for i in range(self.height):
            for j in range(self.width):
                image[i,j,:] = colors[i,j,:]
                # image[i,j,:] = self.get_color(hit_points[i,j,:], mat_idxs[i,j], normals[i,j,:])
        return image


    def find_intersection_vec(self, p0 : np.array, v: np.array):
        #if(rec>self.Settings.rec_level):
            #return None
        n = p0.shape[0]

        inters = []
        mat_idxs = []
        normals = []

        for sphere in self.spheres:
            inter, mat_idx, normal = sphere.intersect_vec(p0, v)
            inters.append(inter)
            mat_idxs.append(mat_idx)
            normals.append(normal)

        for plane in self.planes:
            inter, mat_idx, normal = plane.intersect_vec(p0, v)
            inters.append(inter)
            mat_idxs.append(mat_idx)
            normals.append(normal)

        inters = np.stack(inters)
        inters = np.where(inters>0, inters, np.inf)
        indices = inters.argmin(axis=0)
        mat_idxs = np.array(mat_idxs)
        mat_idxs = np.tile(mat_idxs, (n, 1))
        mat_idxs = mat_idxs[np.arange(n), indices]
        normals = np.stack(normals)
        normals = normals[indices, np.arange(n)].reshape(n,3)
        t = inters[indices, np.arange(n)]
        return mat_idxs, normals, t


    def get_color_vec(self, hit_points, mat_idxs, normals):
        # getting n hit points and normals - shape (n,3), mat_idxs of shape n
        # returns color for each ray (n,3)
        mat_idxs = mat_idxs-1
        n = hit_points.shape[0]
        shadows = 1#self.Settings.shadow_num
        color = np.zeros((n,3))
        for light in self.lights:
            d = normalize(light.pos[np.newaxis, :] - hit_points)
            # d /= np.sqrt(np.sum(d**2, axis=1))[:, np.newaxis]
            # d is of shape (n, 3)
            dir1=np.cross(d,np.array([1,0,0]))
            dir2=np.cross(d,dir1)
            corner_light=light.pos[np.newaxis, :]-dir1*light.radius/2-dir2*light.radius/2
            light_pixel = light.radius/shadows
            x = np.arange(shadows)
            y = np.arange(shadows)
            x, y = np.meshgrid(x, y)
            noise_x = np.random.uniform(size=(shadows, shadows))
            noise_y = np.random.uniform(size=(shadows, shadows))
            x = x + noise_x
            y = y + noise_y
            x = x[np.newaxis, :, :, np.newaxis]*dir1[:, np.newaxis, np.newaxis, :]*light_pixel
            y = y[np.newaxis, :, :, np.newaxis]*dir2[:, np.newaxis, np.newaxis, :]*light_pixel
            light_pos = x + y + corner_light[:, np.newaxis, np.newaxis, :] # shape is (n, shadow, shadow, 3)
            rays_p0 = np.repeat(hit_points, repeats=shadows**2, axis=0)
            rays_v = normalize(light_pos.reshape(n*shadows**2, 3) - rays_p0) # shape is (n*shadow^2 ,3)
            # rays_v = rays_v / np.sqrt(np.sum(rays_v**2, axis=1))[:, np.newaxis]
            _, _, t = self.find_intersection_vec(rays_p0, rays_v)
            t = t.reshape((n, shadows**2))
            precent = np.sum(np.isinf(t), axis=1)/shadows**2
            light_intensity=(1-light.shadow)*1+light.shadow*precent # shape is (n)
            cos = np.sum(d*normals, axis=1) # shape is (n,)
            a = light_intensity*(1-self.mat_trans[mat_idxs])
            b = cos[:, np.newaxis]*light.color[np.newaxis, :]
            c = self.mat_dif[mat_idxs, :]
            color += a[:, np.newaxis] * b * c#+ mat.spec*light.spec)
        color = np.minimum(color, np.ones((n,3)))
        # color /= np.max(color)
        # print(color)
        return color


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Not enough arguments provided. Please specify an input scene file and an output image file for rendering.")
        exit()
    scene_name = sys.argv[1]
    image_name = sys.argv[2]
    res = (500, 500)
    if len(sys.argv) > 3:
        res = (int(sys.argv[3]), int(sys.argv[3]))

    scene = Scene(res[0], res[1])
    scene.parse_scene(scene_name)
    # s = scene.spheres[0]
    # p0 = np.zeros((10,3))
    # v = np.arange(30).reshape(10,3)
    # print(s.intersect_vec(p0, v))
    # p0_rays = np.tile(np.array([1,2,3]), (10,1))
    # print(p0_rays)
    import time
    s = time.time()
    image = scene.ray_cast_vec()
    print(time.time() - s)
    plt.imshow(image,origin='lower')
    plt.show()
