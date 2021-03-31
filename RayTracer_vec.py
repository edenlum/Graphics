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


    def ray_cast_vec(self,):
        image = np.zeros((self.width, self.height, 3))
        pixel_size = self.camera.width / self.width
        height = self.height * pixel_size
        self.camera.calc_axis()
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        p0 = center_screen - self.camera.Vx * self.camera.width/2 - self.camera.Vy * height/2
        rays_p0 = np.tile(self.camera.pos, (self.height*self.width,1))
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)
        x = x[:, :, np.newaxis]*self.camera.Vx[np.newaxis, np.newaxis, :]*pixel_size
        y = y[:, :, np.newaxis]*self.camera.Vy[np.newaxis, np.newaxis, :]*pixel_size
        rays_v = x + y + p0
        rays_v = rays_v.reshape((self.height*self.width,3)) - rays_p0

        rays_v = rays_v / np.sqrt(np.sum(rays_v**2, axis=1))[:, np.newaxis]
        if (np.any(np.sum(rays_v**2, axis=1) > 1)):
            print(rays_v)

        mat_idxs, normals, hit_points = self.find_intersection_vec(rays_p0, rays_v)
        mat_idxs = mat_idxs.reshape(self.height, self.width)
        normals = normals.reshape(self.height, self.width,3)
        hit_points = hit_points.reshape(self.height, self.width,3)
        for i in range(self.height):
            for j in range(self.width):
                image[i,j,:] = self.materials[mat_idxs[i,j]-1].get_color(self.Settings.bg)
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
        indeces = inters.argmin(axis=0)
        mat_idxs = np.array(mat_idxs)
        mat_idxs = np.tile(mat_idxs, (n, 1))
        mat_idxs = mat_idxs[np.arange(n), indeces]
        normals = np.stack(normals)
        normals = normals[indeces, np.arange(n)].reshape(n,3)
        t = inters[indeces, np.arange(n)]
        hit_points = p0 + v*t[:, np.newaxis]
        return mat_idxs, normals, hit_points


    def get_color(self, hit_point, mat_idx, normal):
        # color = self.materials[mat_idx-1].get_color(self.Settings.bg)
        mat = self.materials[mat_idx-1]
        color = np.zeros(3)
        for light in self.lights:
            hit_count=0
            d = light.pos - hit_point
            d /= norm(d)
            dir1=np.cross(d,np.array([1,0,0]))
            dir2=np.cross(d,dir1)
            p0=light.pos-dir1*light.radius/2-dir2*light.radius/2
            for i in range(self.Settings.shadow_num):
                p=p0.copy()+i*light.radius/self.Settings.shadow_num*dir1
                for j in range(self.Settings.shadow_num):
                    d = p-hit_point
                    d /= norm(d)
                    if(self.find_intersection(Ray(hit_point,d))==None):
                        hit_count+=1
                    p+=light.radius/self.Settings.shadow_num*dir2
            precent=hit_count/self.Settings.shadow_num**2
            light_intensity=(1-light.shadow)*1+light.shadow*precent
            cos = np.dot(d, normal)
            if cos>0:
                color += light_intensity*(1-mat.trans) * cos*light.color*(mat.dif)#+ mat.spec*light.spec)
        color = np.minimum(color, np.ones(3))
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
    image = scene.ray_cast_vec()
    plt.imshow(image,origin='lower')
    plt.show()
