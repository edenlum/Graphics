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

    def ray_cast(self,):
        image = np.zeros((self.width, self.height, 3))
        pixel_size = self.camera.width / self.width
        height = self.height * pixel_size
        self.camera.calc_axis()
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        print(center_screen)
        p0 = center_screen - self.camera.Vx * self.camera.width/2 - self.camera.Vy * height/2
        print(p0)
        for i in range(self.width):
            p = p0.copy()
            for j in range(self.height):
                ray = Ray(self.camera.pos, p - self.camera.pos)
                hit_point, mat_idx = self.find_intersection(ray)
                image[i, j, :] = self.get_color(mat_idx)
                p += self.camera.Vx * pixel_size
            p0 += self.camera.Vy * pixel_size
        return image


    def find_intersection(self, ray : Ray):
        min_t = np.Infinity
        mat_idx = None
        for sphere in self.spheres:
            inter = sphere.intersect2(ray)
            if inter == None: continue
            t, idx = inter
            # print(t)
            if t < min_t and t>=0:
                min_t = t
                mat_idx = idx
        for plane in self.planes:
            inter = plane.intersect(ray)
            t, idx = inter
            if t < min_t and t>=0:
                min_t = t
                mat_idx = idx
        hit_point = ray.p0 + ray.vec*min_t
        return hit_point, mat_idx

    def get_color(self, mat_idx):
        if mat_idx == None:
            return self.Settings.bg
        color = self.materials[mat_idx-1].get_color(self.Settings.bg)
        # color = np.minimum(color, np.ones(3))
        # color /= np.max(color)
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
    image = scene.ray_cast()
    plt.imshow(image)
    plt.show()
