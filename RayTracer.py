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
        p0 = center_screen - self.camera.Vx * self.camera.width/2 - self.camera.Vy * height/2
        for i in range(self.width):
            p = p0.copy()
            for j in range(self.height):
                ray = Ray(self.camera.pos, p - self.camera.pos)
                inter = self.find_intersection(ray)
                if inter == None:
                    image[i, j, :] = self.Settings.bg
                else:
                    hit_point, mat_idx, normal = inter
                    image[i, j, :] = self.get_color(hit_point, mat_idx, normal)
                p += self.camera.Vx * pixel_size
            p0 += self.camera.Vy * pixel_size
        return image


    def find_intersection(self, ray : Ray):
        #if(rec>self.Settings.rec_level):
            #return None
        min_t = np.Infinity
        mat_idx = None
        object = None

        for sphere in self.spheres:
            inter = sphere.intersect2(ray)
            if inter == None: continue
            t, idx = inter
            if t < min_t and t>0:
                min_t = t
                mat_idx = idx
                object = sphere

        for plane in self.planes:
            inter = plane.intersect(ray)
            t, idx = inter
            if t < min_t and t>0:
                min_t = t
                mat_idx = idx
                object = plane

        if object == None:
            return None
        hit_point = ray.p0 + ray.vec*min_t
        normal = object.get_normal(hit_point)
        return hit_point, mat_idx, normal

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
            light_intensity=(1-light.shadow)*(1+light.shadow)*precent
            cos = np.dot(d, normal)
            if cos>0:
                color += light_intensity*(1-mat.trans) * cos*light.color*(mat.dif+ mat.spec*light.spec)
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
    image = scene.ray_cast()
    plt.imshow(image,origin='lower')
    plt.show()
