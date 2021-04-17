import numpy as np
import sys
from graphics import *
from numpy.linalg import norm
from matplotlib import pyplot as plt
from PIL import Image

DTYPE = np.float32


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
        self.vectorize_attributes()


    def vectorize_attributes(self):
        # creates np.arrays of shape (m, 3) or (m) where m is the number of materials/lights
        # each array contains the value of a field for all the materials by order
        dif, spec, ref, phong, trans = [], [], [], [], []
        for mat in self.materials:
            dif.append(mat.dif)
            spec.append(mat.spec)
            ref.append(mat.ref)
            phong.append(mat.phong)
            trans.append(mat.trans)

        light_pos, light_rad, light_shadow, light_color = [], [], [], []
        for light in self.lights:
            light_pos.append(light.pos)
            light_rad.append(light.radius)
            light_shadow.append(light.shadow)
            light_color.append(light.color)

        self.mat_dif = np.stack(dif)
        self.mat_spec = np.stack(spec)
        self.mat_ref = np.stack(ref)
        self.mat_phong = np.array(phong)
        self.mat_trans = np.array(trans)
        self.light_pos = np.array(light_pos)
        self.light_radius = np.array(light_rad)
        self.light_shadow = np.array(light_shadow)
        self.light_color = np.stack(light_color)


    def ray_cast_vec(self,):
        image = np.zeros((self.width, self.height, 3), dtype=DTYPE)
        pixel_size = self.camera.width / self.width
        self.camera.calc_axis()
        n = self.height*self.width
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        rays_p0 = np.tile(self.camera.pos, (n,1)) # shape of  n,3
        grid_points = create_grid((self.width, self.height), self.camera.Vx[np.newaxis, np.newaxis, :], \
                    self.camera.Vy[np.newaxis, np.newaxis, :], np.array([pixel_size]),  \
                    center_screen[np.newaxis, :], noise=False)
        rays_v = grid_points.reshape((n,3)) - rays_p0
        # normalization
        rays_v = normalize(rays_v)
        colors=np.zeros((self.height, self.width,3), dtype=DTYPE)
        reflections=np.ones((self.height, self.width, 3), dtype=DTYPE)

        rays_v = rays_v.astype(dtype=DTYPE)
        rays_p0 = rays_p0.astype(dtype=DTYPE)

        for i in range(self.Settings.rec_level):
            print(rays_v.dtype)
            mat_idxs, normals, t = self.find_intersection_vec(rays_p0, rays_v)
            hit_points = rays_p0 + rays_v*t[:, np.newaxis]
            condition=np.isinf(t.reshape((self.height, self.width))[:,:,np.newaxis])
            color = self.get_color_vec(hit_points, mat_idxs, normals, -rays_v)
            colors += reflections*np.where(condition,np.full((self.height, self.width, 3),self.Settings.bg),
                                           color.reshape((self.height, self.width, 3)))
            reflections *= self.mat_ref[mat_idxs-1].reshape((self.height, self.width, 3))
            rays_p0=hit_points
            rays_v=reflection(normals,rays_v)


        colors = np.minimum(colors, np.ones((self.height, self.width, 3)))

        for i in range(self.height):
            for j in range(self.width):
                image[i,j,:] = colors[i,j,:]
        return image


    def find_intersection_vec(self, p0 : np.array, v: np.array, shadow = False):
        #if(rec>self.Settings.rec_level):
            #return None
        n = p0.shape[0]

        inters = []
        mat_idxs = []
        normals = []

        for object in self.spheres + self.planes + self.boxes:
            inter, mat_idx, normal = object.intersect_vec(p0, v)
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


    def get_color_vec(self, hit_points, mat_idxs, normals, camera_vec):
        # getting n hit points and normals - shape (n,3), mat_idxs of shape n
        # returns color for each ray (n,3)
        mat_idxs = mat_idxs-1
        n = hit_points.shape[0]
        shadows = self.Settings.shadow_num
        color = np.zeros((n,3))
        l = len(self.lights)
        hit_points = hit_points + 0.0001*normals
        # for light in self.lights:

        d = normalize(self.light_pos[np.newaxis, :, :] - hit_points[:, np.newaxis, :])
        # d is of shape (n, l, 3)
        dir1=np.cross(d,np.array([1,0,0]))
        dir2=np.cross(d,dir1)
        light_pixel = self.light_radius/shadows # shape of (l,)
        light_points = create_grid((shadows, shadows), dir1, dir2, light_pixel, self.light_pos, noise=True)
        rays_p0 = np.repeat(hit_points, repeats=l*shadows**2, axis=0)
        rays_v = normalize(light_points.reshape(n*l*shadows**2, 3) - rays_p0) # shape is (n*l*shadow^2 ,3)

        # calculates intersections from hit_point to light source
        _, _, t = self.find_intersection_vec(rays_p0, rays_v)
        t = t.reshape((n, l, shadows**2))

        precent = np.sum(np.isinf(t), axis=2)/shadows**2
        light_intensity = (1 - self.light_shadow)[np.newaxis, :] + self.light_shadow[np.newaxis, :] * precent # shape is (n, l)

        cos = np.sum(d*normals[:, np.newaxis, :], axis=2) # shape is (n,l)
        background = self.Settings.bg[np.newaxis, np.newaxis, :] * self.mat_trans[mat_idxs][:, np.newaxis, np.newaxis] # shape of (n, l, 3)
        diffuse = self.mat_dif[mat_idxs][:, np.newaxis, :] * cos [:, :, np.newaxis] # shape of (n, l, 3)
        r = normalize(reflection(np.repeat(normals, repeats = l, axis=0), (-d).reshape(n*l,3))) # reflected normalized ray from light source to hit point - shape (n*l , 3)
        phi = np.sum(r.reshape(n,l,3)*camera_vec[:, np.newaxis, :], axis=2) # angle between reflected ray from light source and vector to camera
        spec = self.mat_spec[mat_idxs][:, np.newaxis, :] * (phi**self.mat_phong[mat_idxs][:, np.newaxis])[:, :, np.newaxis]
        hit_color = self.light_color[np.newaxis, :, :] * background + self.light_color[np.newaxis, :, :] * (diffuse + spec) * (1 - self.mat_trans[mat_idxs])[:, np.newaxis, np.newaxis]
        color = light_intensity[:, :, np.newaxis]*(np.where(np.isinf(hit_points[:, np.newaxis, :]), self.Settings.bg, hit_color))
        color = np.sum(color, axis=1)
        return color


def create_grid(res, dir1: np.array, dir2: np.array, pixel_size: np.array, center: np.array, noise: bool):
    # creates a grid of points in 3d space
    # dir is of shape (n, l, 3)
    w, h = res
    w_size = w*pixel_size # shape of l
    h_size = h*pixel_size
    corner = center[np.newaxis, :] - dir1 * ((w_size - pixel_size)/2)[np.newaxis, :, np.newaxis] - dir2 * ((h_size - pixel_size)/2)[np.newaxis, :, np.newaxis] # of shape (n, l, 3)
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    if noise:
        noise_x = np.random.uniform(-0.5, 0.5, size=(w, h)).astype(dtype=DTYPE)
        noise_y = np.random.uniform(-0.5, 0.5, size=(w, h)).astype(dtype=DTYPE)
        x = x + noise_x
        y = y + noise_y
    x = x[np.newaxis, np.newaxis, :, :, np.newaxis]*dir1[:, :, np.newaxis, np.newaxis, :]*pixel_size[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] # shape of (n, l, w, h, 3)
    y = y[np.newaxis, np.newaxis, :, :, np.newaxis]*dir2[:, :, np.newaxis, np.newaxis, :]*pixel_size[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] # shape of (n, l, w, h, 3)
    points = x + y + corner[:, :, np.newaxis, np.newaxis, :]
    return points



def main():
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
    import time
    s = time.time()
    image = scene.ray_cast_vec()
    image = np.minimum(image, np.ones((res[0], res[1], 3)))
    print(time.time() - s)
    result = Image.fromarray(np.uint8(image*255), mode='RGB') #plt.imshow(image,origin='lower')
    result = result.transpose(Image.FLIP_TOP_BOTTOM)
    result.save(image_name) # plt.show()
    plt.imshow(image, origin='lower')
    plt.savefig("pyplot_try.png")

if __name__ == "__main__":
	main()
