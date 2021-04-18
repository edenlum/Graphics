import numpy as np
import sys
from graphics import *
from numpy.linalg import norm
from matplotlib import pyplot as plt
from PIL import Image

DTYPE = np.float32


class Scene:

    def __init__(self, width: int, height: int):
        self.width = width  # number of pixels
        self.height = height  # number of pixels
        self.materials = []
        self.spheres = []
        self.boxes = []
        self.planes = []
        self.lights = []

    def parse_scene(self, scene_name):
        with open(scene_name, 'r') as scene:
            for line in scene:
                if len(line) == 0 or line[0] == "#":
                    continue  # we skip comment lines
                line = line.split()
                if len(line) == 0:
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

    def ray_cast_vec(self, ):
        shape = (self.width, self.height, 3)
        pixel_size = self.camera.width / self.width
        self.camera.calc_axis()
        n = self.height * self.width
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        rays_p0 = np.tile(self.camera.pos, (n, 1))  # shape of  n,3
        print(rays_p0.dtype)
        grid_points = create_grid((self.width, self.height), self.camera.Vx[np.newaxis, np.newaxis, :], \
                                  self.camera.Vy[np.newaxis, np.newaxis, :], np.array([pixel_size]), \
                                  center_screen[np.newaxis, :], noise=False)
        rays_v = grid_points.reshape((n, 3)) - rays_p0
        # normalization
        rays_v = normalize(rays_v)
        colors = np.zeros(shape, dtype=DTYPE)
        reflections = np.ones(shape, dtype=DTYPE)

        rays_v = rays_v.astype(dtype=DTYPE)
        rays_p0 = rays_p0.astype(dtype=DTYPE)

        mask = np.ones(n, dtype=bool)

        for i in range(self.Settings.rec_level):
            print(rays_v.dtype)
            mat_idxs, normals, t = np.zeros(n, dtype=np.int16), np.zeros((n, 3)), np.full(n, np.inf)
            mat_idxs[mask], normals[mask], t[mask] = self.find_intersection_vec(rays_p0[mask], rays_v[mask])
            mask = np.isfinite(t)
            hit_points = rays_p0[mask] + rays_v[mask] * t[mask, np.newaxis]
            curr_colors = np.full((n, 3), self.Settings.bg)
            curr_colors[mask] = self.get_color_vec(hit_points, mat_idxs[mask], normals[mask], -rays_v[mask])
            colors += reflections * curr_colors.reshape(shape)
            reflections *= self.mat_ref[mat_idxs - 1].reshape(shape)
            rays_p0[mask] = hit_points
            rays_v[mask] = reflection(normals[mask], rays_v[mask])

        colors = np.minimum(colors, np.ones(shape))

        return colors

    def find_intersection_vec(self, p0: np.array, v: np.array, shadow=False):
        # if(rec>self.Settings.rec_level):
        # return None
        n = p0.shape[0]
        inters = []
        mat_idxs = []
        normals = []

        mask = np.ones(n, dtype=bool)

        for obj in self.spheres + self.planes + self.boxes:
            normal, inter = np.zeros((n, 3)), np.full(n, np.inf)
            inter[mask], mat_idx, normal[mask] = obj.intersect_vec(p0[mask], v[mask])
            if shadow:
                mask = inter > 0

            inters.append(inter)
            mat_idxs.append(mat_idx)
            normals.append(normal)

        inters = np.stack(inters)
        inters = np.where(inters > 0, inters, np.inf)
        indices = inters.argmin(axis=0)
        mat_idxs = np.array(mat_idxs)
        mat_idxs = np.tile(mat_idxs, (n, 1))
        mat_idxs = mat_idxs[np.arange(n), indices]
        normals = np.stack(normals)
        normals = normals[indices, np.arange(n)].reshape(n, 3)
        t = inters[indices, np.arange(n)]
        return mat_idxs, normals, t

    def get_color_vec(self, hit_points, mat_idxs, normals, camera_vec):
        # getting n hit points and normals - shape (n,3), mat_idxs of shape n
        # returns color for each ray (n,3)
        mat_idxs = mat_idxs - 1
        n = hit_points.shape[0]
        shadows = self.Settings.shadow_num
        color = np.zeros((n, 3))
        hit_points = hit_points + 0.0001 * normals
        for light in self.lights:
            d = normalize(light.pos[np.newaxis, :] - hit_points)
            # d is of shape (n, 3)
            dir1 = np.cross(d, np.array([1, 0, 0]))
            dir2 = np.cross(d, dir1)
            light_pixel = light.radius / shadows
            light_pos = create_grid((shadows, shadows), dir1[:, np.newaxis, :], dir2[:, np.newaxis, :],
                                    np.array([light_pixel]), light.pos, noise=True)
            rays_p0 = np.repeat(hit_points, repeats=shadows ** 2, axis=0)
            rays_v = normalize(light_pos.reshape(n * shadows ** 2, 3) - rays_p0)  # shape is (n*shadow^2 ,3)

            # calculates intersections from hit_point to light source
            _, _, t = self.find_intersection_vec(rays_p0, rays_v, shadow=True)
            t = t.reshape((n, shadows ** 2))

            precent = np.sum(np.isinf(t), axis=1) / shadows ** 2
            light_intensity = (1 - light.shadow) * 1 + light.shadow * precent  # shape is (n)

            cos = np.sum(d * normals, axis=1)  # shape is (n,)
            background = self.Settings.bg[np.newaxis, :] * self.mat_trans[mat_idxs][:, np.newaxis]
            diffuse = self.mat_dif[mat_idxs] * cos[:, np.newaxis]
            r = normalize(reflection(normals, -d))  # reflected normalized ray from light source to hit point
            phi = np.sum(r * camera_vec, axis=1)  # angle between reflected ray from light source and vector to camera
            spec = self.mat_spec[mat_idxs] * (phi ** self.mat_phong[mat_idxs])[:, np.newaxis]
            hit_color = light.color[np.newaxis, :] * background + light.color[np.newaxis, :] * (diffuse + spec) * (1 - self.mat_trans[mat_idxs])[:, np.newaxis]
            color += light_intensity[:, np.newaxis] * (np.where(np.isinf(hit_points), self.Settings.bg, hit_color))

        color = np.minimum(color, np.ones((n, 3)))
        return color


def create_grid(res, dir1: np.array, dir2: np.array, pixel_size: np.array, center: np.array, noise: bool):
    # creates a grid of points in 3d space
    # dir is of shape (n, l, 3)
    w, h = res
    n = dir1.shape[0]
    w_size = w * pixel_size  # shape of l
    h_size = h * pixel_size
    corner = center[np.newaxis, :] - dir1 * ((w_size - pixel_size) / 2)[np.newaxis, :, np.newaxis] - dir2 * ((h_size - pixel_size) / 2)[np.newaxis,:, np.newaxis]  # of shape (n, l, 3)
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    noise_shape = (n, w, h)
    if noise:
        noise_x = np.random.uniform(-0.5, 0.5, size=noise_shape).astype(dtype=DTYPE)
        noise_y = np.random.uniform(-0.5, 0.5, size=noise_shape).astype(dtype=DTYPE)
    else:
        noise_x = np.zeros(noise_shape).astype(dtype=DTYPE)
        noise_y = np.zeros(noise_shape).astype(dtype=DTYPE)
    x = (x[np.newaxis, np.newaxis, :, :, np.newaxis] + noise_x[:, np.newaxis, :, :, np.newaxis]) * dir1[:, :, np.newaxis, np.newaxis, :] * pixel_size[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]  # shape of (n, l, w, h, 3)
    y = (y[np.newaxis, np.newaxis, :, :, np.newaxis] + noise_y[:, np.newaxis, :, :, np.newaxis]) * dir2[:, :, np.newaxis, np.newaxis, :] * pixel_size[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]  # shape of (n, l, w, h, 3)
    points = x + y + corner[:, :, np.newaxis, np.newaxis, :]
    return points


def main():
    if len(sys.argv) < 3:
        print(
            "Not enough arguments provided. Please specify an input scene file and an output image file for rendering.")
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
    image = np.clip(image, 0, 1)
    print(time.time() - s)
    result = Image.fromarray(np.uint8(image * 255), mode='RGB')  # plt.imshow(image,origin='lower')
    result = result.transpose(Image.FLIP_TOP_BOTTOM)
    result.save(image_name)  # plt.show()
    plt.imshow(image, origin='lower')
    plt.savefig("pyplot_try.png")


if __name__ == "__main__":
    main()
