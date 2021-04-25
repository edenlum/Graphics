import numpy as np
import sys
from graphics import *
from numpy.linalg import norm
from matplotlib import pyplot as plt
from PIL import Image
from fish_eye import fish_eye

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
        self.objects = self.spheres + self.boxes + self.planes
        self.vectorize_attributes()

    def vectorize_attributes(self):
        # creates np.arrays of shape (m, 3) or (m) where m is the number of materials/lights
        # each array contains the value of a field for all the materials by order
        dif, spec, ref, phong, trans, amb = [], [], [], [], [], []
        for mat in self.materials:
            dif.append(mat.dif)
            spec.append(mat.spec)
            ref.append(mat.ref)
            phong.append(mat.phong)
            trans.append(mat.trans)
            amb.append(mat.amb)

        light_pos, light_rad, light_shadow, light_color = [], [], [], []
        for light in self.lights:
            light_pos.append(light.pos)
            light_rad.append(light.radius)
            light_shadow.append(light.shadow)
            light_color.append(light.color)

        self.mat_dif = np.stack(dif)
        self.mat_spec = np.stack(spec)
        self.mat_ref = np.stack(ref)
        self.mat_amb = np.stack(amb)
        self.mat_phong = np.array(phong)
        self.mat_trans = np.array(trans)
        self.light_pos = np.array(light_pos)
        self.light_radius = np.array(light_rad)
        self.light_shadow = np.array(light_shadow)
        self.light_color = np.stack(light_color)

    def ray_cast(self, ):
        shape = (self.width, self.height, 3)
        pixel_size = self.camera.width / self.width
        self.camera.calc_axis()
        n = self.height * self.width
        center_screen = self.camera.pos + self.camera.dist * self.camera.Vz
        rays_p0 = np.tile(self.camera.pos, (n, 1))  # shape of  n,3
        grid_points = create_grid((self.width, self.height), self.camera.Vx[np.newaxis, np.newaxis, :],
                                  self.camera.Vy[np.newaxis, np.newaxis, :], np.array([pixel_size]),
                                  center_screen[np.newaxis, :], noise=False)
        if self.camera.fish_eye:
            rays_v, mask = fish_eye(self.camera.pos, self.camera.k, grid_points.reshape((n, 3)), center_screen, self.camera.dist)
        else:
            rays_v = grid_points.reshape((n, 3)) - rays_p0
            rays_v = normalize(rays_v)
            mask = np.ones(n, dtype=bool)

        rays_v = rays_v.astype(dtype=DTYPE)
        rays_p0 = rays_p0.astype(dtype=DTYPE)

        colors = np.zeros((n, 3))
        colors[mask] = self.color_ray(rays_p0, rays_v, mask=mask)[mask]
        return colors.reshape(shape)

    def color_ray(self, rays_p0, rays_v, intersections=None, index=0, rec_depth=0, mask=None):
        """
        for each ray we calculate color like so:
        color_ray(ray, depth=0)
            c = 0
            find first intersection.
            c += color(intersection) * (1-trans)
            calculate reflection ray.
            c += reflection_color * color_ray(reflection_ray, depth = depth+1)
            find next intersection with original ray
            c += color_ray(trans_ray) * trans
        """
        if mask is None:
            n = rays_p0.shape[0]
        else:
            n = mask.size
        if index >= len(self.objects) or rec_depth > self.Settings.rec_level:
            return np.full((n, 3), self.Settings.bg)
        print(f"Recursion level: {rec_depth}, and transparency level: {index}")
        color = np.full((n, 3), self.Settings.bg)
        if intersections is None:
            mat_idxs, normals, t = self.find_intersection(rays_p0, rays_v)
            intersections = (mat_idxs, normals, t)  # save all the values for later use
        else:
            mat_idxs, normals, t = intersections  # retrieve values from previous calculation
        mat_idxs, normals, t = mat_idxs[index], normals[index], t[index]  # get the intersection depth we are interested in
        if mask is None:
            mask = np.isfinite(t)
        else:
            mask = mask * np.isfinite(t)
        if not mask.any():  # if all mask is 0, no need to calculate anything
            return np.full((n, 3), self.Settings.bg)
        hit_points = rays_p0[mask] + rays_v[mask] * t[mask, np.newaxis]
        color[mask] = self.get_color(hit_points, mat_idxs[mask], normals[mask], rays_v[mask])
        # reflections
        ref_mask = np.sum(self.mat_ref[mat_idxs[mask] - 1], axis=1) > 0  # at least one color > 0
        reflections_p0 = hit_points[ref_mask]
        ref_mask = np.logical_and(np.sum(self.mat_ref[mat_idxs - 1], axis=1) > 0, mask)
        reflections_v = reflection(normals[ref_mask], rays_v[ref_mask])
        color[ref_mask] += self.mat_ref[mat_idxs[ref_mask] - 1] * self.color_ray(reflections_p0, reflections_v, rec_depth=rec_depth+1)
        # transparency
        trans_mask = np.logical_and(mask, self.mat_trans[mat_idxs - 1] > 0)
        color[trans_mask] += self.mat_trans[mat_idxs - 1][trans_mask, np.newaxis] * self.color_ray(rays_p0, rays_v, intersections=intersections,
                                                                                       index=index+1, rec_depth=rec_depth, mask=trans_mask)[trans_mask]
        return color


    def find_intersection(self, p0: np.array, v: np.array, shadow=False):
        n = p0.shape[0]
        inters = []
        mat_idxs = []
        normals = []

        for obj in self.objects:
            inter, mat_idx, normal = obj.intersect(p0, v, calc_normal=(not shadow))

            inters.append(inter)
            mat_idxs.append(mat_idx)
            normals.append(normal)

        inters = np.stack(inters)
        indices = inters.argsort(axis=0)
        if not shadow:  # if shadow then we don't care about these values
            mat_idxs = np.array(mat_idxs)
            mat_idxs = np.tile(mat_idxs, (n, 1))
            mat_idxs = np.take_along_axis(mat_idxs.T, indices, axis=0)
            normals = np.stack(normals)
            normals_indices = np.repeat(indices, 3, axis=1).reshape(normals.shape)
            normals = np.take_along_axis(normals, normals_indices, axis=0)
        t = np.take_along_axis(inters, indices, axis=0)
        if shadow:
            t = t[0]
        return mat_idxs, normals, t

    def get_color(self, hit_points, mat_idxs, normals, camera_vec):
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
            distance2light = light_pos.reshape(n * shadows ** 2, 3) - rays_p0
            rays_v = normalize(distance2light)  # shape is (n*shadow^2 ,3)

            # calculates intersections from hit_point to light source
            _, _, t = self.find_intersection(rays_p0, rays_v, shadow=True)
            # t = t.reshape((n, shadows ** 2))
            cond = t > np.sqrt(np.sum(distance2light**2, axis=1))

            precent = np.sum(cond.reshape((n, shadows**2)), axis=1) / shadows ** 2
            light_intensity = (1 - light.shadow) * 1 + light.shadow * precent  # shape is (n)

            cos = np.sum(d * normals, axis=1)  # shape is (n,)
            # background = self.Settings.bg[np.newaxis, :] * self.mat_trans[mat_idxs][:, np.newaxis]
            diffuse = self.mat_dif[mat_idxs] * cos[:, np.newaxis]
            r = normalize(reflection(normals, -d))  # reflected normalized ray from light source to hit point
            phi = np.sum(r * camera_vec, axis=1)  # angle between reflected ray from light source and vector to camera
            spec = self.mat_spec[mat_idxs] * (phi ** self.mat_phong[mat_idxs])[:, np.newaxis]
            hit_color = light.color[np.newaxis, :] * (diffuse + spec) * (1 - self.mat_trans[mat_idxs])[:, np.newaxis]
            color += light_intensity[:, np.newaxis] * hit_color
            # adding ambient lighting
            color += self.get_ambient_color(hit_points, self.mat_amb[mat_idxs])

        color = np.minimum(color, np.ones((n, 3)))
        return color


    def get_ambient_color(self, hit_points, mat_amb):
        # gets array of points position, and material color for each point.
        # calculates the ambient light color for each point by averaging over all light sources (closer = more weight)
        # multiplies light color by material color and by ambient intensity and returns colors for each point.
        n = hit_points.shape[0]
        color = np.zeros((n, 3))
        total_d_sq = np.zeros(n)
        for light in self.lights:
            d_sq = np.sum((light.pos[np.newaxis, :] - hit_points)**2, axis=1)
            color += light.color[np.newaxis, :] * d_sq[:, np.newaxis]
            total_d_sq += d_sq
        color /= total_d_sq[:, np.newaxis]  # normalizing
        color *= mat_amb  # multiplying by material ambient color
        color *= self.Settings.amb_intensity  # multiplying by total ambient intensity in our scene
        return color




def create_grid(res, dir1: np.array, dir2: np.array, pixel_size: np.array, center: np.array, noise: bool):
    # creates a grid of points in 3d space
    # dir is of shape (n, l, 3)
    w, h = res
    n = dir1.shape[0]
    w_size = w * pixel_size  # shape of l
    h_size = h * pixel_size
    corner = center[np.newaxis, :] - dir1 * (w_size/ 2)[np.newaxis, :, np.newaxis] - dir2 * (h_size / 2)[np.newaxis,:, np.newaxis]  # of shape (n, l, 3)
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
    image = scene.ray_cast()
    image = np.clip(image, 0, 1)
    print(time.time() - s)
    result = Image.fromarray(np.uint8(image * 255), mode='RGB')  # plt.imshow(image,origin='lower')
    result = result.transpose(Image.FLIP_TOP_BOTTOM)
    result.save(image_name)
    result.show()


if __name__ == "__main__":
    main()
