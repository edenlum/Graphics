import numpy as np
import sys
import graphics

options = {"cam" : Camera, "set" : Settings, "mtl" : Material, "sph" : Sphere, "pln" : Plane, "box" : Box, "lgt" : Light}


class Scene:
    def __init__(self,):
        self.materials = []
        self.spheres = []
        self.boxes = []
        self.planes = []
        self.lights = []

    def parse_scene(scene_name):
        with open(scene_name, 'r') as scene:
            for line in scene:
                if line[0] == "#" or len(line)==0:
                    continue # we skip comment lines
                line = line.split()
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




if __name__ == "__main__":
    scene_name = sys.argv[1]
    image_name = sys.argv[2]
    res = (500, 500)
    if len(sys.argv) > 3:
        res = (int(sys.argv[3]), int(sys.argv[3]))
