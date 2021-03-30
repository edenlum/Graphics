import numpy as np
import sys
import graphics

options = {"cam" : Camera, "set" : , "mtl", "sph", "pln", "box", "lgt"}

def parse_scene(scene_name):
    with open(scene_name, 'r') as scene:
        for line in scene:
            if line[0] == "#" or len(line)==0:
                continue # we skip comment lines
            line = line.split()
            options[line[0]](*line[1:])




if __name__ == "__main__":
    scene_name = sys.argv[1]
    image_name = sys.argv[2]
    res = (500, 500)
    if len(sys.argv) > 3:
        res = (int(sys.argv[3]), int(sys.argv[3]))
