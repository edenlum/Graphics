from PIL import Image
import numpy as np
import sys


def main():
    name1, name2 = sys.argv[1:]

    img1 = Image.open(name1)
    img2 = Image.open(name2)

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    diff = np.maximum(arr1, arr2) - np.minimum(arr1, arr2)

    diff_image = Image.fromarray(diff)
    diff_image.show()


if __name__ == "__main__":
    main()
