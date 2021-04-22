import numpy as np
from graphics import normalize


def fish_eye(camera_pos, k, pixel_points, center_screen, screen_dist):
    """consider the sensor plane (screen plane), pass a perpendicular line h to that plane and through the camera.
     call the point of intersection on the plane Q, and the distance between the camera and the plane f.
     for every pixel, R = length(pixel_pos - Q)
                            f/k * tan(k*theta) for k>0
     R also satisfies R = { f * theta          for k=0
                            f/k * sin(k*theta) for k<0
     k is a parameter between -1 and 1
     theta is the angle between line h and the ray of light that passes though the pixel
     we can compute theta from R by reversing the formula:
                              1/k * arctan(R*k/f) for k>0
                    theta = { R/f                 for k=0
                              1/k * arcsin(R*k/f) for k<0
     once we have theta, we can construct a ray by using v = sin(theta)*normalize(pixel_pos-Q) + cos(theta)*normalize(Q-camera) """
    Q = center_screen
    f = screen_dist
    R_vec = pixel_points - Q
    R = np.sqrt(np.sum(R_vec*R_vec, axis=1))
    if k > 0:
        theta = np.arctan(R*k/f)/k
    elif k == 0:
        theta = R/f
    else:
        theta = np.arcsin(R*k/f)/k

    # theta = np.where(theta<0, theta + np.pi*2, theta)

    new_R = f*np.tan(theta)
    mask = (new_R >= 0).astype(bool)
    new_pixel_points = Q[np.newaxis, :] + new_R[:, np.newaxis]*normalize(R_vec)
    v = normalize(new_pixel_points - camera_pos)
    return v, mask
