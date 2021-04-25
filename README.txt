Ambient Lighting:

In order to add ambient lighting to each hit point, 
we will estimate the ambient light in this point by performing a weighting average of the colors of all the lights.
The weights for each light are 1/r^2 when r is the distance to each light.

We add to the final color the term (material.ambient_color) * (ambient_color(hit_point)) * Ambient intensity

In order to calculate the ambient light we will add RGB values for ambient color of each material (default 0), and Ambient intensity in settings (0-1)