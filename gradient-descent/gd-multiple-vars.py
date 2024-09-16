import numpy as np
import matplotlib.pyplot as plt


def z_function(x, y):
    return x**2 + y**2

def calculate_gradients(x, y):
    return 2*x, 2*y

x = np.arange(-100, 100, 0.1)
y = np.arange(-100, 100, 0.1)

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

current_position = (80, 80, z_function(80, 80))
learning_rate = 0.01

ax = plt.subplot(projection='3d', computed_zorder=False)

for _ in range(1000):
    grad_desc = calculate_gradients(current_position[0], current_position[1])
    new_x = current_position[0] - learning_rate * grad_desc[0]
    new_y = current_position[1] - learning_rate * grad_desc[1]
    new_z = z_function(new_x, new_y)
    current_position = (new_x, new_y, new_z)

    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.scatter(current_position[0], current_position[1], current_position[2], color='magenta', zorder=1)
    plt.pause(0.001)
    ax.clear()