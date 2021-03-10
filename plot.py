"""Plotting utilities"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('macosx')


def plot_value_function(x_range: range, y_range: range, V: np.ndarray, filename: str):
    """
    Plots the value function as a surface.

    :param range x_range: The range of x values
    :param range y_range: The range of y values
    :param np.ndarray V: The value function
    :param str filename: The name for the plot file
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x_range, y_range, indexing='ij')
    ax.plot_surface(x, y, V, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('Value')
    plt.savefig(f'output/{filename}.png', bbox_inches='tight')
