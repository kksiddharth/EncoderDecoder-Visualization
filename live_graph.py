#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:27:14 2019

@author: ts-siddharth.kumar
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import ast
#from mpl_toolkits.mplot3d import Axes3D 

#style.use('fivethirtyeight')

fig = plt.figure()
#ax1 = fig.add_subplot(111,projection='3d')
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    col = []
    for line in lines:
        if len(line) > 1:
            x, y, c = line.split('&')
#            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
#            zs.append(float(z))
            col.append(ast.literal_eval(c))
    ax1.clear()
    ax1.scatter(xs, ys, c = col)
#    ax1.scatter(xs, ys, zs, c = col)
    
    
ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()