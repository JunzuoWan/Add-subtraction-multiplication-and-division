# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:37:06 2018

@author: J.Wan
"""

import matplotlib.pyplot as plt

x1 = [2, 3, 4, 5, 6]
y1 = [5, 5, 5, 5, 5]

x2 = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [2, 3, 2, 3, 7, 3, 5, 9]

x3 = [11, 12, 13, 14, 15]
y3 = [6, 8, 7, 18, 27]

x4 = [3, 6, 9]
y4 = [6, 8, 7]

plt.scatter(x1, y1)
plt.scatter(x2, y2, marker='v', color='r')
plt.scatter(x3, y3, marker='^', color='m')
plt.scatter(x4, y4, marker='*', color='y')
plt.xlabel("x-value")
plt.ylabel("y-value")
plt.title('Plot with 4 Groups of Scatter Date')
plt.show()