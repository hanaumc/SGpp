import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import Bspline
import scipy 
from scipy import special
from mpl_toolkits import mplot3d
from numpy import linalg as LA, transpose, full, vstack
from matplotlib.pyplot import axis


# # Fehler auf vollen Gittern gnxy: Grad n=1, Level x,y 
# g122 = 3.06456925
# g132 = 2.05072078
# g142 = 1.92660081
# g152 = 1.90561163
# g162 = 1.90137736
# g123 = 2.28126256
# g133 = 0.77069485
# g143 = 0.48419723
# g153 = 0.44837857
# g163 = 0.44323095
# g124 = 2.19715673
# g134 = 0.60899005
# g144 = 0.19741351
# g154 = 0.1216636
# g164 = 0.11276051
# g125 = 2.18068289
# g135 = 0.58864788
# g145 = 0.15462222
# g155 = 0.04912229
# g165 = 0.03048647
# g126 = 2.17715534
# g136 = 0.5851171
# g146 = 0.14973064
# g156 = 0.386204
# 
# 
# # Fehler auf vollen Gittern gnxy: Grad n=3, Level x,y 
# g333 = 0.38047882
# g343 = 0.21750155
# g353 = 0.21679535
# g363 = 0.2167305
# g334 = 0.31234951
# g344 = 0.00236114
# g354 = 0.00173869
# g364 = 0.00120445
# 
# 
# # Fehler auf vollen Gittern gnxy: Grad n=5, Level x,y 
# g544 = 0.00241462
# 
# 
# # Kombination zu Sparse Grid sgnlvl: grad=n, level=lvl
# sg13 = g123 + g132 - g122
# sg14 = g124 + g133 + g142 - g123 - g132
# sg15 = g125 + g134 + g143 + g152 - g124 - g133 - g142
# sg16 = g126 + g135 + g144 + g153 + g162 - g125 - g134 - g143 - g152
# 
# sg34 = g334 + g343 - g333
# #sg35 = g335 + g344 + g353 - g334 - g343 
# 
# 
# plt.scatter(3, sg13, c='crimson', s=40, lw=0)
# plt.scatter(4, sg34, c='mediumblue', s=40, lw=0)
# plt.legend(('n=1', 'n=3'),loc='upper right')
# plt.scatter(4, sg14, c='crimson', s=40, lw=0)
# plt.scatter(5, sg15, c='crimson', s=40, lw=0)
# plt.scatter(6, sg16, c='crimson', s=40, lw=0)
# plt.plot([3,4,5,6], [sg13,sg14,sg15,sg16], 'crimson')
# plt.xlabel("Level")
# plt.ylabel("L2-Error")
# plt.show()
j=10
for i in range(2,10):
    for k in range(2,j):
       print(k,i)
    j=j-1

