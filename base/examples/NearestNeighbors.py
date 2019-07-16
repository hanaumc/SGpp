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
from numpy import linalg as LA, transpose, full
from matplotlib.pyplot import axis


# Sucht nach dem naehesten (n+1)**2 Block
def NNsearch(I_all, sort, degree, j, q):
  xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
  yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
  block = np.meshgrid(xblock, yblock)
  eval_block = np.zeros(((degree+1)**2, 1))
  s=0
  for i in range(degree+1):
    for t in range(degree+1):
      eval_block[s] = circle(radius,[block[0][t,i], block[1][t,i]])
      s=s+1
  if np.all(eval_block>0) == True:
    s=0
    for i in range(degree+1):
      for t in range(degree+1):
        NN[j,s] = [block[0][t,i], block[1][t,i]]
        s=s+1
  else:
    xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
    yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
    block = np.meshgrid(xblock, yblock)
    s=0
    for i in range(degree+1):
      for t in range(degree+1):
        eval_block[s] = circle(radius,[block[0][t,i],block[1][t,i]])
        s=s+1
    if np.all(eval_block>0) == True:
      s=0
      for i in range(degree+1):
        for t in range(degree+1):
          NN[j,s] = [block[0][t,i], block[1][t,i]]
          s=s+1
    else:
      xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
      yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
      block = np.meshgrid(xblock, yblock)
      s=0
      for i in range(degree+1):
        for t in range(degree+1):
          eval_block[s] = circle(radius,[block[0][t,i],block[1][t,i]])
          s=s+1
      if np.all(eval_block>0) == True:
        s=0
        for i in range(degree+1):
          for t in range(degree+1):
            NN[j,s] = [block[0][t,i], block[1][t,i]]
            s=s+1
      else:
        xblock = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
        yblock = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
        block = np.meshgrid(xblock, yblock)
        s=0
        for i in range(degree+1):
          for t in range(degree+1):
            eval_block[s] = circle(radius,[block[0][t,i],block[1][t,i]])
            s=s+1
        if np.all(eval_block>0) == True:
          s=0
          for i in range(degree+1):
            for t in range(degree+1):
              NN[j,s] = [block[0][t,i], block[1][t,i]]
              s=s+1
        elif q == len(I_all)-1:
          print('Fehler: Gitter nicht fein genug. Erhoehe Level.')
          quit()
        else:
          NNsearch(sort, j, q+1)
  return NN