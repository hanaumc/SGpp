import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


# def evalBspline(n, k, xi, x):
#     if n == 0:
#         return 1 if xi[int(k)] <= x < xi[k+1] else 0
#     if xi[int(k+(n-1)/2)+n] == xi[int(k+(n-1)/2)]:
#         c1 = 0
#     else:
#         c1 = (x-xi[int(k+(n-1)/2)])/(xi[int(k+(n-1)/2)+n]-xi[int(k+(n-1)/2)]) * evalBspline(n-1, int(k+(n-1)/2), xi, x)
#     if xi[int(k+(n-1)/2)+n+1] == xi[int(k+(n-1)/2)+1]:
#         c2 = 0
#     else:
#         c2 = (1-(x-xi[int(k+(n-1)/2)+1])/(xi[int(k+(n-1)/2)+n+1]-xi[int(k+(n-1)/2)+1])) * evalBspline(n-1, int(k+(n-1)/2)+1, xi, x)
#     return c1 + c2


#ohne verschiebung: funktioniert
def B(n, k, xi, x):
    if n == 0:
       return 1.0 if xi[int(k)] <= x < xi[int(k+1)] else 0.0
    if xi[int(k+n)] == xi[int(k)]:
       c1 = 0.0
    else:
       c1 = (x - xi[int(k)])/(xi[int(k+n)] - xi[int(k)]) * B(n-1, k, xi, x)
    if xi[int(k+n+1)] == xi[int(k+1)]:
       c2 = 0.0
    else:
       c2 = (1-(x-xi[int(k+1)])/(xi[int(k+n+1)] - xi[int(k+1)])) * B(n-1, k+1, xi, x)
    return c1 + c2



def evalBspline(n, k, xi, x):
    if n == 0:
       return 1.0 if xi[int(k+(n-1)/2)] <= x-(n-1)/2 < xi[int(k+(n-1)/2+1)] else 0.0
    if xi[int(k+(n-1)/2+n)] == xi[int(k+(n-1)/2)]:
       c1 = 0.0
    else:
       c1 = (x - xi[int(k+(n-1)/2)])/(xi[int(k+(n-1)/2+n)] - xi[int(k+(n-1)/2)]) * B(n-1, k+(n-1)/2, xi, x)
    if xi[int(k+(n-1)/2+n+1)] == xi[int(k+(n-1)/2+1)]:
       c2 = 0.0
    else:
       c2 = (1-(x -xi[int(k+(n-1)/2+1)])/(xi[int(k+(n-1)/2+n+1)] - xi[int(k+(n-1)/2+1)])) * B(n-1, k+(n-1)/2+1, xi, x)
    return c1 + c2
# 
# degree = 5
# level_x = 3
#   
# # Gitterweite
# h_x = 2**(-level_x)
#   
# #xi = np.arange(-(degree+1)/2, 1/h_x+(degree+1)/2+1, 1)*h_x*8
#   
# xi = np.zeros(2**level_x+degree+1+1)
# for k in range(2**level_x+degree+1+1):
#     if k in range(degree+1):
#         xi[k] = (k-degree)*h_x
#     elif k in range(degree+1, 2**level_x+1):
#         xi[k] = ((k+(degree-1)/2)-degree)*h_x
#     elif k in range(2**level_x+1, 2**level_x+degree+1+1):
#         xi[k] = ((k+degree-1)-degree)*h_x
#   
#   
# print(xi) 
#   
#   
#   
# x = np.linspace(-.5, 1.5, 200)
# eval = np.zeros(len(x))
# for i in range(0,9):
#     for j in range(len(x)):
#         eval[j] = B(degree, i, xi, x[j])
#     plt.plot(x,eval)
#   
# plt.show()
#   
# x = np.linspace(-.5, 1.5, 200)
# eval = np.zeros(len(x))
# for i in range(-2,7):
#     for j in range(len(x)):
#         eval[j] = evalBspline(degree, i, xi, x[j])
#     plt.plot(x,eval)
#   
# plt.show()
# 
# 
# 
# 
# 
# 
# 


