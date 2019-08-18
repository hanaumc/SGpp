

# Rekursive Definition der B-Splines
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


# Verschiebung der B-Splines
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