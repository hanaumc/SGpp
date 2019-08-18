import numpy as np
import weightfunction
import Bspline


# Sucht nach dem naechsten (n+1)x(n+1) Array
def NNsearch(degree, sort, j, q, I_all, h_x, h_y, NN, radius):
    xarray = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
    yarray = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
    array = np.meshgrid(xarray, yarray)
    eval_array = np.zeros(((degree+1)**2, 1))
    s=0
    for i in range(degree+1):
        for t in range(degree+1):
            eval_array[s] = weightfunction.circle(radius,[array[0][t,i], array[1][t,i]])
            s=s+1
    if np.all(eval_array>0) == True:
        s=0
        for i in range(degree+1):
            for t in range(degree+1):
                NN[j,s] = [array[0][t,i], array[1][t,i]]
                s=s+1
    else:
        xarray = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]+(degree+1)*h_x, h_x)
        yarray = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
        array = np.meshgrid(xarray, yarray)
        s=0
        for i in range(degree+1):
            for t in range(degree+1):
                eval_array[s] = weightfunction.circle(radius,[array[0][t,i],array[1][t,i]])
                s=s+1
        if np.all(eval_array>0) == True:
            s=0
            for i in range(degree+1):
                for t in range(degree+1):
                    NN[j,s] = [array[0][t,i], array[1][t,i]]
                    s=s+1
        else:
            xarray = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
            yarray = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]-(degree+1)*h_y, -h_y)
            array = np.meshgrid(xarray, yarray)
            s=0
            for i in range(degree+1):
                for t in range(degree+1):
                    eval_array[s] = weightfunction.circle(radius,[array[0][t,i],array[1][t,i]])
                    s=s+1
            if np.all(eval_array>0) == True:
                s=0
                for i in range(degree+1):
                    for t in range(degree+1):
                        NN[j,s] = [array[0][t,i], array[1][t,i]]
                        s=s+1
            else:
                xarray = np.arange(I_all[int(sort[q,1]), 0], I_all[int(sort[q,1]), 0]-(degree+1)*h_x, -h_x)
                yarray = np.arange(I_all[int(sort[q,1]), 1], I_all[int(sort[q,1]), 1]+(degree+1)*h_y, h_y)
                array = np.meshgrid(xarray, yarray)
                s=0
                for i in range(degree+1):
                    for t in range(degree+1):
                        eval_array[s] = weightfunction.circle(radius,[array[0][t,i],array[1][t,i]])
                        s=s+1
                if np.all(eval_array>0) == True:
                    s=0
                    for i in range(degree+1):
                        for t in range(degree+1):
                            NN[j,s] = [array[0][t,i], array[1][t,i]]
                            s=s+1
                elif q == len(I_all)-1:
                    print('Fehler: kein (n+1)x(n+1) array im Gebiet gefunden. Erhoehe Level.')
                    quit()
                else:
                    NNsearch(degree, sort, j, q+1, I_all, h_x, h_y, NN, radius)
    return NN