import numpy as np
import pickle
import os

# for lx in range(3,8):
#    for ly in range(3,8):

# exemplary grid : (2,3)
degree = 3

level = [2, 3]
# levelx = 2
# levely = 3

# inner
I = [1, 2, 3]

# outer
J = [0, 0, 9]

# interpolation matrix
A = np.zeros((3, 3))
A[0, 2] = 17

# extension coefficients
E = [1, 2, 3, 4]

data = {'level': level,
        'I':I,
        'J':J,
        'A':A,
        }
data['E'] = E

filename = 'data_{}_{}_circle_degree{}.pkl'.format(level[0], level[1], degree)
path = '/home/hanausmc/pickle/circle'
filepath = os.path.join(path, filename)

with open(filepath, 'wb') as fp:
    pickle.dump(data, fp)
    print('saved data to {}'.format(filepath))

with open(filepath, 'rb') as fp:
    data_loaded = pickle.load(fp)
    
print(data_loaded.keys())
A_loaded = data_loaded['A']
print(A_loaded)
