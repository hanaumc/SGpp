import numpy as np
import pickle
import os
import math
import sys
import matplotlib 
import matplotlib.pyplot as plt
import weightfunction
import Bspline
import scipy 
from numpy import linalg as LA
import time




################## Testfunktion 1 auf Kreis ###################

# degree =  5
# radius = 0.4
# radius1 = 0.45
# radius2 = 0.1
#    
# subfunc = {}
#       
# levelymax = 8
# for levelx in range(4,8):
#     for levely  in range(4,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely  
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#                    
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#             
#         #print(data['eval_Interpolation'].shape)
#         #print(data.keys())      
# #         eval_Interpolation = data['eval_Interpolation']
# #         data['eval_Interpolation_1'] = eval_Interpolation
# # #         
# #         with open(filepath, 'wb') as fp:
# #             pickle.dump(data, fp)
#         subfunc[level_x,level_y,degree]  = data['eval_Interpolation_1']
#         eval_function_1 = data['eval_function_1']
#         #print(eval_function_1.shape)
#     levelymax=levelymax-1
# #print(subfunc)
#  
#      
# eval_function_1 = data['eval_function_1']
# print(eval_function_1.shape)
### n=1 ###

# f221 = subfunc[2,2,1]
# f231 = subfunc[2,3,1]
# f241 = subfunc[2,4,1]
# f251 = subfunc[2,5,1]
# f261 = subfunc[2,6,1]
# f271 = subfunc[2,7,1]
# f281 = subfunc[2,8,1]
# f291 = subfunc[2,9,1]
# f321 = subfunc[3,2,1]
# f331 = subfunc[3,3,1]
# f341 = subfunc[3,4,1]
# f351 = subfunc[3,5,1]
# f361 = subfunc[3,6,1]
# f371 = subfunc[3,7,1]
# f381 = subfunc[3,8,1]
# f421 = subfunc[4,2,1]
# f431 = subfunc[4,3,1]
# f441 = subfunc[4,4,1]
# f451 = subfunc[4,5,1]
# f461 = subfunc[4,6,1]
# f471 = subfunc[4,7,1]
# f521 = subfunc[5,2,1]
# f531 = subfunc[5,3,1]
# f541 = subfunc[5,4,1]
# f551 = subfunc[5,5,1]
# f561 = subfunc[5,6,1]
# f621 = subfunc[6,2,1]
# f631 = subfunc[6,3,1]
# f641 = subfunc[6,4,1]
# f651 = subfunc[6,5,1]
# f721 = subfunc[7,2,1]
# f731 = subfunc[7,3,1]
# f741 = subfunc[7,4,1]
# f821 = subfunc[8,2,1]
# f831 = subfunc[8,3,1]
# f921 = subfunc[9,2,1]

# 
# sg31 = f231+f321-f221
# sg41 = f241+f331+f421-(f231+f321)
# sg51 = f251+f341+f431+f521-(f241+f331+f421)
# sg61 = f261+f351+f441+f531+f621-(f251+f341+f431+f521)
# sg71 = f271+f361+f451+f541+f631+f721-(f261+f351+f441+f531+f621)
# sg81 = f281+f371+f461+f551+f641+f731+f821-(f271+f361+f451+f541+f631+f721)
# sg91 = f291+f381+f471+f561+f651+f741+f831+f921-(f281+f371+f461+f551+f641+f731+f821)



# print(np.linalg.norm(sg31-eval_function_1))
# print(np.linalg.norm(sg41-eval_function_1))
# print(np.linalg.norm(sg51-eval_function_1))
# print(np.linalg.norm(sg61-eval_function_1))
# print(np.linalg.norm(sg71-eval_function_1))
# print(np.linalg.norm(sg81-eval_function_1))
# print(np.linalg.norm(sg91-eval_function_1))

### n=3 ###

# f333 = subfunc[3,3,3]
# f343 = subfunc[3,4,3]
# f353 = subfunc[3,5,3]
# f363 = subfunc[3,6,3]
# f373 = subfunc[3,7,3]
# f383 = subfunc[3,8,3]
# f433 = subfunc[4,3,3]
# f443 = subfunc[4,4,3]
# f453 = subfunc[4,5,3]
# f463 = subfunc[4,6,3]
# f473 = subfunc[4,7,3]
# f533 = subfunc[5,3,3]
# f543 = subfunc[5,4,3]
# f553 = subfunc[5,5,3]
# f563 = subfunc[5,6,3]
# f633 = subfunc[6,3,3]
# f643 = subfunc[6,4,3]
# f653 = subfunc[6,5,3]
# f733 = subfunc[7,3,3]
# f743 = subfunc[7,4,3]
# f833 = subfunc[8,3,3]
#    
# sg43 = f343+f433-f333
# sg53 = f353+f443+f533-(f343+f433)
# sg63 = f363+f453+f543+f633-(f353+f443+f533)
# sg73 = f373+f463+f553+f643+f733-(f363+f453+f543+f633)
# sg83 = f383+f473+f563+f653+f743+f833-(f373+f463+f553+f643+f733)
#    
#    
# print(np.linalg.norm(sg43-eval_function_1))
# print(np.linalg.norm(sg53-eval_function_1))
# print(np.linalg.norm(sg63-eval_function_1))
# print(np.linalg.norm(sg73-eval_function_1))
# print(np.linalg.norm(sg83-eval_function_1))

### n=5 ###
 
# f445 = subfunc[4,4,5]
# f455 = subfunc[4,5,5]
# f465 = subfunc[4,6,5]
# f475 = subfunc[4,7,5]
# f545 = subfunc[5,4,5]
# f555 = subfunc[5,5,5]
# f565 = subfunc[5,6,5]
# f645 = subfunc[6,4,5]
# f655 = subfunc[6,5,5]
# f745 = subfunc[7,4,5]
#      
#   
# sg55 = f455+f545-(f445)
# sg65 = f465+f555+f645-(f455+f545)
# sg75 = f475+f565+f655+f745-(f465+f555+f645)
#   
#      
#      
# print(np.linalg.norm(sg55-eval_function_1))
# print(np.linalg.norm(sg65-eval_function_1))
# print(np.linalg.norm(sg75-eval_function_1))


##################### Testfunktion 2 auf Kreis #######################

# degree =  5
# radius = 0.4
# radius1 = 0.45
# radius2 = 0.1
#   
# subfunc = {}
#      
# levelymax = 8
# for levelx in range(4,8):
#     for levely  in range(4,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely  
#         filename = 'data_{}_{}_circle_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/circle'
#         filepath = os.path.join(path, filename)
#                   
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#         #print(data.keys())
#         #print(data['eval_Interpolation_2'].shape)      
#         subfunc[level_x,level_y,degree]  = data['eval_Interpolation_2']
#     levelymax=levelymax-1
# #print(subfunc)
#    
# eval_function_2 = data['eval_function_2']

# 
#### n = 1 #####
#
# f221 = subfunc[2,2,1]
# f231 = subfunc[2,3,1]
# f241 = subfunc[2,4,1]
# f251 = subfunc[2,5,1]
# f261 = subfunc[2,6,1]
# f271 = subfunc[2,7,1]
# f281 = subfunc[2,8,1]
# f291 = subfunc[2,9,1]
# f321 = subfunc[3,2,1]
# f331 = subfunc[3,3,1]
# f341 = subfunc[3,4,1]
# f351 = subfunc[3,5,1]
# f361 = subfunc[3,6,1]
# f371 = subfunc[3,7,1]
# f381 = subfunc[3,8,1]
# f421 = subfunc[4,2,1]
# f431 = subfunc[4,3,1]
# f441 = subfunc[4,4,1]
# f451 = subfunc[4,5,1]
# f461 = subfunc[4,6,1]
# f471 = subfunc[4,7,1]
# f521 = subfunc[5,2,1]
# f531 = subfunc[5,3,1]
# f541 = subfunc[5,4,1]
# f551 = subfunc[5,5,1]
# f561 = subfunc[5,6,1]
# f621 = subfunc[6,2,1]
# f631 = subfunc[6,3,1]
# f641 = subfunc[6,4,1]
# f651 = subfunc[6,5,1]
# f721 = subfunc[7,2,1]
# f731 = subfunc[7,3,1]
# f741 = subfunc[7,4,1]
# f821 = subfunc[8,2,1]
# f831 = subfunc[8,3,1]
# f921 = subfunc[9,2,1]
#  
#  
# sg31 = f231+f321-f221
# sg41 = f241+f331+f421-(f231+f321)
# sg51 = f251+f341+f431+f521-(f241+f331+f421)
# sg61 = f261+f351+f441+f531+f621-(f251+f341+f431+f521)
# sg71 = f271+f361+f451+f541+f631+f721-(f261+f351+f441+f531+f621)
# sg81 = f281+f371+f461+f551+f641+f731+f821-(f271+f361+f451+f541+f631+f721)
# sg91 = f291+f381+f471+f561+f651+f741+f831+f921-(f281+f371+f461+f551+f641+f731+f821)
# 
# 
# 
# print(np.linalg.norm(sg31-eval_function_2))
# print(np.linalg.norm(sg41-eval_function_2))
# print(np.linalg.norm(sg51-eval_function_2))
# print(np.linalg.norm(sg61-eval_function_2))
# print(np.linalg.norm(sg71-eval_function_2))
# print(np.linalg.norm(sg81-eval_function_2))
# print(np.linalg.norm(sg91-eval_function_2))
#
#
##### n = 3 #####
#
#
# f333 = subfunc[3,3,3]
# f343 = subfunc[3,4,3]
# f353 = subfunc[3,5,3]
# f363 = subfunc[3,6,3]
# f373 = subfunc[3,7,3]
# f383 = subfunc[3,8,3]
# f433 = subfunc[4,3,3]
# f443 = subfunc[4,4,3]
# f453 = subfunc[4,5,3]
# f463 = subfunc[4,6,3]
# f473 = subfunc[4,7,3]
# f533 = subfunc[5,3,3]
# f543 = subfunc[5,4,3]
# f553 = subfunc[5,5,3]
# f563 = subfunc[5,6,3]
# f633 = subfunc[6,3,3]
# f643 = subfunc[6,4,3]
# f653 = subfunc[6,5,3]
# f733 = subfunc[7,3,3]
# f743 = subfunc[7,4,3]
# f833 = subfunc[8,3,3]
#    
# sg43 = f343+f433-f333
# sg53 = f353+f443+f533-(f343+f433)
# sg63 = f363+f453+f543+f633-(f353+f443+f533)
# sg73 = f373+f463+f553+f643+f733-(f363+f453+f543+f633)
# sg83 = f383+f473+f563+f653+f743+f833-(f373+f463+f553+f643+f733)
#    
#    
# print(np.linalg.norm(sg43-eval_function_2))
# print(np.linalg.norm(sg53-eval_function_2))
# print(np.linalg.norm(sg63-eval_function_2))
# print(np.linalg.norm(sg73-eval_function_2))
# print(np.linalg.norm(sg83-eval_function_2))



### n=5 ###
#  
# f445 = subfunc[4,4,5]
# f455 = subfunc[4,5,5]
# f465 = subfunc[4,6,5]
# f475 = subfunc[4,7,5]
# f545 = subfunc[5,4,5]
# f555 = subfunc[5,5,5]
# f565 = subfunc[5,6,5]
# f645 = subfunc[6,4,5]
# f655 = subfunc[6,5,5]
# f745 = subfunc[7,4,5]
#      
#   
# sg55 = f455+f545-(f445)
# sg65 = f465+f555+f645-(f455+f545)
# sg75 = f475+f565+f655+f745-(f465+f555+f645)
#   
#      
#      
# print(np.linalg.norm(sg55-eval_function_2))
# print(np.linalg.norm(sg65-eval_function_2))
# print(np.linalg.norm(sg75-eval_function_2))




##################### Testfunktion 1 auf Ellipse #######################

degree =  3
radius = 0.4
radius1 = 0.45
radius2 = 0.1
   
subfunc = {}
      
levelymax = 9
for levelx in range(4,9):
    for levely  in range(4,levelymax):
        start = time.time()
        print(levelx,levely)
        level_x = levelx
        level_y = levely  
        filename = 'data_{}_{}_ellipse_degree_{}.pkl'.format(level_x, level_y, degree)
        path = '/home/hanausmc/pickle/ellipse'
        filepath = os.path.join(path, filename)
                   
        with open(filepath, 'rb') as fp:
            data = pickle.load(fp)
        print(data.keys())
        #print(data['eval_Interpolation_2'].shape)      
        subfunc[level_x,level_y,degree]  = data['eval_Interpolation_1']
    levelymax=levelymax-1
#print(subfunc)
    
eval_function_1 = data['eval_function_1']
#
#
##### n=1 ######
# 
# f221 = subfunc[2,2,1]
# f231 = subfunc[2,3,1]
# f241 = subfunc[2,4,1]
# f251 = subfunc[2,5,1]
# f261 = subfunc[2,6,1]
# f271 = subfunc[2,7,1]
# f281 = subfunc[2,8,1]
# f291 = subfunc[2,9,1]
# f321 = subfunc[3,2,1]
# f331 = subfunc[3,3,1]
# f341 = subfunc[3,4,1]
# f351 = subfunc[3,5,1]
# f361 = subfunc[3,6,1]
# f371 = subfunc[3,7,1]
# f381 = subfunc[3,8,1]
# f421 = subfunc[4,2,1]
# f431 = subfunc[4,3,1]
# f441 = subfunc[4,4,1]
# f451 = subfunc[4,5,1]
# f461 = subfunc[4,6,1]
# f471 = subfunc[4,7,1]
# f521 = subfunc[5,2,1]
# f531 = subfunc[5,3,1]
# f541 = subfunc[5,4,1]
# f551 = subfunc[5,5,1]
# f561 = subfunc[5,6,1]
# f621 = subfunc[6,2,1]
# f631 = subfunc[6,3,1]
# f641 = subfunc[6,4,1]
# f651 = subfunc[6,5,1]
# f721 = subfunc[7,2,1]
# f731 = subfunc[7,3,1]
# f741 = subfunc[7,4,1]
# f821 = subfunc[8,2,1]
# f831 = subfunc[8,3,1]
# f921 = subfunc[9,2,1]
#    
#    
# sg31 = f231+f321-f221
# sg41 = f241+f331+f421-(f231+f321)
# sg51 = f251+f341+f431+f521-(f241+f331+f421)
# sg61 = f261+f351+f441+f531+f621-(f251+f341+f431+f521)
# sg71 = f271+f361+f451+f541+f631+f721-(f261+f351+f441+f531+f621)
# sg81 = f281+f371+f461+f551+f641+f731+f821-(f271+f361+f451+f541+f631+f721)
# sg91 = f291+f381+f471+f561+f651+f741+f831+f921-(f281+f371+f461+f551+f641+f731+f821)
#   
#   
#   
# print(np.linalg.norm(sg31-eval_function_1))
# print(np.linalg.norm(sg41-eval_function_1))
# print(np.linalg.norm(sg51-eval_function_1))
# print(np.linalg.norm(sg61-eval_function_1))
# print(np.linalg.norm(sg71-eval_function_1))
# print(np.linalg.norm(sg81-eval_function_1))
# print(np.linalg.norm(sg91-eval_function_1))
 

##### n=3 ##### 
 
f443 = subfunc[4,4,3]
f453 = subfunc[4,5,3]
f463 = subfunc[4,6,3]
f473 = subfunc[4,7,3]
f483 = subfunc[4,8,3]
f543 = subfunc[5,4,3]
f553 = subfunc[5,5,3]
f563 = subfunc[5,6,3]
f573 = subfunc[5,7,3]
f643 = subfunc[6,4,3]
f653 = subfunc[6,5,3]
f663 = subfunc[6,6,3]
f743 = subfunc[7,4,3]
f753 = subfunc[7,5,3]
f843 = subfunc[8,4,3]
     
sg53 = f453+f543-f443
sg63 = f463+f553+f643-(f453+f543)
sg73 = f473+f563+f653+f743-(f463+f553+f643)
sg83 = f483+f573+f663+f753+f843-(f473+f563+f653+f743)
     
     
print(np.linalg.norm(sg53-eval_function_1))
print(np.linalg.norm(sg63-eval_function_1))
print(np.linalg.norm(sg73-eval_function_1))
print(np.linalg.norm(sg83-eval_function_1))




##################### Testfunktion 2 auf Ellipse #######################

# degree =  3
# radius = 0.4
# radius1 = 0.45
# radius2 = 0.1
#    
# subfunc = {}
#       
# levelymax = 9
# for levelx in range(4,9):
#     for levely  in range(4,levelymax):
#         start = time.time()
#         print(levelx,levely)
#         level_x = levelx
#         level_y = levely  
#         filename = 'data_{}_{}_ellipse_degree_{}.pkl'.format(level_x, level_y, degree)
#         path = '/home/hanausmc/pickle/ellipse'
#         filepath = os.path.join(path, filename)
#                    
#         with open(filepath, 'rb') as fp:
#             data = pickle.load(fp)
#         #print(data.keys())
#         #print(data['eval_Interpolation_2'].shape)      
#         subfunc[level_x,level_y,degree]  = data['eval_Interpolation_2']
#     levelymax=levelymax-1
# #print(subfunc)
#     
# eval_function_2 = data['eval_function_2']
#   
#
###### n=1 ######
#
#
# f221 = subfunc[2,2,1]
# f231 = subfunc[2,3,1]
# f241 = subfunc[2,4,1]
# f251 = subfunc[2,5,1]
# f261 = subfunc[2,6,1]
# f271 = subfunc[2,7,1]
# f281 = subfunc[2,8,1]
# f291 = subfunc[2,9,1]
# f321 = subfunc[3,2,1]
# f331 = subfunc[3,3,1]
# f341 = subfunc[3,4,1]
# f351 = subfunc[3,5,1]
# f361 = subfunc[3,6,1]
# f371 = subfunc[3,7,1]
# f381 = subfunc[3,8,1]
# f421 = subfunc[4,2,1]
# f431 = subfunc[4,3,1]
# f441 = subfunc[4,4,1]
# f451 = subfunc[4,5,1]
# f461 = subfunc[4,6,1]
# f471 = subfunc[4,7,1]
# f521 = subfunc[5,2,1]
# f531 = subfunc[5,3,1]
# f541 = subfunc[5,4,1]
# f551 = subfunc[5,5,1]
# f561 = subfunc[5,6,1]
# f621 = subfunc[6,2,1]
# f631 = subfunc[6,3,1]
# f641 = subfunc[6,4,1]
# f651 = subfunc[6,5,1]
# f721 = subfunc[7,2,1]
# f731 = subfunc[7,3,1]
# f741 = subfunc[7,4,1]
# f821 = subfunc[8,2,1]
# f831 = subfunc[8,3,1]
# f921 = subfunc[9,2,1]
#    
#    
# sg31 = f231+f321-f221
# sg41 = f241+f331+f421-(f231+f321)
# sg51 = f251+f341+f431+f521-(f241+f331+f421)
# sg61 = f261+f351+f441+f531+f621-(f251+f341+f431+f521)
# sg71 = f271+f361+f451+f541+f631+f721-(f261+f351+f441+f531+f621)
# sg81 = f281+f371+f461+f551+f641+f731+f821-(f271+f361+f451+f541+f631+f721)
# sg91 = f291+f381+f471+f561+f651+f741+f831+f921-(f281+f371+f461+f551+f641+f731+f821)
#   
#   
#   
# print(np.linalg.norm(sg31-eval_function_2))
# print(np.linalg.norm(sg41-eval_function_2))
# print(np.linalg.norm(sg51-eval_function_2))
# print(np.linalg.norm(sg61-eval_function_2))
# print(np.linalg.norm(sg71-eval_function_2))
# print(np.linalg.norm(sg81-eval_function_2))
# print(np.linalg.norm(sg91-eval_function_2))
 
 
##### n=3 ###### 
 

# f443 = subfunc[4,4,3]
# f453 = subfunc[4,5,3]
# f463 = subfunc[4,6,3]
# f473 = subfunc[4,7,3]
# f483 = subfunc[4,8,3]
# f543 = subfunc[5,4,3]
# f553 = subfunc[5,5,3]
# f563 = subfunc[5,6,3]
# f573 = subfunc[5,7,3]
# f643 = subfunc[6,4,3]
# f653 = subfunc[6,5,3]
# f663 = subfunc[6,6,3]
# f743 = subfunc[7,4,3]
# f753 = subfunc[7,5,3]
# f843 = subfunc[8,4,3]
#     
# sg53 = f453+f543-f443
# sg63 = f463+f553+f643-(f453+f543)
# sg73 = f473+f563+f653+f743-(f463+f553+f643)
# sg83 = f483+f573+f663+f753+f843-(f473+f563+f653+f743)
#     
#     
# print(np.linalg.norm(sg53-eval_function_2))
# print(np.linalg.norm(sg63-eval_function_2))
# print(np.linalg.norm(sg73-eval_function_2))
# print(np.linalg.norm(sg83-eval_function_2))











def function_2(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.circle(radius,x)
    return f


def function_1(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.circle(radius, x)
    return f

def function_1_ellipse(x):
    f = 0
    f = np.sin(8* x[0]) + np.sin(7 * x[1])
    f = f * weightfunction.ellipse(radius1, radius2, x)
    return f

def function_2_ellipse(x):
    f = 0
    f = np.sin(np.pi*x[0]*x[1])
    f = f * weightfunction.ellipse(radius1,radius2,x)
    return f

