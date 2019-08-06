import pysgpp
import math
import sys
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


# x wird Spaltenvektor mit eintraegen von Punkt 1 in Zeile 1, Punkt 2 in Zeile 2, usw...

def circle(radius, x):
    w = radius**2
    for i in range(len(x)):
        w = w - (x[i]-0.5)**2
    return w




def ellipse(radius1, radius2, x):
    w = radius1**2
    w = -((w-((x[0]-0.5)**2+2*(x[1]-0.5)**2))*(radius2**2-((x[0]-0.4)**2+(x[1]-0.6)**2)))
    return w
