# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:46:47 2024

@author: f24lerou
"""

import matplotlib.pyplot as plt 

nrows = 2
ncols = 2

val1 = ["col1", "col2"]
val2 = ["row1", "row2"] 
val3 = [["lala" for c in range(ncols)] for r in range(nrows)] 
   
fig, ax = plt.subplots() 
ax.set_axis_off() 
table = ax.table( 
    cellText = val3,  
    rowLabels = val2,  
    colLabels = val1, 
    rowColours =["palegreen"]*nrows,  
    colColours =["palegreen"]*ncols, 
    cellLoc ='center',  
    loc ='upper left')         
   
ax.set_title('matplotlib.axes.Axes.table() function Example', 
             fontweight ="bold") 
   
plt.show() 