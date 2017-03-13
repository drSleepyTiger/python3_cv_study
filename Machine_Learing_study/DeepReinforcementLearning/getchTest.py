# import msvcrt
#
# while(1):
#     inkey = msvcrt.getch()
#     print ('this key is:', inkey)
#
#
import numpy as np

vector = np.array([0,0.5,0,0.5])

m = np.amax(vector)
print(m)
print(vector == m)

indices = np.nonzero(vector == m)[0]
print(indices)