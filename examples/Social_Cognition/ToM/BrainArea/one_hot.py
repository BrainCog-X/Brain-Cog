from numpy import argmax
import numpy as np

def one_hot(value):
    num = '01'
    letter = [0 for _ in range(len(num))]
    letter[value] = 1
    letter = np.array([letter])
    return letter



# print(one_hot(4))

