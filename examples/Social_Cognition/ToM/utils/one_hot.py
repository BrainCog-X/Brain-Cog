import numpy as np

def one_hot(value):
    num = '12345'
    letter = [0 for _ in range(len(num))]
    letter[value-1] = 1
    letter = np.array([letter])
    # print(letter)
    return letter
