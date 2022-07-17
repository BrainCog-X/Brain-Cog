import numpy as np

def data_transfer(B_txt, A_txt):
    """
    Aim:读取训练数据，并将其转换为可以处理的形式
    @param B_txt:Before processing -txt
    @param A_txt:After processing -txt
    @return:After processing -data
    """
    with open(B_txt, 'r') as f:  #'dataA_B.txt'
        data_all =[]
        data_1 = []
        data_2 = []
        data_3 = []
        data = f.read() #Read all the data in txt  ...str
        data_split = data.split('\n\n') #Divide the data with '\n\n'
        for i in range(len(data_split)-1):  #There are (len(data_split)-1) sets of valid data
            data_split[i] = data_split[i].split('\n')   #Remove '\n' from each set of data
            for j in range(len(data_split[i])):
                # Split number
                data_split[i][j] = " ".join(data_split[i][j])
                data_split[i][j] = data_split[i][j].split(' ')
                data_split[i][j] = list(map(int, data_split[i][j])) #str-int

            data_split[i] = np.array(data_split[i])    #list-np.array
            # Data expansion
            data_all.append(data_split[i])
            data_1.append(np.flipud(data_split[i])) #上下对称
            # data_2_split = data_split[i][:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
            # data_2.append(np.fliplr(data_2_split))    #左右对称
            # data_3_split = data_split[i][:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
            # data_3.append(np.fliplr(data_3_split))

        data_all.extend(data_1)
        # data_all.extend(data_2)
        # data_all.extend(data_3)

    data_all = np.array(data_all)
    data_all = data_all.reshape(data_all.shape[0]*data_all.shape[1], data_all.shape[2])
    data_all = data_all.astype(int)

    # new_data = np.repeat(data_all, repeats=num, axis=0)
    # new_data = np.repeat(new_data, repeats=num, axis=1)

    np.savetxt(A_txt, data_all, fmt='%i')  #'train.txt'

    # Read TXT data into numpy
    state = np.loadtxt(A_txt, dtype = np.int)
    print(data_all.shape)
    return state
