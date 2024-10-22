import numpy as np
def conv(input_image, kernel_size, parameter):
    input_image = np.array(input_image)
    blocks = list()
    for i in range(len(input_image)):
        for j in range(len(input_image)):
            if i!=len(input_image)-1 and j!=len(input_image)-1:
                blocks.append((i,j,kernel_size))
    test_result = list()
    for x in blocks:
        block_list = list()
        for i in range(x[0],x[0]+x[2]):
            for j in range(x[1],x[1]+x[2]):
                if i<len(input_image) and j<len(input_image):
                    block_list.append(input_image[i][j])
        test_result.append(np.array(block_list).reshape((kernel_size,kernel_size))) # здесь хранятся матрицы каждого блока (то есть все возможные сдвиги)
    smth = [0 for _ in range(int(len(blocks)))]
    for i in range(len(smth)):
        for j in range(kernel_size):
            for r in range(kernel_size):
                smth[i]+= test_result[i][j][r]*parameter[j][r]
    return np.array(smth).reshape((kernel_size,kernel_size))

input_image = np.array([[10,20,30],[40,50,60],[70,80,90]])
parameter = np.array([[1,2],[3,4]])
print(conv(input_image, 2, parameter))