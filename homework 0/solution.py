import numpy as np


def make_array_from_list(some_list):
    print(type(some_list))
    some_list = np.array(some_list)
    print(type(some_list))
    return some_list


def make_array_from_number(num):
    print(num)
    print(type(num))
    num = np.array([num])
    print(num)
    print(type(num), len(num)) 
    return num


class NumpyBasics:
    def add_arrays(self, a, b):
        print(a)
        print(b)
        assert len(a)==len(b), "Error: sizes of arrays not same"
        sumab = np.add(a,b)
        print(sumab)
        return sumab

    def add_array_number(self, a, num):
        print(a)
        print(num)
        s = np.array(a)+num
        print(s)
        return s

    def multiply_elementwise_arrays(self, a, b):
        print(a)
        print(b)
        assert len(a)==len(b), "Error: sizes of arrays not same"
        c = np.multiply(a,b)
        print(c)
        return c

    def dot_product_arrays(self, a, b):
        print(a)
        print(b)
        assert len(a)==len(b), "Error: size of arrays not same"
        c = np.dot(a,b)
        print(c)
        return c

    def dot_1d_array_2d_array(self, a, m):
        # consider the 2d array to be like a matrix
        print(a)
        print(m)
        print(np.array(m).shape)
        assert len(a)==np.array(m).shape[0], "Error: dot product not possible as size of array is not same as the row dimension of matrix"
        c = np.dot(a,m)
        print(c)
        return c

if __name__ == "__main__":
    make_array_from_list([1,2,3])
    make_array_from_number(2)
    nb = NumpyBasics()
    nb.add_arrays([1,2,3],[10,20,30])
    nb.add_array_number([1,2,3],10)
    nb.multiply_elementwise_arrays([1,2,3],[2,4,8])
    nb.dot_product_arrays([1,2,3],[2,4,8])
    nb.dot_1d_array_2d_array([1,2,3],[[2,4,8],[3,9,27],[1,1,1]])
