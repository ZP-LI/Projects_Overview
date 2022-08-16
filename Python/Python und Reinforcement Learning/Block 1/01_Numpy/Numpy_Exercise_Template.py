import numpy as np


def create_array(array_length):
    # Input:
    #   - array_length:     int; specifies length of array
    # Return:
    #   - rand_array:       numpy array; dim: (array_length, )
    # Function:
    #   - create an array with random integers in range [0, 20]
    rand_array = np.random.randint(0, 20, size=(1, array_length))

    return rand_array


def append_array(array1, array2):
    # Input:
    #   - array1:   numpy array
    #   - array2:   numpy array, same dimensions as array1
    # Return:
    #   - array3:   numpy array; dim(2 * length(array1))
    # Function:
    #   - append array1 to array2 (mind the order!)
    array3 = np.concatenate((array1, array2), axis=1)

    return array3


def sort_array(array):
    # Input:
    #   - array:        numpy array, dim (20, )
    # Return:
    #   - array_sym:    numpy array, dim (20, )
    # Function:
    #   - sort the values of array in ascending order and store the result in array_asc
    #   - sort the array in descending order
    #   - add both arrays and return the result as array_sym

    # Create a copy of the array which is sorted in ascending order
    array_asc = np.sort(array)
    # Sort the array in descending order
    array = array[::-1].sort()
    # Add both arrays and return the result as array_sym
    array_sym = array_asc + array

    return array_sym


def scale(array_in):
    # Input:
    #   - array_in:     numpy array, dim (20, )
    # Return:
    #   - array_scaled: numpy array, dim (20, )
    # Function:
    #   - since we added the two arrays, values up to 40 are possible
    #   - make sure, that the highest value is exactly 20, by scaling all values by the same factor
    #   - afterwards round all values to 2 decimals
    max = np.max(array_in)
    factor = 20.0 / max
    array_scaled = np.round(array_in * factor, decimals=2)

    return array_scaled


def indirect_sort(array_in):
    # Input:
    #   - array_in:    numpy array, dim (20, )
    # Return:
    #   - matrix:   numpy_array, dim (2, 10)
    # Function:
    #   - reshape array_in to dimension (2, 10)
    #   - create an array idx with dim (1, 10).
    #   - This array has to contain of all integers from 1 to 10, but in arbitrary order
    #   - insert the newly created array as the second row of array_in
    #   - sort the columns of the matrix by the first row and return the matrix

    # Reshape array_in
    array_in = array_in.reshape(2, 10)
    # Create an array containing all integers from 1 to 10
    idx = np.arange(1, 11)
    # Shuffle idx
    np.random.shuffle(idx)
    # Insert idx as second row of array_in
    array_in[1, :] = idx
    # Sort columns of matrix by the first row
    matrix = array_in[:, array_in[0, :].argsort()]

    # To sort matrix by column
    # array_in[array_in[:, 0].argsort()]

    return matrix


def matrix_sum(matrix_in):
    # Input:
    #   - matrix_in:    2D numpy array with arbitrary dimensions
    # Return:
    #   - sum_all:      numpy array containing the sum of all elements of matrix_in
    #   - sum_row:      numpy array containing the sum of each row of matrix_in
    #   - sum_column:   numpy array containing the sum of each column of matrix_in
    # Function:
    #   - compute sum of all elements of matrix_in and the sums of its columns and rows
    sum_all = np.sum(matrix_in)
    sum_row = np.sum(matrix_in, axis=1)
    sum_column = np.sum(matrix_in, axis=0)

    return sum_all, sum_row, sum_column

if __name__ == "__main__":
    length = 10
    array_a = create_array(length)
    array_b = create_array(length)
    print('\nThe two random arrays are:\nArray a = ', array_a, '\nArray b = ', array_b)
    array_c = append_array(array_a, array_b)
    print('\nThe appended array: ', array_c)
    array_d = sort_array(array_c)
    print('\nSorted in ascending and descending order. Afterwards summed up:\nThe result: ', array_d)
    array_e = scale(array_d)
    print('\nAnd now scaled such that the maximum value is 20.00:\n', array_e)
    matrix = indirect_sort(array_e)
    print('\nThe created and sorted matrix:\n', matrix)
    sum_all, sum_row, sum_column = matrix_sum(matrix)
    print('\nSum of all elements =\t {:.2f}'.format(sum_all),
          '\nSum of each row =\t\t', sum_row,
          '\nSum of each column =\t', sum_column)
