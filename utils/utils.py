def check_dims(dims, operation, verbose=True):
    if operation == 'init':
        if isinstance(dims, list):
            if len(dims) == 2:
                if isinstance(dims[0], int):
                    if isinstance(dims[1], int):
                        pass
                    else:
                        if verbose: print('ERROR: 1st member of the argument "dims" is not an integer.')
                        return False
                else:
                    if verbose: print('ERROR: 0th member of the argument "dims" is not an integer.')
                    return False
            else:
                if verbose: print('ERROR: The length of the list "dims" is not two.')
                return False
        else:
            if verbose: print('ERROR: Argument "dims" is not a list.')
            return False

        return True
    elif operation == 'add':
        if isinstance(dims[0], list) and isinstance(dims[1], list):
            if dims[0][0] == dims[1][0] and dims[0][1] == dims[1][1]:
                pass
            else:
                if verbose: print('ERROR: The dimensions do not match when adding matrices.')
                return False
        else:
            if verbose: print('ERROR: Given arguments are not lists.')
            return False
        return True
    else:
        return False

def check_build_from_arrays(arrays, verbose=False):
    if isinstance(arrays, list):
        if len(arrays) > 0:
            if isinstance(arrays[0], list):
                length = len(arrays[0])
            else:
                length = None
            for a_index, array in enumerate(arrays):
                if isinstance(array, list):
                    if len(array) > 0:
                        if len(array) == length:
                            for m_index, member in enumerate(array):
                                if isinstance(member, float) or isinstance(member, int):
                                    pass
                                else:
                                    if verbose: print('ERROR: member {} of the array {} \
                                            is not a float or an integer.'.format(m_index, a_index))
                                    return False
                        else:
                            if verbose: print('ERROR: array {} is not the same length as the 0th one.'
                                    .format(a_index))
                            return False
                    else:
                        if verbose: print('ERROR: array {} is empty.'.format(a_index))
                        return False
                else:
                    if verbose: print('ERROR: array {} is not a list.'.format(a_index))
                    return False
        else:
            if verbose: print('ERROR: argument "arrays" is an empty list.')
            return False
    else:
        if verbose: print('ERROR: argument "arrays" is not a list of lists.')
        return False
    
    return True

def check_add_matrix(a, b, verbose=True):
    from matrix import Matrix
    if type(a).__name__ == 'Matrix':
        if type(b).__name__ == 'Matrix':
            if check_dims([a.dimensions, b.dimensions], 'add'):
                pass
            else:
                if verbose: print('ERROR: Dimension check failed when adding matrices.')
                return False
        else:
            if verbose: print('ERROR: The second argument is not a matrix.')
            return False
    else:
        if verbose: print('ERROR: The first argument is not a matrix.')
        return False
    
    return True

def check_add_row_to_row(m, row, r_index, verbose=False):
    if isinstance(row, list):
        if len(row) == m.dimensions[1]:
            for m_index, member in enumerate(row):
                if isinstance(member, int) or isinstance(member, float):
                    pass
                else:
                    if verbose: print('ERROR: Element {} of the given list is not a float or an integer.'
                            .format(m_index))
                    return False
        else:
            if verbose: print("ERROR: The length of the given list does not match the matrix's dimensions.")
            return False
    else:
        if verbose: print('ERROR: The given argument is not a list.')
        return False

    if isinstance(r_index, int):
        if r_index < m.dimensions[0] and r_index >= 0:
            pass
        else:
            if verbose: print('ERROR: The given row index is out of bounds.')
            return False
    else:
        if verbose: print('ERROR: The given row index is not an integer.')
        return False

    return True

def check_multiply_row(self, r_index, scalar, verbose=False):
    if isinstance(r_index, int):
        if r_index < self.dimensions[0] and r_index >= 0:
            if isinstance(scalar, int) or isinstance(scalar, float):
                pass
            else:
                if verbose: print('ERROR: The given scalar is not a float or an integer.')
                return False
        else:
            if verbose: print('ERROR: The given row index is out of bounds.')
            return False
    else:
        if verbose: print('ERROR: The given row index is not an integer.')
        return False
    
    return True

def check_add_row(self, row, r_index, verbose=False):
    if isinstance(row, list):
        if len(row) == self.dimensions[1]:
            for m_index, member in enumerate(row):
                if isinstance(member, int) or isinstance(member, float):
                    pass
                else:
                    if verbose: print("ERROR: Element {} of the given list is not an integer or a float.".format(m_index))
                    return False
        else:
            if verbose: print("ERROR: The length of the given row does not match the matrix's dimensions.")
            return False
    else:
        if verbose: print('ERROR: The given "row" argument is not a list.')
        return False

    if isinstance(r_index, int):
        if r_index <= self.dimensions[0]  and r_index >= 0:
            pass
        else:
            if verbose: print('ERROR: The given row index is out of bounds.')
            return False
    else:
        if verbose: print('ERROR: The given row index is not an integer.')
        return False

    return True

def check_add_col(self, col, c_index, verbose=False):
    if isinstance(col, list):
        if len(col) == self.dimensions[0]:
            for m_index, member in enumerate(col):
                if isinstance(member, int) or isinstance(member, float):
                    pass
                else:
                    if verbose: print('ERROR: Element {} of the given list is not an integer or a float.'.format(m_index))
                    return False
        else:
            if verbose: print("ERROR: The length of the given column does not match the matrix's dimensions.")
            return False
    else:
        if verbose: print('ERROR: The given "col" argument is not a list.')
        return False

    if isinstance(c_index, int):
        if c_index <= self.dimensions[1] and c_index >= 0:
            pass
        else:
            if verbose: print('ERROR: The given column index is out of bounds.')
            return False
    else:
        if verbose: print('ERROR: The given column index is not an integer.')
        return False

    return True
        
def check_multiply_matrix(matrices, verbose=True):
    from matrix import Matrix
    if isinstance(matrices, list):
        if len(matrices) != 0:
            for matrix_index, matrix in enumerate(matrices):
                if type(matrix).__name__ == 'Matrix':
                    pass
                else:
                    if verbose: print('ERROR: Element {} is not a matrix.'.format(matrix_index))
                    return False
        else:
            if verbose: print('ERROR: The given list is empty.')
            return False
    else:
        if verbose: print('ERROR: The given argument is not a list of matrices.')
        return False
    
    dimensions = (matrices[len(matrices)-1].dimensions[0], matrices[len(matrices)-1].dimensions[1])
    for i in range(len(matrices)-1, 0, -1):
        if dimensions[1] == matrices[i-1].dimensions[0]:
            dimensions = (dimensions[0], matrices[i-1].dimensions[1])
        else:
            if verbose: print('ERROR: Dimensions of matrices are not consistent')
            return False

    return True

def check_multiply_matrix_two(a, b, verbose=True):
    from matrix import Matrix
    if type(a).__name__ == 'Matrix' and type(b).__name__ == 'Matrix':
        if a.dimensions[1] == b.dimensions[0]:
            pass
        else:
            if verbose: print('ERROR: Inconsistent matrix dimensions when multiplying matrices.')
            return False
    else:
        if verbose: print('ERROR: One of the given arguments is not a matrix.')
        return False
    
    return True

def check_dot(vectors):
    from matrix import Matrix
    for vector in vectors:
        if type(vector).__name__ == 'Matrix':
            if vector.dimensions[1] == 1:
                pass
            else:
                if verbose: print('ERROR: Can not get a dot product of any element other than a n*1 matrix.')
                return False
        else:
            if verbose: print('ERROR: The given list contains elements other than matrices.')
            return False

    return True
