from matrix import Matrix

def test_init_dims():
    passed = True
    failed_tests = []
    options = [0, [1], [1, 'b'], 'test', [1, 1.7], [10, 10]]
    for o_index, option in enumerate(options):
        test_matrix = Matrix(option)
        if o_index != len(options) - 1:
            if test_matrix.valid:
                passed = False
                failed_tests.append(o_index)
        else:
            if not test_matrix.valid:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Initial dimensions failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Initial dimensions tests passed.')

    return passed

def test_build_from_arrays_initial():
    passed = True
    failed_tests = []
    options = [1, [1], [[],[]], [[1],['c']], [['a'],[1]], [[1,2],[1]], [[1],[1,2]], [[1,2,3],[1,2,3]]]

    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows(option)
        if o_index != len(options) - 1:
            if test_matrix.valid:
                passed = False
                failed_tests.append(o_index)
        else:
            if not test_matrix.valid:
                passed = False
                failed_tests.append(o_index)

    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_cols(option)
        if o_index != len(options) - 1:
            if test_matrix.valid:
                passed = False
                failed_tests.append(o_index)
        else:
            if not test_matrix.valid:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Build from arrays initial failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Build from arrays initial test passed.')

def test_build_from_rows():
    passed = True
    failed_tests = []
    options = [
            [[1, 2, 3], [1, 2, 3]],
            [[2, 3], [2, 3]]
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows(option)
        for i in range(test_matrix.dimensions[0]):
            for j in range(test_matrix.dimensions[1]):
                if test_matrix.values[i][j] == option[i][j]:
                    pass
                else:
                    passed = False
                    failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Build from rows failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Build from rows test passed.')

def test_build_from_cols():
    passed = True
    failed_tests = []
    options = [
            [[1, 2, 3], [1, 2, 3]],
            [[2, 3], [2, 3]]
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_cols(option)
        for i in range(test_matrix.dimensions[0]):
            for j in range(test_matrix.dimensions[1]):
                if test_matrix.values[i][j] == option[j][i]:
                    pass
                else:
                    passed = False
                    failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Build from cols failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Build from cols test passed.')

def test_add_matrix_initial():
    passed = True
    failed_tests = []
    options = [
            [[1, 2], [1, 2]],
            [[1, 2, 3], [1, 2, 3]],
            ]
    for o_index, option in enumerate(options):
        test_matrix_a = Matrix.build_from_rows([[1, 2, 3], [1, 2, 3]])
        test_matrix_b = Matrix.build_from_rows(option)
        result_matrix = Matrix.add([test_matrix_a, test_matrix_a, test_matrix_a, test_matrix_b])
        if o_index != len(options) - 1: 
            if result_matrix.valid:
                passed = False
                failed_tests.append(o_index)
        else:
            if not result_matrix.valid:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Add matrices initial failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Add matrices initial test passed.')


def test_add_matrix():
    passed = True
    failed_tests = []
    options = [
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            ]
    for o_index, option in enumerate(options):
        test_matrix_a = Matrix.build_from_rows([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        test_matrix_b = Matrix.build_from_rows(option)
        matrices_to_add = [test_matrix_a, test_matrix_b]
        correct_values = []
        for i in range(test_matrix_a.dimensions[0]):
            correct_values.append([])
            for j in range(test_matrix_a.dimensions[1]):
                correct_values[i].append(0)
                for k in range(len(matrices_to_add)):
                    correct_values[i][j] += matrices_to_add[k].values[i][j]

        result_matrix = Matrix.add(matrices_to_add)
        for i in range(result_matrix.dimensions[0]):
            for j in range(result_matrix.dimensions[1]):
                if result_matrix.values[i][j] != correct_values[i][j]:
                    passed = False
                    failed_tests.append(o_index)
        
    if not passed:
        print('TEST FAILED: Add matrices failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Add matrices test passed.')

def test_add_row_to_row():
    passed = True
    failed_tests = []
    options = [
            (0, [1,2,3]),
            (3, [1,2]),
            (0, [1,2])
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows([[1,2],[3,4],[5,6]])
        test_matrix.add_row_to_row(options[o_index][0], options[o_index][1])
        if o_index != len(options) - 1:
            if not test_matrix.errors:
                passed = False
                failed_tests.append(o_index)
        else:
            if test_matrix.errors:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Add row to row failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Add row to row test passed.')


def test_compare_matrices():
    passed = True
    failed_tests = []
    options = [
            [[]],
            [[1,2]],
            [[1,2,3],[1,2,3]],
            [[3,4],[1,2]],
            [[1,2],[1,2]]
            ]
    for o_index, option in enumerate(options):
        test_matrix_a = Matrix.build_from_rows([[1,2],[1,2]])
        test_matrix_b = Matrix.build_from_rows(option)
        if o_index != len(options) - 1:
            if Matrix.compare_matrices([test_matrix_a, test_matrix_b]):
                passed = False
                failed_tests.append(o_index)
        else:
            if not Matrix.compare_matrices([test_matrix_a, test_matrix_b]):
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Compare matrices failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Compare matrices test passed.')

def test_multiply_row():
    passed = True
    failed_tests = []
    options = [
            (3, 2),
            ('a', 2),
            (1, 'a'),
            (0, 10)
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows([[1,2,3],[2,3,4],[4,5,6]])
        test_matrix.multiply_row(option[0], option[1])
        if o_index != len(options) - 1:
            if not test_matrix.errors:
                passed = False
                failed_tests.append(o_index)
        else:
            if test_matrix.errors:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Multiply row failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Multiply row test passed.')

def test_add_row_initial():
    passed = True
    failed_tests = []
    options = [
            ([1,2], 0),
            ('a', 1),
            (['a',2,3], 0),
            ([1,2,3], 4),
            ([1,2,3], 0)
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows([[1,2,3],[1,2,3],[1,2,3]])
        test_matrix.add_row(option[1], option[0])
        if o_index != len(options) - 1:
            if not test_matrix.errors:
                passed = False
                failed_tests.append(o_index)
        else:
            if test_matrix.errors:
                passed = False
                failed_tests.append(o_index)
    
    if not passed:
        print('TEST FAILED: Add row intial failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Add row initial test passed.')

def test_add_col_initial():
    passed = True
    failed_tests = []
    options = [
            ([4,4,4], -1),
            ([4,4], 0),
            ('a', 0),
            (['a',4,4], 0),
            ([4,4,4], 0)
            ]
    for o_index, option in enumerate(options):
        test_matrix = Matrix.build_from_rows([[1,2,3], [1,2,3], [1,2,3]])
        test_matrix.add_row(option[1], option[0])
        if o_index != len(options) - 1:
            if not test_matrix.errors:
                passed = False
                failed_tests.append(o_index)
        else:
            if test_matrix.errors:
                passed = False
                failed_tests.append(o_index)

    if not passed:
        print('TEST FAILED: Add col initial failed tests: {}.'.format(failed_tests))
    else:
        print('TEST PASSED: Add col initial test passed.')

   




























