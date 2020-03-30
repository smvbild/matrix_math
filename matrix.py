from utils import utils, tests
from operator import itemgetter
import math, random


class Matrix(object):
    def __init__(self, dims):
        self.valid = utils.check_dims(dims, "init")
        self.errors = False
        self.dimensions = dims if self.valid else None

        if self.valid:
            self.values = self.zeros()

    def __eq__(self, other):
        threshold = 0.01
        if isinstance(other, Matrix):
            if (
                self.dimensions[0] == other.dimensions[0]
                and self.dimensions[1] == other.dimensions[1]
            ):
                for i in range(self.dimensions[0]):
                    for j in range(self.dimensions[1]):
                        if (self.values[i][j] >= other.values[i][j] - threshold) and (
                            self.values[i][j] <= other.values[i][j] + threshold
                        ):
                            pass
                        else:
                            return False
            else:
                return False
        else:
            return False

        return True

    def __pow__(self, other):
        if self.is_vector() and other.is_vector():
            return (self.transpose() * other).values[0][0]

    def __add__(self, other):
        if isinstance(other, Matrix):
            if (
                self.dimensions[0] == other.dimensions[0]
                and self.dimensions[1] == self.dimensions[1]
            ):
                matrix = Matrix.copy(self)
                for i in range(self.dimensions[0]):
                    for j in range(self.dimensions[1]):
                        matrix.values[i][j] += other.values[i][j]

                return matrix

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if (
                self.dimensions[0] == other.dimensions[0]
                and self.dimensions[1] == self.dimensions[1]
            ):
                matrix = Matrix.copy(self)
                for i in range(self.dimensions[0]):
                    for j in range(self.dimensions[1]):
                        matrix.values[i][j] -= other.values[i][j]

                return matrix

    def __mul__(a, b):
        if isinstance(b, Matrix):
            valid = utils.check_multiply_matrix_two(a, b)
            if valid:
                matrix = Matrix([a.dimensions[0], b.dimensions[1]])
                matrix.zeros()
                for i in range(matrix.dimensions[0]):
                    for j in range(matrix.dimensions[1]):
                        temp_sum = 0
                        for k in range(a.dimensions[1]):
                            temp_sum += a.values[i][k] * b.values[k][j]

                        matrix.values[i][j] = temp_sum

                return matrix

        if isinstance(b, float) or isinstance(b, int):
            matrix = Matrix.copy(a)
            for i in range(a.dimensions[0]):
                for j in range(a.dimensions[1]):
                    matrix.values[i][j] *= b
            return matrix

        return None

    def __truediv__(a, b):
        if isinstance(b, float) or isinstance(b, int):
            matrix = Matrix.copy(a)
            for i in range(a.dimensions[0]):
                for j in range(a.dimensions[1]):
                    matrix.values[i][j] /= b
            return matrix

    def __repr__(matrix):
        widths = []
        for i in range(matrix.dimensions[0]):
            widths.append([])
            for j in range(matrix.dimensions[1]):
                widths[i].append(len(str(matrix.values[i][j])))

        max_widths = []
        for i in range(matrix.dimensions[1]):
            max_widths.append(max([width[i] for width in widths]))

        print("\n")
        print(
            "*"
            * (sum(max_widths) + 4 * matrix.dimensions[1] - (matrix.dimensions[1] - 1))
        )
        for i in range(matrix.dimensions[0]):
            for j in range(matrix.dimensions[1]):
                padding = 0
                padding = max_widths[j] - len(str(matrix.values[i][j])) + 1
                if j == 0:
                    if matrix.dimensions[1] == 1:
                        print("* {}".format(matrix.values[i][j]) + " " * padding + "*")
                    else:
                        print(
                            "* {}".format(matrix.values[i][j]) + " " * padding + "*",
                            end="",
                        )
                elif j == (matrix.dimensions[1] - 1):
                    print(" {}".format(matrix.values[i][j]) + " " * padding + "*")
                else:
                    print(
                        " {}".format(matrix.values[i][j]) + " " * padding + "*", end=""
                    )
            print(
                "*"
                * (
                    sum(max_widths)
                    + 4 * matrix.dimensions[1]
                    - (matrix.dimensions[1] - 1)
                )
            )

        return ""

    def __invert__(self):
        matrix = Matrix.copy(self)
        threshold = 0.0001
        if matrix.dimensions[0] == matrix.dimensions[1]:
            if (matrix.determinant() >= threshold):
                for i in range(matrix.dimensions[0]):
                    col = []
                    for j in range(matrix.dimensions[0]):
                        if i == j:
                            col.append(1)
                        else:
                            col.append(0)
                    matrix.add_col(matrix.dimensions[1], col)

                matrix.gaussian_elimination(matrix.dimensions[0])

                for i in range(matrix.dimensions[0]):
                    if matrix.values[i][i] == 0:
                        return None

                for i in range(matrix.dimensions[0]):
                    matrix.delete_col(0)

                return matrix
            else:
                print('The matrix is singular.')
                return None
        else:
            print("ERROR: Cannot invert a non-square matrix.")
            return None

    def __or__(v, u):
        if v.is_vector() and u.is_vector():
            if v.dimensions[0] == u.dimensions[0]:
                if u.is_zero():
                    return u
                elif v.is_zero():
                    return v
                else:
                    return u * ((v ** u) / (u ** u))
        
    
    @staticmethod
    def is_matrix(matrix):
        return isinstance(matrix, Matrix)

    def is_zero(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if self.values[i][j] == 0:
                    pass
                else:
                    return False
        return True

    def is_invertible(self):
        if ~self is None:
            return False
        else:
            return True

    def is_symmetric(self):
        return self == self.transpose()

    def is_vector(self):
        return self.dimensions[1] == 1 and self.dimensions[0] > 0

    def is_square(self):
        return self.dimensions[0] == self.dimensions[1]

    def is_collinear(self, vector):
        if self.is_vector() and vector.is_vector():
            a = Matrix.copy(self)
            b = Matrix.copy(vector)

            threshold = 0.01

            for i in range(self.dimensions[0]):
                if a.get_col(0)[i] != 0 and b.get_col(0)[i] != 0:
                    a = a * (1 / a.get_col(0)[i])
                    b = b * (1 / b.get_col(0)[i])
                    for j in range(self.dimensions[0]):
                        if (
                            a.values[j][0] >= b.values[j][0] - threshold
                            and a.values[j][0] <= b.values[j][0] + threshold
                        ):
                            pass
                        else:
                            return False
                    return True
            return False

    def is_orthogonal(self, vector):
        if self.is_vector() and vector.is_vector():
            a = Matrix.copy(self)
            b = Matrix.copy(vector)
            threshold = 0.01
            if a ** b >= -threshold and a ** b <= threshold:
                return True
            else:
                return False

    def is_upper_triangular(self):
        if not (self.is_square()):
            return False
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if i > j and self.values[i][j] != 0:
                    return False
        return True

    def is_lower_triangular(self):
        if not (self.is_square()):
            return False
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if j > i and self.values[i][j] != 0:
                    return False
        return True

    def is_triangular(self):
        return self.is_lower_triangular() or self.is_upper_triangular()

    @staticmethod
    def generate_dense_square_matrix(dim):
        m = Matrix([dim, dim])
        for i in range(dim):
            for j in range(dim):
                m.values[i][j] = random.random() * 200 - 100

        return m

    @staticmethod
    def generate_sparse_square_matrix(dim):
        m = Matrix([dim, dim])
        num_vals = dim ** 2
        num_not_zero = random.randint(int(num_vals / 4), int(num_vals / 2))
        i = list(range(dim))
        j = list(range(dim))

        values = []
        for k in range(num_not_zero):
            index_1, index_2 = (
                i[random.randint(0, len(i) - 1)],
                j[random.randint(0, len(j) - 1)],
            )
            values.append((index_1, index_2, random.randint(-100, 100)))

        for value in values:
            m.values[value[0]][value[1]] = value[2]

        return m

    def matrix_norm_1(self):
        a = Matrix.copy(self)
        sums = []
        for j in range(a.dimensions[1]):
            sums.append(sum(list(map((lambda x: abs(x)), (a.get_col(j))))))

        return max(sums)

    def matrix_norm_inf(self):
        a = Matrix.copy(self)
        sums = []
        for i in range(a.dimensions[0]):
            sums.append(sum(list(map((lambda x: abs(x)), (a.get_row(i))))))

        return max(sums)

    def frobenius_norm(self):
        return math.sqrt((self.transpose() * self).trace())


    def sort_rows(self):
        matrix = Matrix.copy(self)
        for i in range(matrix.dimensions[0] - 1):
            if matrix.values[i][i] == 0:
                for j in range(i + 1, matrix.dimensions[0]):
                    if matrix.values[j][i] != 0:
                        matrix = matrix.swap_rows(i, j)
        return matrix

    def cleanup(self):
        num_digits = 4
        threshold = 0.1 ** num_digits
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if self.values[i][j] >= -threshold and self.values[i][j] <= threshold:
                    self.values[i][j] = 0
                else:
                    self.values[i][j] = round(self.values[i][j], num_digits)
        return self

    def set_dimensions(self, dims):
        self.valid = utils.check_dims(dims, "init")
        self.dimensions = dims if self.valid else None
        self.values = self.zeros()

        return self

    def zeros(self):
        values = []
        for i in range(self.dimensions[0]):
            values.append([])
            for j in range(self.dimensions[1]):
                values[i].append(0)
        return values

    def identity(self):
        self.zeros()
        for i in range(min([self.dimensions[0], self.dimensions[1]])):
            self.values[i][i] = 1.0
        return self

    def euclidean_norm(self):
        if self.is_vector():
            return math.sqrt(self ** self)

    def rank(self):
        a = self.gaussian_elimination_ref(0)
        zero_rows = 0
        for i in range(a.dimensions[0]):
            zero = True
            for j in range(a.dimensions[1]):
                if a.values[i][j] != 0:
                    zero = False
            if zero:
                zero_rows += 1
        
        return a.dimensions[1] - zero_rows

    def swap_rows(self, i, j):
        if i < self.dimensions[0] and j < self.dimensions[0] and i >= 0 and j >= 0:
            matrix = Matrix.copy(self)
            temp_i = self.get_row(j)
            temp_j = self.get_row(i)
            matrix.values[i] = temp_i
            matrix.values[j] = temp_j
            return matrix

    def minor(self, i, j):
        minor = Matrix.copy(self)
        minor.delete_row(i - 1)
        minor.delete_col(j - 1)
        return minor

    def add_row_to_row(self, r_index, row):
        self.errors = False if utils.check_add_row_to_row(self, row, r_index) else True
        if not self.errors:
            for i in range(len(row)):
                self.values[r_index][i] += row[i]

    def multiply_row(self, r_index, scalar):
        self.errors = False if utils.check_multiply_row(self, r_index, scalar) else True
        if not self.errors:
            for i in range(self.dimensions[1]):
                self.values[r_index][i] *= scalar

    def get_col(self, c_index):
        if isinstance(c_index, int):
            if c_index < self.dimensions[1]:
                col = []
                for i in range(self.dimensions[0]):
                    col.append(self.values[i][c_index])
                return col
            else:
                print("ERROR: The given column index is out of bounds.")
                return None
        else:
            print("ERROR: The given column index is not an integer.")
            return None

    def get_row(self, r_index):
        if isinstance(r_index, int):
            if r_index < self.dimensions[0]:
                return self.values[r_index]
            else:
                print("ERROR: The given row index is out of bounds.")
                return None
        else:
            print("ERROR: The given row index is not an integer.")
            return None

    def is_positive_definite(self):
        if self.cholesky() is not None:
            return True
        else:
            return False

    def delete_row(self, r_index):
        if isinstance(r_index, int):
            if r_index < self.dimensions[0]:
                self.values.pop(r_index)
                self.dimensions[0] -= 1
            else:
                print("ERROR: The given row index is out of bounds.")
        else:
            print("ERROR: The given row index is not an integer.")

    def delete_col(self, c_index):
        if isinstance(c_index, int):
            if c_index < self.dimensions[1]:
                for i in range(self.dimensions[0]):
                    self.values[i].pop(c_index)
                self.dimensions[1] -= 1
        else:
            print("ERROR: The given column index is not an integer.")

    def add_row(self, r_index, row):
        self.errors = False
        if not utils.check_add_row(self, row, r_index):
            self.errors = True

        if not self.errors:
            self.values.insert(r_index, row)
            self.dimensions[0] += 1

    def add_col(self, c_index, col):
        self.errors = False
        if not utils.check_add_col(self, col, c_index):
            self.errors = True

        if not self.errors:
            for i in range(self.dimensions[0]):
                self.values[i].insert(c_index, col[i])
            self.dimensions[1] += 1

    def v_length(vector):
        if vector.is_vector():
            v = Matrix.copy(vector)
            return math.sqrt(v ** v)

    def normalize_vector(vector):
        if vector.is_vector():
            v = Matrix.copy(vector)
            if vector.v_length() > 0:
                quotient = 1 / vector.v_length()
                return v * quotient
            else:
                return v
        else:
            print("ERROR: The given argument is not a matrix.")

    @staticmethod
    def copy(matrix):
        if isinstance(matrix, Matrix):
            return Matrix.build_from_rows(matrix.values)
        else:
            print("ERROR: The given argument is not a matrix.")
            return None

    @staticmethod
    def build_from_rows(rows):
        matrix = Matrix([1, 1])
        matrix.valid = utils.check_build_from_arrays(rows)

        if matrix.valid:
            matrix.set_dimensions([len(rows), len(rows[0])])
            for i in range(len(rows)):
                for j in range(len(rows[0])):
                    matrix.values[i][j] = rows[i][j]

        return matrix

    @staticmethod
    def build_from_cols(cols):
        matrix = Matrix([1, 1])
        matrix.valid = utils.check_build_from_arrays(cols)

        if matrix.valid:
            matrix.set_dimensions([len(cols[0]), len(cols)])
            for i in range(len(cols[0])):
                for j in range(len(cols)):
                    matrix.values[i][j] = cols[j][i]

        return matrix

    def gaussian_elimination(self, augmented):
        a = Matrix.copy(self).sort_rows()

        for i in range(a.dimensions[1] - augmented):
            if a.values[i][i] != 0:
                a.multiply_row(i, 1 / a.values[i][i])
                for j in range(a.dimensions[1] - augmented):
                    if i != j:
                        if a.values[j][i] != 0:
                            row = a.get_row(i)
                            row = list(map(lambda x: x * -a.values[j][i], row))
                            a.add_row_to_row(j, row)
                a = a.cleanup()

        return a

    def gaussian_elimination_ref(self, augmented):
        for i in range(self.dimensions[1] - augmented):
            if self.values[i][i] != 0:
                self.multiply_row(i, 1 / self.values[i][i])
                row = self.get_row(i)
                for j in range(i, self.dimensions[0]):
                    if i != j:
                        self.add_row_to_row(
                            j, list(map(lambda x: x * -self.values[j][i], row))
                        )
                self = self.cleanup()

        return self

    def lu_decomposition(self):
        u = Matrix.copy(self)

        l = Matrix([u.dimensions[0], u.dimensions[1]])
        l.identity()

        for i in range(u.dimensions[1]):
            if u.values[i][i] != 0:
                for j in range(i + 1, u.dimensions[0]):
                    l.values[j][i] = u.values[j][i] / u.values[i][i]
                    row = list(
                        map(
                            lambda x: -x * u.values[j][i] / u.values[i][i], u.get_row(i)
                        )
                    )
                    u.add_row_to_row(j, row)

        return (l, u)

    def ldu_decomposition(self):
        if self.is_symmetric():
            pass
        else:
            print("ERROR: Can not do an LDU-decomposition on a non-symmetric matrix.")
            return None

        l, u = self.lu_decomposition()
        d = Matrix([self.dimensions[0], self.dimensions[1]])
        d.zeros()
        for i in range(self.dimensions[0]):
            d.values[i][i] = u.values[i][i]

        return (l, d, l.transpose())

    def transpose(self):
        matrix = Matrix([self.dimensions[1], self.dimensions[0]])
        matrix.zeros()
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                matrix.values[j][i] = self.values[i][j]
        return matrix

    def cholesky(self):
        if self.is_symmetric():
            pass
        else:
            print(
                "ERROR: Can not do a Cholesky decomposition on a non-symmetric matrix."
            )
            return None

        try:
            l = Matrix([self.dimensions[0], self.dimensions[1]])
            l.zeros()
            for i in range(self.dimensions[0]):
                for j in range(self.dimensions[1]):
                    if i == j:
                        l.values[i][i] = math.sqrt(
                            self.values[i][i]
                            - sum([l.values[j][k] ** 2 for k in range(j)])
                        )
                    else:
                        if i > j:
                            l.values[i][j] = (1 / l.values[j][j]) * (
                                self.values[i][j]
                                - sum(
                                    [l.values[i][k] * l.values[j][k] for k in range(j)]
                                )
                            )
                        else:
                            pass
            return l
        except Exception:
            print(
                "ERROR: Can not do a Cholesky decomposition on a non-positive-definite matrix."
            )
            return None

    def trace(self):
        return sum([self.values[i][i] for i in range(min([self.dimensions[0], self.dimensions[1]]))])


    def determinant(self):
        if self.dimensions[0] == self.dimensions[1]:
            if self.dimensions[0] == 2:
                return (
                    self.values[0][0] * self.values[1][1]
                    - self.values[0][1] * self.values[1][0]
                )

            l = self.cholesky()
            if l is not None:
                determinant = 1
                for i in range(l.dimensions[0]):
                    determinant *= l.values[i][i]
                return determinant ** 2

            else:
                determinant = 1
                l, u = self.lu_decomposition()
                for i in range(l.dimensions[0]):
                    determinant *= l.values[i][i] * u.values[i][i]
            return determinant

        else:
            print("ERROR: Can not calculate the determinant of a non-square matrix.")
            return None

    def qr_method(self, iterations):
        a = Matrix.copy(self)
        q, r = a.qr_householder()
        s = q

        for i in range(iterations):
            a = r * q
            q, r = a.qr_householder()
            s = s * q

        return a, s

    def find_eigenpairs(self):
        a = Matrix.copy(self)
        
        e, s = a.qr_method(100)
        
        eigenpairs = []

        for i in range(e.dimensions[0]):
            val = e.values[i][i]
            vec = Matrix.build_from_cols([s.get_col(i)]).cleanup()

            vector = vec.get_col(0)
            vector = list(map(lambda x: abs(x), vector))
            vector = list(filter(lambda x: x != 0, vector))
            if len(vector) != 0:
                vec = vec / min(vector)

            vec = vec.cleanup()

            if a * vec == vec * val:
                eigenpairs.append((val, vec))

        eigenpairs = sorted(eigenpairs, key=itemgetter(0), reverse=True)

        return eigenpairs


    def householder_reflection(self):
        x = Matrix.copy(self)

        x = Matrix.build_from_cols([x.get_col(0)])

        target = Matrix.copy(x)

        for i in range(target.dimensions[0]):
            if i == 0:
                target.values[i][0] = x.v_length()
            else:
                target.values[i][0] = 0

        i = Matrix([x.dimensions[0], x.dimensions[0]]).identity()

        v = target - x

        if (v.v_length() != 0):
            f = i - (((v * v.transpose()) / (v ** v)) * 2)
        else:
            f = i - (v * v.transpose())

        return f

    def qr_householder(self):
        r = Matrix.copy(self)
        eye = Matrix([r.dimensions[0], r.dimensions[1]]).identity()
        q_final = Matrix.copy(eye)

        for i in range(min([r.dimensions[1] - 1, r.dimensions[0]])):
            if i == 0:
                q = r.householder_reflection()
                r = q * r
            else:
                m = r.minor(i, i)
                q_m = m.householder_reflection()
                q = Matrix.copy(eye)
                for j in range(r.dimensions[0]-1, r.dimensions[0]-m.dimensions[0]-1, -1):
                    for k in range(r.dimensions[1]-1, r.dimensions[1]-m.dimensions[1]-1, -1):
                        q.values[j][k] = q_m.values[j-1][k-1]

                r = q * r
            
            q_final = q_final * q.transpose()

        q = Matrix.copy(q_final)

        return q, r

    def svd(self):
        m = Matrix.copy(self)
        sigma = Matrix([self.dimensions[0], self.dimensions[1]])
        l = m * m.transpose()
        q,t = l.qr_householder()

        r = m.transpose() * m

        q,t = r.qr_householder()

        eigenpairs_u = l.find_eigenpairs()
        eigenpairs_v = r.find_eigenpairs()


        for i in range(min([len(eigenpairs_u), len(eigenpairs_v)])):
            sigma.values[i][i] = math.sqrt(abs(eigenpairs_v[i][0]))

        u = Matrix.build_from_cols([vec.normalize_vector().get_col(0) for _, vec in eigenpairs_u])
        v = Matrix.build_from_cols([vec.normalize_vector().get_col(0) for _, vec in eigenpairs_v])



        return u, sigma, v

def main():
    pass

if __name__ == "__main__":
    main()
