import random
from typing import Union


Types = Union[int, float, bool]



class Matrix:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(cols)] for _ in range(rows)]
    
    @classmethod
    def from_data(cls, data: list[list]) -> 'Matrix':
        rows, cols = len(data), len(data[0])
        result = cls(rows, cols)
        result.data = data
        return result
    
    def __call__(self, w: list[list]) -> None:
        # self.data = w
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = w[i][j]

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols
        
    def __str__(self) -> str:
        return str(self.data).replace('],', '],\n')
    
    def __add__(self, other) -> 'Matrix':
        assert self.rows == other.rows and self.cols == other.cols, "Matrices must be of same size"

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self[i, j] + other[i, j]
        return result

    def __sub__(self, other) -> 'Matrix':
        assert self.rows == other.rows and self.cols == other.cols, "Matrices must be of same size"

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self[i, j] - other[i, j]
        return result
    
    def __mul__(self, other: Types) -> 'Matrix':
        assert isinstance(other, int) or isinstance(other, float), "Matrix must be multiplied by a scaler"
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] * other
        return result
    
    def __truediv__(self, other: Types) -> 'Matrix':
        assert other != 0, "Cannot divide by zero"
        assert isinstance(other, int) or isinstance(other, float), "Matrix must be divided by a scaler"

        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] / other
        return result
    
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        
        # if other.rows == 1:
        #     other = other.T

        assert self.cols == other.rows, "Matrices must be of same size"
        
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for k in range(other.cols):
                for j in range(self.cols):
                    result[i, k] += self[i, j] * other[j, k]
        return result

    def __getitem__(self, ij: tuple) -> Types:
        i, j = ij
        return self.data[i][j]

    def __setitem__(self, ij: tuple, value: Types) -> None:
        i, j = ij
        self.data[i][j] = value
    
    def __eq__(self, value) -> bool:
        if not isinstance(value, Matrix):
            return False

        if self.rows != value.rows or self.cols != value.cols:
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] != value[i, j]:
                    return False
        return True

    @property
    def T(self) -> 'Matrix':
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result


class Vector(Matrix):
    def __init__(self, rows):
        super().__init__(rows, 1)
    
    def __getitem__(self, ij: Union[tuple, int]):
        i, j = ij if isinstance(ij, tuple) else (ij, 0)
        return self.data[i][j]

    def __setitem__(self, ij: Union[tuple, int], value: Types):
        i, j = ij if isinstance(ij, tuple) else (ij, 0)
        self.data[i][j] = value

    def __len__(self):
        return self.rows
    
    def __eq__(self, value) -> bool:
        for i in range(self.rows):
            if self[i] != value[i]:
                return False
        return True

    @property
    def T(self):
        result = Matrix(1, self.rows)
        for i in range(self.rows):
            result[0, i] = self[i]
        return result
    
    @classmethod
    def all(cls, rows: int, value: float) -> 'Vector':
        result = cls(rows)
        result.__set_all(value)
        return result

    @classmethod
    def ones(self, rows: int) -> 'Vector':
        return Vector.all(rows, 1)
    
    @classmethod
    def zeros(self, rows: int) -> 'Vector':
        return Vector.all(rows, 0)
    
    def __set_all(self, value: float) -> None:
        for i in range(self.rows):
            self[i, 0] = value

    def to_diagonal_matrix(self) -> Matrix:
        if self.rows == 1:
            self = self.T
        result = Matrix(self.rows, self.rows)
        for i in range(self.rows):
            result[i, i] = self[i]
        return result
    
    @classmethod
    def from_data(cls_v, data: list[list]) -> 'Vector':
        rows = len(data)
        result = cls_v(rows)
        result.data = data
        return result
    
    def __add__(self, other) -> 'Vector':
        res = super().__add__(other)
        res = Vector.from_data(res.data)
        return res 

    def __sub__(self, other) -> 'Vector':
        res = super().__sub__(other)
        res = Vector.from_data(res.data)
        return res

    def hadamard_product(self, other) -> 'Vector':
        assert self.rows == other.rows, "Vectors must be of same size"

        result = Vector(self.rows)
        for i in range(self.rows):
            result[i] = self[i] * other[i]
        return result
        
    def __mul__(self, other: Types) -> 'Vector':
        res = super().__mul__(other)
        res = Vector.from_data(res.data)
        return res 
    
    def __rmul__(self, other: Types) -> 'Vector':
        return self.__mul__(other)
    
    def __truediv__(self, other: Types) -> 'Vector':
        res = super().__truediv__(other)
        res = Vector.from_data(res.data)
        return res

    def to_column_vector(self) -> 'Vector':
        self.data = [[0] for _ in range(self.cols)]
        for i in range(self.rows):
            self.data[i][0] = self[i]
            
    @staticmethod
    def random(rows: int, min: float, max: float) -> 'Vector':
        result = Vector(rows)
        for i in range(rows):
            result[i] = random.uniform(min, max)
        return result