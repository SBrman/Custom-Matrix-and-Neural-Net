import random
from la import Matrix, Vector, Types
from enum import Enum, auto
from typing import Union


class Optimizer(Enum):
    GradientDescent = auto()
    LocalRandomSearch = auto()
    


class NeuralNetwork:
    def __init__(self, layers, rows, cols, weights, biases, relus, example_input=None, example_output=None) -> None:
        self.layers: int = layers
        self.rows: list[int] = rows
        self.cols: list[int] = cols 
        self.weights: list[Matrix] = weights
        self.biases: list[Vector] = biases 
        self.relus: list[bool] = relus 

        # Example input and output
        self.example_input: Union[Vector, None] = example_input
        self.example_output: Union[Vector, None] = example_output
        
        # Store the derivatives of the loss function wrt inputs for backpropagation
        self.dx: list = []

    def __str__(self) -> str:
        return f"NN < RxCxrelu = {list(zip(self.rows, self.cols, self.relus))} >"

    def __call__(self, input_: Vector) -> Vector:
        return self.feed_forward(input_, 0)

    def feed_forward(self, input_: Union[Matrix, Vector], layer: int) -> Vector:
        if layer == self.layers:
            return Vector.from_data(input_.data)

        out = self.weights[layer] @ input_ + self.biases[layer]
        # Store the derivative of the loss function wrt inputs
        self.gradient_tape(out, layer)
        
        if self.relus[layer]:
            out = self.relu(out)
            self.gradient_tape(out, layer, relu=True)

        return self.feed_forward(out, layer+1)
    
    def gradient_tape(self, out: Union[Matrix, Vector], layer: int, relu: bool = False) -> None:
        if not relu:
            self.dx.append(self.weights[layer])
        else:
            vec: Vector = Vector.zeros(out.rows)
            for row in range(out.rows):
                vec[row, 0] = 1 if out[row, 0] > 0 else 0
            self.dx.append(vec.to_diagonal_matrix())

    def gradient(self, input_: Vector) -> Vector:
        """
        TODO: Fix this, shape related issues in the first layer
        Just use numerical gradient for now
        """
        self.dx = []
        self(input_)

        grad: Vector = self.dx[-1]
        for i in range(self.layers-1, -1, -1):
            grad = grad @ self.dx[i]
        
        # Making it row matrix
        grad_data = [[i] for i in grad.data[0]]
        return Vector.from_data(grad_data)
    
    def numerical_gradient(self, input_: Vector) -> Vector:
        grad: Vector = Vector.zeros(input_.rows)
        EPSILON = 0.0001
        for i in range(input_.rows):
            input_[i, 0] += EPSILON
            loss1 = self.loss_function(self(input_))
            input_[i, 0] -= 2*EPSILON
            loss2 = self.loss_function(self(input_))
            grad[i, 0] = (loss1 - loss2) / (2 * EPSILON)
        return grad

    @staticmethod
    def relu(x: Union[Matrix, Vector]) -> Vector | Matrix:
        for i in range(x.rows):
            x[i, 0] = max(0, x[i, 0])
        return x
    
    @staticmethod
    def loss_function(solution: Vector) -> float:
        loss = 0
        for i in range(solution.rows):
            for j in range(solution.cols):
                loss += abs(solution[i, j])
        return loss
    
    def stochastic_gradient_deletion(self, grad: Vector) -> Vector:
        for i in range(grad.rows):
            if random.random() < 0.5:
                grad[i, 0] = 0
        return grad
    
    def step_size_with_backtracking_line_search(self, solution: Vector, gradient: Vector) -> float:
        best_step_size = 1
        step_size = 1
        for i in range(1, 1000):
            # Diminishing step size 
            step_size = 1 / i
            # gradient descent update
            new_solution = solution - step_size * gradient
            if (self.loss_function(self(new_solution)) >= self.loss_function(self(solution))):
                return step_size

        return step_size
            
    def __gradient_descent(self, stochastic: bool = False, max_iter: int = 1000, beta: float = 0.9):
        inp = Vector.zeros(self.weights[0].cols)
        grad = self.numerical_gradient(inp)
        prev_grad = grad

        best_loss = float("inf")
        step_size = 0.001
        beta = 0.9
        
        for i in range(max_iter):
            # grad = self.numerical_gradient(inp)
            grad = self.gradient(inp)

            # Stochastic gradient descent with momentum
            if stochastic:
                grad = self.stochastic_gradient_deletion(grad)

            grad_with_momentum = beta * prev_grad + (1 - beta) * grad

            # Get an adaptive step size using backtracking line search
            step_size = self.step_size_with_backtracking_line_search(inp, grad_with_momentum)
            
            c_inp = inp - step_size * grad_with_momentum
            c_loss = self.loss_function(self(c_inp))

            if c_loss < best_loss:
                best_loss = c_loss
                inp = c_inp

            if c_loss < 0.00001:
                break

            prev_grad = grad_with_momentum

        return inp
    
    def __local_random_search(self, max_iter: int = 10001, diminishing_search_step: bool = True):
        input_size = self.weights[0].cols
        inp: Vector = Vector.zeros(input_size)
        best_loss: float = self.loss_function(self(inp))
        
        for i in range(1, max_iter):
            if diminishing_search_step:
                # Diminishing search step with a minimum of 0.00001
                d_inp = Vector.random(input_size, -max(1/i, 0.00001), max(1/i, 0.00001))
            else:
                d_inp = Vector.random(input_size, -1, 1)

            current_loss = self.loss_function(self(inp + d_inp))
            if current_loss < best_loss:
                best_loss = current_loss
                inp = inp + d_inp
            
            if current_loss < 0.0001:
                break

        return inp

    def optimize(self, optimizer: Optimizer = Optimizer.GradientDescent) -> Vector:
        if optimizer.value == Optimizer.GradientDescent.value:
            return self.__gradient_descent()
        elif optimizer.value == Optimizer.LocalRandomSearch.value:
            return self.__local_random_search()
        else:
            raise NotImplementedError("Optimizer not implemented")
        
    def loss_minimizer_input(self, max_allowed_loss: float, network_index: int) -> Vector:
        max_iterations = 10001

        for i in range(2):

            solT = self.__local_random_search(max_iter=max_iterations, diminishing_search_step=True)
            lossT = self.loss_function(self(solT))

            # Only network 3 was struggling to get below 2 which is the sample loss for network 3, so I set the threshold to 2
            if lossT < max_allowed_loss:
                return solT

            sol = self.__local_random_search(max_iter=max_iterations, diminishing_search_step=False)
            loss = self.loss_function(self(sol))
            if loss < max_allowed_loss:
                return sol
            
            if network_index > 6:
                break
            
            max_iterations *= 10
            

        return solT if lossT < loss else sol



