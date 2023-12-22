Details of the libraries for GitHub:

Linear Algebra Library:
-----------------------
    la.py

    This library contains Matrix and Vector classes. The libraries are written to have functionality very similar to numpy.
    However, since the operators are implemented in Python (using Python for loops), the performance of the operators is
    slower than numpy. Only the random module from Python's standard library is used in this library. No other external
    libraries are used. The library contains the following classes:

    1. Matrix: 
        1.0 Construction: 
            Matrix(rows, cols) -> Matrix and then Matrix(data) -> Matrix
            or Matrix.from_data(data) -> Matrix
            The data has to be a list of lists, where each list is a row of the matrix.
        
        1.1 Addition: 
            Matrix + Matrix -> Matrix
        
        1.2 Subtraction: 
            Matrix - Matrix -> Matrix
        
        1.3 Matrix Multiplication: 
            Matrix @ Matrix -> Matrix 
            or Matrix @ Vector -> Matrix
        
        1.4 Scalar Multiplication (Element-wise): 
            Matrix * Scalar -> Matrix
        
        1.5 Scaler Division (Element-wise): 
            Matrix / Scalar -> Matrix
        
        1.6 Transpose: 
            Matrix.T -> Matrix

        1.7 Get item: 
            Matrix[i, j] -> Scalar
        
        1.8 Set item: 
            Matrix[i, j] = Scalar
        
        1.9 Equality check:
            Matrix == Matrix -> Boolean

        1.10 Shape:
            Matrix.shape -> (rows, cols)


    2. Vector:
        Inherited from the Matrix class. The only difference is that the Vector class has only one column.
        
        Additional or overwritten methods:
        2.0 Construction: 
            Vector(size) -> Vector (returns a zero vector of dimension = (size x 1))
            or, Vector.from_data(data) -> Vector
            or, Vector(data) -> Vector
            The data has to be a list of scalars.

            Vector.ones(rows) -> Vector (returns a vector of ones of dimension = (rows x 1))
            Vector.zeros(rows) -> Vector (returns a vector of zeros of dimension = (rows x 1))
            Vector.random(rows, min, max) -> Vector (returns a vector of random numbers between min and max of dimension = (size x 1))

        2.1 Length of the vector:
            len(Vector) -> int (returns the number of rows)
        
        2.2 Vector to Diagonal Matrix:
            Vector.to_diagonal_matrix() -> Matrix (returns a diagonal matrix with the vector as the diagonal)
        
        2.3 Vector to column vector:
            Vector.to_column_vector() -> Vector (returns the vector as a column vector)
        
        2.4 Hadamard Product:
            Vector.hadamard_product(Vector) -> Vector (returns the element-wise product of the two vectors)


Neural Network Library:
-----------------------
    nn.py

    Contains the NeuralNetwork class.

    Calling an instance of the NeuralNetwork class with a valid input will evaluate the feed-forward network and return the
    output. The input has to be a Vector and the output will also be a Vector. 

    Evaluation of the feed-forward network:
    The feed-forward network is evaluated using a recursion. The recursive function is called to start the evaluation of the
    first layer. The output of the first layer is then passed to the next layer. The recursive function is just called again
    with information about the next layer and the recently computed input to the next layer. The recursion terminates once
    the last layer is reached. Then the output of the last layer is returned.
    
    Optimization of the loss function: 
    ----------------------------------
    
    The goal was to find an input vector for which the sum of absolute values of the elements of the output vector is 0.
    This is equivalent to minimizing the loss function: sum(abs(output_vector)). Since the absolute values of the
    elements will be non-negative the minimum value of the loss function can not be smaller than 0. To optimize this
    loss function, random search worked sufficiently well (losses were less than the sample losses that were given for
    the 10 networks).

    To use random search to minimize the loss function, first a zero input Vector is created. Then, a random Vector is
    created at each iteration. The random Vector is then added to the input Vector and the loss function is evaluated.
    If the loss is smaller than the previous loss, the input Vector is updated to the new input Vector. This is repeated
    for some maximum number of iterations.

    To find the random Vector, two approaches were implemented. The first approach was to create a random Vector with
    values between -1 and 1. The second approach created a random Vector with values between -max(0.00001, 1/iteration)
    and max(0.00001, 1/iteration). This approach decreased the search space with the number of iterations. The assumption
    here is that with the increase in the number of iterations better quality of solution will be found. So, reducing
    the search space with increasing iterations should help reach a better solution.



-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
Gradient descent with momentum and adaptive learning rate using backtracking line search was also implemented.
Stochastic gradient descent can also be allowed. The way SGD is implemented here is that, with 0.5 probability some
element of the gradient is set to 0. This ensures that a step is not taken in the direction of the gradient in those
dimensions. This helps escape local minima by moving to a potentially sub-optimal solution. The gradient calculation
here may not be correct [NEED TO FIX LATER]. I just used random search to minimize the loss function. Random search
worked well enough.
