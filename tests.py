from la import Matrix, Vector, Types
from nn import NeuralNetwork
from reader import read


def matrix_tests():
    
    import numpy as np 
    
    w = np.random.rand(10, 10).tolist()
    w_np = np.array(w)
    
    rows, cols = len(w), len(w[0])
    # Constructor overloading is not supported in Python
    w_py = Matrix(rows, cols)
    w_py(w)
    
    np_res = w_np @ w_np
    py_res = w_py @ w_py
    
    assert np.allclose(np.array(py_res.data), np_res)
    print("Matrix multiplication test passed")
    
    # Test addition, subtraction
    w1 = np.random.rand(10, 10).tolist()
    w2 = np.random.rand(10, 10).tolist()
    w1_np = np.array(w1)
    w2_np = np.array(w2)
    w1_py = Matrix(rows, cols)
    w1_py(w1)
    w2_py = Matrix(rows, cols)
    w2_py(w2)
    
    assert np.allclose(np.array((w1_py + w2_py).data), w1_np + w2_np)
    print("Matrix addition test passed")

    assert np.allclose(np.array((w1_py - w2_py).data), w1_np - w2_np)
    print("Matrix subtraction test passed")
    
    
    # Test scalar multiplication, division
    for i in [2, 2.0]:
        assert np.allclose(np.array((w1_py * i).data), w1_np * i)
        assert np.allclose(np.array((w1_py / i).data), w1_np / i)

    print("Matrix scalar multiplication test passed")
    print("Matrix scalar division test passed")
    
    
    # Test matrix-vector multiplication
    v1 = np.random.rand(10, 1).tolist()
    v1_np = np.array(v1)
    v1_py = Vector(rows)
    v1_py(v1)
    
    assert np.allclose(np.array((w1_py @ v1_py).data), w1_np @ v1_np)
    print("Matrix-vector multiplication test passed")
    
    

def test_net(nets: list[NeuralNetwork]) -> None:
    for i, net in enumerate(nets):
        if net.example_input is None:
            continue
        assert net.example_output == net(net.example_input)
        print(f"{i}: {net} passed")
        
        
if __name__ == "__main__":
    # Testing the Linear Algebra library
    matrix_tests()
    
    # Testing the Neural Network library
    nets: list[NeuralNetwork] = read("data/networks.txt")
    test_net(nets)