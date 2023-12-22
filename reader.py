from nn import NeuralNetwork
from la import Matrix, Vector, Types


def read(filepath: str, example_available=True) -> list[NeuralNetwork]: 
    with open(filepath) as f:
        nets: list[NeuralNetwork] = []

        for line in f.readlines():
            if line.startswith("Layers"):
                layers = 0
                rows = []
                cols = []
                weights = []
                biases = []
                relus = []
                example_input = []
                example_output = []

                layers = int(line.strip().split(": ")[1])
                
            elif line.startswith("Rows"):
                row = int(line.strip().split(": ")[1])
                rows.append(row)
            elif line.startswith("Cols"):
                col = int(line.strip().split(": ")[1])
                cols.append(col)
            elif line.startswith("Weights"):
                weight = Matrix(rows[-1], cols[-1])
                w = eval(line.strip().split(": ")[1])
                weight(w)
                weights.append(weight)
            elif line.startswith("Biases"):
                bias_inp = eval(line.strip().split(": ")[1])
                bias = Vector(len(bias_inp))
                bias(bias_inp)
                biases.append(bias)
            elif line.startswith("Relu"):
                relu_str = line.strip().split(": ")[1]
                relu = True if relu_str == "true" else False
                relus.append(relu)

            else:
                if not example_available:
                    # Create the neural network
                    nn = NeuralNetwork(layers, rows, cols, weights, biases, relus)
                    nets.append(nn)
                
                elif line.startswith("Example_Input"):
                    inp = eval(line.strip().split(":")[1])
                    input_vector = Vector(len(inp))
                    input_vector(inp)
                    example_input.append(input_vector)
                elif line.startswith("Example_Output"):
                    outp = eval(line.strip().split(":")[1])
                    output_vector = Vector(len(outp))
                    output_vector(outp)
                    example_output.append(output_vector)

                    # Create the neural network
                    nn = NeuralNetwork(layers, rows, cols, weights, biases, relus, input_vector, output_vector)
                    nets.append(nn)

        if not example_available:
            nn = NeuralNetwork(layers, rows, cols, weights, biases, relus)
            nets.append(nn)

    return nets
