from nn import *
from reader import read

def main():
    
    nets1: list[NeuralNetwork] = read("data/networks.txt")

    with open("data/inputs.txt", 'w') as input_writer, \
        open("data/losses.txt", 'w') as loss_writer:

        for i, net in enumerate(nets1):
            # inp = net.optimize(optimizer=Optimizer.LocalRandomSearch)
            max_allowed_loss = 2 if i == 2 else 0.0001
            inp = net.loss_minimizer_input(max_allowed_loss=max_allowed_loss, network_index=i)

            # print(inp)
            loss = net.loss_function(net(inp))
            
            input_string = str.join(",", [str(row[0]) for row in inp.data])
            input_writer.write(f"{input_string}\n")
            loss_writer.write(f"{loss}\n")
            
            print(f"Network {i} done.")
            
if __name__ == "__main__":
    main()