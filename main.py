from neuron import *


training = {
        "values": [[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1]],
        "answers": [[0], [1], [1]]
    }

# Создание и обучение нейрона с сдвигом -3
neuron = Neuron(bias = -3)
neuron.education(training, 1000)

while True:
    x = [[int(input()), int(input()), int(input())]]
    print(x)
    print(neuron.compute(x))