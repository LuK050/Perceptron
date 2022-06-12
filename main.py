from neuron import Neuron


training = {
        "values": [[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1]],
        "answers": [[0], [1], [1]]
    }

# Создание и обучение нейрона с сдвигом -3
neuron = Neuron(bias = -3)
neuron.education(training, 5000)

while True:
    x = (input("> ")).split(" ")
    result = neuron.compute([[int(i) for i in x]])[0]
    print("Результат:", result, "≈", round(result), "\n")