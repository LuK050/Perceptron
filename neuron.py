from random import *


# Число Эйлера
E = 2.7182818284

# Функция активации нейрона. В данном случае сигмоида
def sigmoid(x: float, der: bool = False) -> float:
    return x * (1 - x) if der else 1 / (1 + E ** -x)


# Скалярное произведение двух массивов
def dot(x: list, y: list) -> list:
    return sum([j * w for j, w in zip(x, y)])

    
class Neuron:
     # weight - начальные рандомные веса для синапсов
     # bias - смещение нейрона
     def __init__(self, weight = None, bias = 0) -> None:
          self.weight = weight or [random(), random(), random()]
          self.bias = bias
     
     def compute(self, x):
          return [sigmoid(dot(i, self.weight) + self.bias) for i in x]
    
     
     def education(self, training, attempts):
          answer = None
          for _ in range(attempts):
               answer = self.compute(training["values"]) # результат от нейрона
               error = [x[0] - y for x, y in zip(training["answers"], answer)] # на сколько ошиблась нейросеть
               
               corrections = [x * y for x, y in zip(error, [sigmoid(i, True) for i in answer])] # корректировка весов синапсов
               self.weight = [x + y for x, y in zip(self.weight, [dot(i, corrections) for i in training["values"]])] 
          
          return self.weight, answer
