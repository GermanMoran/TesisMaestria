import numpy as np
import random
print("Vector de valores aleatorios: ", np.random.rand(10))
print("Valor Aleatorio Especifico: ", np.random.uniform(0, 1))
print("Valor muy pequeño negativo: ", float("-inf"))


vector = [[3, 4, 7, 8], [6,8,9,0]]
print(len(vector))


for i in range(len(vector)):

    print(vector[i][0:len(vector[i])-1])


print("Numero aleatorio entre cero y uno: ", random.random())


for j in range(0,5):
    print(j)