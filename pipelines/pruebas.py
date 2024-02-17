
import numpy as np
from solution import solution
import time
import random
# Donde  SearchAgents_no : Agentes de busqueda

SearchAgents_no =5
dim = 10
lb=-100
ub=100
Max_iter=10




# initialize alpha, beta, and delta_pos
# Las tres mejores soluciones o los 3 mejores lobos

Alpha_pos = np.zeros(dim)
Alpha_score = float("inf")


Beta_pos = np.zeros(dim)
Beta_score = float("inf")

Delta_pos = np.zeros(dim)
Delta_score = float("inf")


print(f"Dimenciones Alfa {Alpha_pos}, y fitnes {Alpha_score}")

# Esta linea solo chequea que los limites superiores e inferiores nos ean una lista lb y ub
# En caso de sean toma un valor especifico y devuelve el resultado


if not isinstance(lb, list):
    lb = [lb] * dim
if not isinstance(ub, list):
    ub = [ub] * dim


# Initialize the positions of search agents
# Aqui lo unico que esta haciendo inicialmete es creado la poblacion de lobos es decir los w
      
Positions = np.zeros((SearchAgents_no, dim))
print(Positions)


# Aqui lo unico que esta haciendo es generando muestras aleatorias con distribución uniforme en el rango [-100, 100]
# que corresponde a los limites Superiores (Ub) Y limites inferiores (lb) del espacio de muestra.
# Esta es una forma especifica de generar la población de lobos



for i in range(dim):
    Positions[:, i] = (
        np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    )


# Población de lobos
print(" Posiciones de la población de lobos: ",  Positions)



Convergence_curve = np.zeros(Max_iter)
s = solution()



# Loop counter
#print('GWO is optimizing  "' + objf.__name__ + '"')

timerStart = time.time()
s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")








for l in range(0, Max_iter):
    for i in range(0, SearchAgents_no):

        # Retornar los lobos de busqueda que van mas alla del espacio de busqueda.
        for j in range(dim):
            Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

        # Calculamos la funcion objetivo para acda agente de busqueda
        fitness = objf(Positions[i, :])

        # Update Alpha, Beta, and Delta
        if fitness < Alpha_score:
            Delta_score = Beta_score  # Update delta
            Delta_pos = Beta_pos.copy()
            Beta_score = Alpha_score  # Update beta
            Beta_pos = Alpha_pos.copy()
            Alpha_score = fitness
            # Update alpha
            Alpha_pos = Positions[i, :].copy()

        if fitness > Alpha_score and fitness < Beta_score:
            Delta_score = Beta_score  # Update delte
            Delta_pos = Beta_pos.copy()
            Beta_score = fitness  # Update beta
            Beta_pos = Positions[i, :].copy()

        if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
            Delta_score = fitness  # Update delta
            Delta_pos = Positions[i, :].copy()

    a = 2 - l * ((2) / Max_iter)
    # a decreases linearly fron 2 to 0

'''