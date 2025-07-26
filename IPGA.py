import random
import copy
from typing import List
import mutation_operator
from Individual import Individual  # Suponiendo que tu clase está en individual.py

def generate_population(n_max: int, N: int, M: int, S: int) -> List[Individual]:
    """
    Genera una población de n_max individuos, cada uno con una ruta permutada (1..N)
    y breakpoints válidos según los parámetros S (vendedores) y M (mínimo de ciudades por vendedor).
    """
    population = []
    for _ in range(n_max):
        route = random.sample(range(1, N + 1), N)  # Permutación aleatoria de 1 a N
        breakpoints = mutation_operator.modify_breaks(N, S, M)       # Breakpoints válidos
        individual = Individual(route=route, breakpoints=breakpoints)
        population.append(individual)
    return population

def evaluate_fitness(population, distance_matrix):
    """
    Calcula la distancia total para cada individuo en la población y su fitness.
    Las rutas se separan según los breakpoints, y cada subruta forma un ciclo.
    """
    for ind in population:
        route = ind.get_route()
        breakpoints = ind.get_breakpoints()
        subroutes = []

        # Dividir en subrutas usando breakpoints
        prev = 0
        for bp in breakpoints:
            subroutes.append(route[prev:bp])
            prev = bp
        subroutes.append(route[prev:])  # Última subruta

        total_distance = 0.0

        for subroute in subroutes:
            # Calcular distancia del ciclo (regresa al origen)
            for i in range(len(subroute)):
                city_from = subroute[i]
                city_to = subroute[(i + 1) % len(subroute)]  # siguiente o volver al inicio
                total_distance += distance_matrix[city_from - 1][city_to - 1]

        ind.set_total_distance(total_distance)
        ind.set_fitness(1.0 / total_distance if total_distance > 0 else 0.0)



def ipga_iteration(population, distance_matrix, num_salesmen, min_cities):
    """
    Aplica el operador genético de IPGA según el artículo original.
    Devuelve una nueva población generada a partir de mutaciones intensivas.
    """
    n = len(population)
    indices_disponibles = list(range(n))
    new_population = []

    while len(indices_disponibles) >= 10:
        # Paso 1: Seleccionar 10 individuos sin repetir
        seleccionados = random.sample(indices_disponibles, 10)
        for idx in seleccionados:
            indices_disponibles.remove(idx)

        grupo = [population[idx] for idx in seleccionados]

        # Paso 2: Elegir el mejor del grupo según fitness
        best = max(grupo, key=lambda ind: ind.get_fitness())

        descendants = []
        # Paso 3: Clonar 10 veces al mejor
        for _ in range(10):
            new_ind = Individual()
            new_ind.set_route(best.get_route()[:])  # copia de la lista
            new_ind.set_breakpoints(best.get_breakpoints()[:])  # copia de la lista
            new_ind.set_fitness(best.get_fitness())
            new_ind.set_total_distance(best.get_total_distance())
            descendants.append(new_ind)

        # Paso 4: Seleccionar segmentos I, J y posición P para mutaciones
        route_len = len(best.get_route())
        if route_len < 3:
            i, j = 0, route_len - 1
        else:
            i, j = sorted(random.sample(range(route_len), 2))
        p = random.randint(0, route_len)

        descendants = apply_mutations(descendants,i,j,p,len(best.get_route()),num_salesmen,min_cities)

        # Evaluar descendants y agregarlos
        evaluate_fitness(descendants, distance_matrix)
        new_population.extend(descendants)

    return new_population



def apply_mutations(descendants, I, J, P, N, S, M):
    if len(descendants) != 10:
        raise ValueError("Se esperaban exactamente 10 descendants.")

    # (0) nada

    # (1) FlipInsert
    descendants[1].set_route(mutation_operator.flip_insert(descendants[1].get_route(), I, J, P))

    # (2) SwapInsert
    descendants[2].set_route(mutation_operator.swap_insert(descendants[2].get_route(), I, J, P))

    # (3) LSlideInsert
    descendants[3].set_route(mutation_operator.lslide_insert(descendants[3].get_route(), I, J, P))

    # (4) RSlideInsert
    descendants[4].set_route(mutation_operator.rslide_insert(descendants[4].get_route(), I, J, P))

    # (5) Modify Breaks
    descendants[5].set_breakpoints(mutation_operator.modify_breaks(N, S, M))

    # (6) FlipInsert + Modify Breaks
    descendants[6].set_route(mutation_operator.flip_insert(descendants[6].get_route(), I, J, P))
    descendants[6].set_breakpoints(mutation_operator.modify_breaks(N, S, M))

    # (7) SwapInsert + Modify Breaks
    descendants[7].set_route(mutation_operator.swap_insert(descendants[7].get_route(), I, J, P))
    descendants[7].set_breakpoints(mutation_operator.modify_breaks(N, S, M))

    # (8) LSlideInsert + Modify Breaks
    descendants[8].set_route(mutation_operator.lslide_insert(descendants[8].get_route(), I, J, P))
    descendants[8].set_breakpoints(mutation_operator.modify_breaks(N, S, M))

    # (9) RSlideInsert + Modify Breaks
    descendants[9].set_route(mutation_operator.rslide_insert(descendants[9].get_route(), I, J, P))
    descendants[9].set_breakpoints(mutation_operator.modify_breaks(N, S, M))

    return descendants



def ipga(k: int,
         n_max: int,
         N: int,
         M: int,
         S: int,
         distance_matrix) -> Individual:
    """
    Ejecuta el algoritmo IPGA completo por k iteraciones.
    Devuelve el mejor individuo encontrado.
    """
    # Paso 1: Generar población inicial
    population = generate_population(n_max=n_max, N=N, M=M, S=S)

    # Paso 2: Evaluar población inicial
    evaluate_fitness(population, distance_matrix)

    # Paso 3: Inicializar mejor solución observada
    best_solution = Individual()
    best_solution.set_fitness(-1.0)

    # Paso 4: Iteraciones del IPGA
    for iteration in range(k):
        population = ipga_iteration(
            population=population,
            distance_matrix=distance_matrix,
            num_salesmen=S,
            min_cities=M
        )

        # Buscar mejor individuo en la iteración actual
        current_best = max(population, key=lambda ind: ind.get_fitness())

        # Actualizar si mejora el mejor global
        if current_best.get_fitness() > best_solution.get_fitness():
            best_solution = current_best

        print(f"Iteración {iteration+1}: Best fitness = {current_best.get_fitness():.4f}")

    return best_solution
