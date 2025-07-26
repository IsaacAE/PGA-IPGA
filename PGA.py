import random
from typing import List
from mutation_operators import *
from Individual import Individual  # Suponiendo que tu clase está en individual.py


def generate_population(n_max: int, N: int, M: int, S: int) -> List[Individual]:
    """
    Genera una población de n_max individuos, cada uno con una ruta permutada (1..N)
    y breakpoints válidos según los parámetros S (vendedores) y M (mínimo de ciudades por vendedor).
    """
    population = []
    for _ in range(n_max):
        route = random.sample(range(1, N + 1), N)  # Permutación aleatoria de 1 a N
        breakpoints = modify_breaks(N, S, M)       # Breakpoints válidos
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

import random
from typing import List

def roulette_elitist_selection(population: List, gamma: float) -> List:
    """
    Selecciona una nueva población usando selección por ruleta con elitismo.
    gamma: fracción (0 < gamma <= 1) de la población original a retener.
    """
    n_selected = max(1, int(len(population) * gamma))
    if n_selected > len(population):
        raise ValueError("gamma selecciona más individuos que los existentes.")

    # Ordenar por fitness descendente
    sorted_pop = sorted(population, key=lambda ind: ind.get_fitness(), reverse=True)

    # Elitismo: conservar al mejor individuo
    new_population = [sorted_pop[0]]

    # Construir ruleta
    total_fitness = sum(ind.get_fitness() for ind in population)
    if total_fitness == 0:
        # Si todos los fitness son 0, seleccionar al azar (menos el elitista)
        selected = random.sample(population[1:], n_selected - 1)
        new_population += selected
        return new_population

    # Crear ruleta acumulativa
    cumulative_probs = []
    cumulative = 0.0
    for ind in population:
        cumulative += ind.get_fitness() / total_fitness
        cumulative_probs.append(cumulative)

    # Selección por ruleta
    while len(new_population) < n_selected:
        r = random.random()
        for idx, cp in enumerate(cumulative_probs):
            if r <= cp:
                new_population.append(population[idx])
                break

    return new_population

def pga_iteration(population: List,distance_matrix: List[List[float]], gamma: float, route_mut_prob: float, break_mut_prob: float, num_salesmen: int, min_cities: int) -> List:
    """
    Ejecuta una iteración del PGA:
    - Evalúa la población
    - Selecciona mediante ruleta + elitismo
    - Muta individuos seleccionados
    - Repite hasta restaurar el tamaño original
    """
    n_max = len(population)

    # Evaluar fitness de la población actual
    evaluate_fitness(population, distance_matrix)

    # Selección inicial
    selected = roulette_elitist_selection(population, gamma)

    # Nueva población en crecimiento
    new_population = selected.copy()

    while len(new_population) < n_max:
        # Clonar seleccionados para no alterar originales
        clones = [ind for ind in selected]
        mutate_population(clones, route_mut_prob, break_mut_prob, num_salesmen, min_cities)

        # Evaluar clones mutados
        evaluate_fitness(clones, distance_matrix)

        # Agregar hasta alcanzar n_max
        space_left = n_max - len(new_population)
        new_population.extend(clones[:space_left])

    return new_population

def PGA(k: int,
        n_max: int,
        N: int,
        M: int,
        S: int,
        gamma: float,
        route_mut_prob: float,
        break_mut_prob: float,
        distance_matrix) -> Individual:
    """
    Ejecuta el algoritmo PGA completo por k iteraciones.
    Devuelve el mejor individuo encontrado.
    """
    # Población inicial
    population = generate_population(n_max=n_max, N=N, M=M, S=S)

    # Evaluar población inicial
    evaluate_fitness(population, distance_matrix)

    # Inicializar best_solution con fitness -1
    best_solution = Individual()
    best_solution.set_fitness(-1.0)

    for iteration in range(k):
        # Ejecutar una iteración del PGA
        population = pga_iteration(
            population=population,
            distance_matrix=distance_matrix,
            gamma=gamma,
            route_mut_prob=route_mut_prob,
            break_mut_prob=break_mut_prob,
            num_salesmen=S,
            min_cities=M
        )

        # Buscar mejor individuo de la iteración
        current_best = max(population, key=lambda ind: ind.get_fitness())

        # Actualizar mejor solución global si mejora
        if current_best.get_fitness() > best_solution.get_fitness():
            best_solution = current_best

        print(f"Iteración {iteration+1}: Best fitness = {current_best.get_fitness():.4f}")

    return best_solution