import random
from copy import deepcopy
from typing import List


def mutate_population(population: List, route_mut_prob: float, break_mut_prob: float,
                      num_salesmen: int, min_cities: int):
    for individual in population:
        if random.random() < route_mut_prob:
            mutate_route(individual)
        if random.random() < break_mut_prob:
            mutate_breakpoints(individual, num_salesmen, min_cities)


def mutate_route(individual):
    route = individual.get_route()
    n = len(route)
    if n < 2:
        return

    I, J = sorted(random.sample(range(n), 2))
    P = random.randint(0, n - (J - I + 1))

    operation = random.choice([
        flip_insert,
        swap_insert,
        lslide_insert,
        rslide_insert
    ])

    mutated_route = operation(deepcopy(route), I, J, P)
    individual.set_route(mutated_route)


def flip_insert(route, I, J, P):
    segment = route[I:J+1][::-1]
    del route[I:J+1]
    for idx, val in enumerate(segment):
        route.insert(P + idx, val)
    return route


def swap_insert(route, I, J, P):
    route[I], route[J] = route[J], route[I]
    segment = route[I:J+1]
    del route[I:J+1]
    for idx, val in enumerate(segment):
        route.insert(P + idx, val)
    return route


def lslide_insert(route, I, J, P):
    segment = route[I:J+1]
    segment = segment[1:] + [segment[0]]
    del route[I:J+1]
    for idx, val in enumerate(segment):
        route.insert(P + idx, val)
    return route


def rslide_insert(route, I, J, P):
    segment = route[I:J+1]
    segment = [segment[-1]] + segment[:-1]
    del route[I:J+1]
    for idx, val in enumerate(segment):
        route.insert(P + idx, val)
    return route


def mutate_breakpoints(individual, num_salesmen: int, min_cities: int):
    route_length = len(individual.get_route())
    new_breakpoints = modify_breaks(route_length, num_salesmen, min_cities)
    individual.set_breakpoints(new_breakpoints)


def modify_breaks(N: int, S: int, m: int) -> List[int]:
    # cantidad mínima de ciudades requeridas por la separación
    min_total = m * S
    if min_total > N:
        raise ValueError("No se pueden asignar al menos m ciudades por segmento con el N dado.")
    
    # ciudades extra a distribuir libremente
    extras = N - min_total

    # repartir extras aleatoriamente entre los S segmentos
    additions = [0] * S
    for _ in range(extras):
        additions[random.randint(0, S - 1)] += 1

    # longitud real de cada segmento
    lengths = [m + x for x in additions]

    # calcular los puntos de ruptura acumulando las longitudes
    breaks = []
    acc = 0
    for length in lengths[:-1]:  # S - 1 rupturas
        acc += length
        breaks.append(acc)
    
    return breaks

