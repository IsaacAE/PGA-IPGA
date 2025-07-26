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


def modify_breaks(N: int, S: int, M: int) -> List[int]:
    max_attempts = 1000
    for _ in range(max_attempts):
        breaks = sorted(random.sample(range(1, N), S - 1))
        if (all(b2 - b1 >= M for b1, b2 in zip([0] + breaks, breaks + [N]))
                and breaks[0] >= M
                and N - breaks[-1] >= M):
            return breaks
    raise ValueError("Unable to generate valid breakpoints with given constraints.")
