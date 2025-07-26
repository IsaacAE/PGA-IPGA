import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import List
import numpy as np
import tsplib95
import PGA, IPGA

def read_tsp_coordenates(filepath):
    coords = []
    read = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                read = True
                continue
            if read:
                if line == 'EOF' or line == '':
                    break
                pieces = line.split()
                if len(pieces) >= 3:
                    x, y = float(pieces[1]), float(pieces[2])
                    coords.append([x, y])
    return np.array(coords)


def tsp_to_distance_matrix(file_path):
    problem = tsplib95.load(file_path)
    n = problem.dimension
    nodes = list(problem.get_nodes())
    
    # Inicializar matrix vacía
    matrix = np.zeros((n, n))

    # Llenar la matrix de distances
    for i_index, i in enumerate(nodes):
        for j_index, j in enumerate(nodes):
            matrix[i_index, j_index] = problem.get_weight(i, j)

    return matrix

def plot_tsp_nodes(file_path):
    problem = tsplib95.load(file_path)

    if not problem.node_coords:
        raise ValueError("El archivo .tsp no contiene coordenadas explícitas.")

    coords = problem.node_coords

    x_vals = []
    y_vals = []
    labels = []

    for node, (x, y) in coords.items():
        x_vals.append(x)
        y_vals.append(y)
        labels.append(str(node))

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, color='blue', zorder=5)
    
    # Etiquetas de nodos
    for i, label in enumerate(labels):
        plt.text(x_vals[i], y_vals[i], label, fontsize=9, ha='right', va='bottom')

    plt.title("Ubicaciones de las cities (TSP)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")  # Mantener proporción
    plt.show()



def plot_feasible_solution(individual, coords):
    route = individual.get_route()  # Lista de ciudades (ej. [1,7,8,3,2,4,6,5])
    breakpoints = individual.get_breakpoints()  # Lista de índices de corte (ej. [3,6])

    if not isinstance(route, list) or not isinstance(breakpoints, list):
        raise TypeError("Los atributos 'route' y 'breakpoints' deben ser listas.")

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords debe tener forma (n, 2), con coordenadas 2D por ciudad.")

    # Obtener rutas individuales según los breakpoints
    breakpoints_sorted = sorted(breakpoints)
    routes: List[List[int]] = []
    start = 0
    for bp in breakpoints_sorted:
        routes.append(route[start:bp])
        start = bp
    routes.append(route[start:])  # último segmento

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps['tab10']
    depots = set()

    for idx, r in enumerate(routes):
        if not r:
            continue

        # Cerrar el ciclo
        if r[0] != r[-1]:
            r.append(r[0])

        route_idx = [i - 1 for i in r]  # Base 1 a base 0
        if any(i >= len(coords) or i < 0 for i in route_idx):
            raise IndexError(f"Ruta {idx} contiene índices fuera de rango válido.")

        route_coords = coords[route_idx]
        color = cmap(idx % 10)

        for i in range(len(route_coords) - 1):
            x0, y0 = route_coords[i]
            x1, y1 = route_coords[i + 1]

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle='->',
                color=color,
                linewidth=2,
                mutation_scale=10,
            )
            ax.add_patch(arrow)

        # Nodo de inicio
        depot_idx = route_idx[0]
        depots.add(depot_idx)
        ax.plot([], [], color=color, label=f'route {idx + 1}')

    # Dibujar nodos
    for idx, (x, y) in enumerate(coords):
        if idx in depots:
            ax.scatter(x, y, c='gold', edgecolor='black', s=41, zorder=5)
        else:
            ax.scatter(x, y, c='black', s=40, zorder=5)
        ax.text(x + 0.1, y + 0.1, str(idx + 1), fontsize=9)

     # Título con fitness y distancia total
    fitness = individual.get_fitness()
    distance = individual.get_total_distance()

   
    ax.set_title(f"routes del individual\nfitness: {fitness:.7f} | distance: {distance:.2f}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    path_file="Instances/bier127.tsp"
    problem = tsplib95.load(path_file)
    n = problem.dimension
    distance_matrix= tsp_to_distance_matrix(path_file)
    coords = read_tsp_coordenates(path_file)
    # Parámetros del problema
    N = n        # Número de ciudades
    M = 10       #Mínimo de ciudades por visitar
    S = 4          # Número de agentes viajeros
    n_max = 100      # Tamaño de la población
    k = 100     # Número de iteraciones
    gamma = 0.20    # Porcentaje de elitismo (20%)

    

    # Ejecutar el algoritmo PGA
    '''best_solution = PGA.pga(n_max=n_max, N=N, M=M, S=S, k=k, gamma=gamma, distance_matrix=distance_matrix)
    print(best_solution)
    # Ploteo del mejor individuo
    plot_feasible_solution(best_solution, coords)'''
    best_solution = IPGA.ipga(k=k, n_max = n_max, N = N, M=M, S=S, distance_matrix = distance_matrix)
    print(best_solution)
    # Ploteo del mejor individuo
    plot_feasible_solution(best_solution, coords)


if __name__ == "__main__":
    main()