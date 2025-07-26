class Individual:
    def __init__(self, route=None, breakpoints=None, fitness=0.0, total_distance = 0.0):
        self._route = route if route is not None else []
        self._breakpoints = breakpoints if breakpoints is not None else []
        self._fitness = fitness
        self._total_distance = total_distance

    # Getters
    def get_route(self):
        return self._route

    def get_breakpoints(self):
        return self._breakpoints

    def get_fitness(self):
        return self._fitness

    def get_total_distance(self):
        return self._total_distance

    # Setters
    def set_route(self, route):
        self._route = route

    def set_breakpoints(self, breakpoints):
        self._breakpoints = breakpoints

    def set_fitness(self, fitness):
        self._fitness = fitness
    
    def set_total_distance(self, total_distance):
        self._total_distance = total_distance

    # String representation
    def __str__(self):
        return (
            f"Individual:\n"
            f"  Route: {self._route}\n"
            f"  Breakpoints: {self._breakpoints}\n"
            f"  Fitness: {self._fitness:.4f}"
        )
