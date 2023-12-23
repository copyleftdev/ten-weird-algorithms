import numpy as np

def calculate_total_distance(route, distance_matrix):
    """Calculate the total distance of a given route."""
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i-1]][route[i]]
    return total_distance

def swap_two_cities(route):
    """Swap two cities in the route."""
    a, b = np.random.choice(len(route), 2, replace=False)
    route[a], route[b] = route[b], route[a]
    return route

def simulated_annealing_tsp(distance_matrix, initial_temp, min_temp, alpha):
    """Simulated annealing algorithm to solve the Traveling Salesman Problem."""

    # Initialize with a random route
    num_cities = len(distance_matrix)
    current_route = list(np.random.permutation(num_cities))
    current_distance = calculate_total_distance(current_route, distance_matrix)
    
    best_route = current_route.copy()
    best_distance = current_distance
    temp = initial_temp

    while temp > min_temp:
        # Create new route by swapping two cities and calculate its distance
        candidate_route = swap_two_cities(current_route.copy())
        candidate_distance = calculate_total_distance(candidate_route, distance_matrix)

        # Decide if we should accept the new route
        if candidate_distance < current_distance or np.random.rand() < np.exp((current_distance - candidate_distance) / temp):
            current_route = candidate_route
            current_distance = candidate_distance

            # Update best route found
            if candidate_distance < best_distance:
                best_route = candidate_route
                best_distance = candidate_distance

        # Cool down the temperature
        temp *= alpha

    return best_route, best_distance

# Example: A simple distance matrix for demonstration
distance_matrix = np.array([[ 0, 10,  5,  8],
       [10,  0,  9,  6],
       [ 5,  9,  0,  9],
       [ 8,  6,  9,  0]])

# Parameters for the simulated annealing
initial_temp = 100000
minimum_temp = 1
cooling_rate = 0.995

# Run the algorithm
best_route, best_distance = simulated_annealing_tsp(distance_matrix, initial_temp, minimum_temp, cooling_rate)

print("Best Route:", best_route)
print("Total Distance of Best Route:", best_distance)

