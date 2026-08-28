import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


class PDPTWSolver:
    """
    Solver for the Pickup and Delivery Problem with Time Windows (PDPTW)
    using Google OR-Tools.
    """

    def __init__(self, data):
        self.data = data
        self.manager = None
        self.routing = None
        self.solution = None
        self.distance_matrix = None
        self.time_matrix = None

    def validate_data(self):
        required = {
            "locations",
            "num_vehicles",
            "depot",
            "pickups_deliveries",
            "time_windows",
        }
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"Missing required data keys: {sorted(missing)}")

        n = len(self.data["locations"])
        if len(self.data["time_windows"]) != n:
            raise ValueError("time_windows must contain one window per location.")

        depot = self.data["depot"]
        if not 0 <= depot < n:
            raise ValueError("depot index is out of range.")

        if self.data["num_vehicles"] < 1:
            raise ValueError("num_vehicles must be at least 1.")

        for pickup, delivery in self.data["pickups_deliveries"]:
            if not (0 <= pickup < n and 0 <= delivery < n):
                raise ValueError(
                    f"Invalid pickup-delivery pair: ({pickup}, {delivery})"
                )
            if pickup == delivery:
                raise ValueError("Pickup and delivery nodes must be different.")

    def create_distance_matrix(self):
        """Create an integer Euclidean distance matrix."""
        locations = self.data["locations"]
        n = len(locations)
        matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            x1, y1 = locations[i]
            for j in range(n):
                if i == j:
                    continue
                x2, y2 = locations[j]
                matrix[i, j] = int(round(np.hypot(x1 - x2, y1 - y2)))

        return matrix

    def create_time_matrix(self, speed=5):
        """Create travel-time matrix from distances and vehicle speed."""
        if speed <= 0:
            raise ValueError("speed must be greater than 0.")

        if self.distance_matrix is None:
            self.distance_matrix = self.create_distance_matrix()

        return np.ceil(self.distance_matrix / speed).astype(int)

    def solve(self, speed=5, service_time=10, time_limit_seconds=30):
        """Solve the PDPTW problem."""
        self.validate_data()

        num_locations = len(self.data["locations"])
        num_vehicles = self.data["num_vehicles"]
        depot = self.data["depot"]

        self.manager = pywrapcp.RoutingIndexManager(
            num_locations,
            num_vehicles,
            depot,
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.distance_matrix = self.create_distance_matrix()
        self.time_matrix = self.create_time_matrix(speed=speed)

        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node, to_node])

        distance_callback_index = self.routing.RegisterTransitCallback(
            distance_callback
        )
        self.routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        self.routing.AddDimension(
            distance_callback_index,
            0,
            100_000,
            True,
            "Distance",
        )
        distance_dimension = self.routing.GetDimensionOrDie("Distance")
        distance_dimension.SetGlobalSpanCostCoefficient(1)

        def time_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)

            travel_time = int(self.time_matrix[from_node, to_node])
            node_service_time = 0 if from_node == depot else service_time
            return travel_time + node_service_time

        time_callback_index = self.routing.RegisterTransitCallback(time_callback)

        self.routing.AddDimension(
            time_callback_index,
            100_000,
            100_000,
            False,
            "Time",
        )
        time_dimension = self.routing.GetDimensionOrDie("Time")
        time_dimension.SetGlobalSpanCostCoefficient(1)

        time_windows = self.data["time_windows"]

        for node, (start, end) in enumerate(time_windows):
            if start > end:
                raise ValueError(f"Invalid time window at node {node}: {(start, end)}")

            if node == depot:
                continue

            index = self.manager.NodeToIndex(node)
            time_dimension.CumulVar(index).SetRange(start, end)

        depot_start, depot_end = time_windows[depot]
        for vehicle_id in range(num_vehicles):
            start_index = self.routing.Start(vehicle_id)
            end_index = self.routing.End(vehicle_id)
            time_dimension.CumulVar(start_index).SetRange(depot_start, depot_end)
            time_dimension.CumulVar(end_index).SetRange(depot_start, depot_end)

        for pickup_node, delivery_node in self.data["pickups_deliveries"]:
            pickup_index = self.manager.NodeToIndex(pickup_node)
            delivery_index = self.manager.NodeToIndex(delivery_node)

            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)

            self.routing.solver().Add(
                self.routing.VehicleVar(pickup_index)
                == self.routing.VehicleVar(delivery_index)
            )

            self.routing.solver().Add(
                time_dimension.CumulVar(pickup_index)
                <= time_dimension.CumulVar(delivery_index)
            )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds

        self.solution = self.routing.SolveWithParameters(search_parameters)
        return self.solution is not None

    def get_solution(self):
        """Extract the current solution."""
        if self.solution is None:
            raise RuntimeError("No solution is available. Call solve() first.")

        routes = []
        total_distance = 0
        max_route_distance = 0
        time_dimension = self.routing.GetDimensionOrDie("Time")

        for vehicle_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(vehicle_id)
            route = []
            route_times = []
            route_distance = 0

            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)

                route.append(node)
                route_times.append(
                    [self.solution.Min(time_var), self.solution.Max(time_var)]
                )

                next_index = self.solution.Value(self.routing.NextVar(index))
                next_node = self.manager.IndexToNode(next_index)
                route_distance += int(self.distance_matrix[node, next_node])
                index = next_index

            end_node = self.manager.IndexToNode(index)
            end_time_var = time_dimension.CumulVar(index)
            route.append(end_node)
            route_times.append(
                [
                    self.solution.Min(end_time_var),
                    self.solution.Max(end_time_var),
                ]
            )

            routes.append(
                {
                    "vehicle": vehicle_id,
                    "route": route,
                    "times": route_times,
                    "distance": route_distance,
                }
            )

            total_distance += route_distance
            max_route_distance = max(max_route_distance, route_distance)

        return {
            "total_distance": total_distance,
            "max_route_distance": max_route_distance,
            "routes": routes,
        }

    def visualize_solution(self, save_path=None):
        """Visualize the current solution with matplotlib."""
        if self.solution is None:
            raise RuntimeError("No solution is available. Call solve() first.")

        solution_data = self.get_solution()
        routes = solution_data["routes"]
        locations = self.data["locations"]
        depot = self.data["depot"]

        plt.figure(figsize=(12, 10))

        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        plt.scatter(x_coords, y_coords, c="gray", s=120, zorder=1)

        depot_x, depot_y = locations[depot]
        plt.scatter(
            depot_x,
            depot_y,
            c="red",
            s=200,
            marker="*",
            zorder=2,
        )
        plt.annotate(
            "Depot",
            (depot_x, depot_y),
            fontsize=12,
            xytext=(10, 10),
            textcoords="offset points",
        )

        for i, (x, y) in enumerate(locations):
            if i != depot:
                plt.annotate(str(i), (x, y), fontsize=10, ha="center", va="center")

        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(routes))))

        for i, route_data in enumerate(routes):
            route = route_data["route"]
            if len(route) < 2:
                continue

            color = colors[i]
            points = [locations[node] for node in route]

            for j in range(len(points) - 1):
                x1, y1 = points[j]
                x2, y2 = points[j + 1]
                plt.plot(
                    [x1, x2],
                    [y1, y2],
                    "o-",
                    c=color,
                    linewidth=2,
                    markersize=0,
                    zorder=0,
                    alpha=0.7,
                )

            if any(node != depot for node in route):
                x1, y1 = points[0]
                x2, y2 = points[1]
                plt.annotate(
                    f"Route {i + 1}",
                    ((x1 + x2) / 2, (y1 + y2) / 2),
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                    xytext=(10, 10),
                    textcoords="offset points",
                )

        plt.title("PDPTW Solution Visualization", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        legend_text = [
            f'Total Distance: {solution_data["total_distance"]}',
            f'Max Route Distance: {solution_data["max_route_distance"]}',
            f'Number of Vehicles: {len(routes)}',
        ]
        plt.figtext(
            0.02,
            0.02,
            "\n".join(legend_text),
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def create_test_data(num_requests=5, grid_size=100):
    """Create reproducible demo data."""
    if num_requests < 1:
        raise ValueError("num_requests must be at least 1.")

    rng = np.random.default_rng(42)
    num_locations = 1 + num_requests * 2

    locations = [(grid_size // 2, grid_size // 2)]
    for _ in range(num_locations - 1):
        locations.append(
            (
                int(rng.integers(0, grid_size)),
                int(rng.integers(0, grid_size)),
            )
        )

    pickups_deliveries = [
        (i * 2 + 1, i * 2 + 2)
        for i in range(num_requests)
    ]

    max_time = 480
    time_windows = [(0, max_time) for _ in range(num_locations)]

    return {
        "locations": locations,
        "num_vehicles": min(3, num_requests),
        "depot": 0,
        "pickups_deliveries": pickups_deliveries,
        "time_windows": time_windows,
    }


def main():
    data = create_test_data(num_requests=5, grid_size=100)
    solver = PDPTWSolver(data)

    if not solver.solve():
        print("No solution found!")
        return

    solution = solver.get_solution()
    print("Solution found!")
    print(f"Total distance: {solution['total_distance']}")
    print(f"Maximum route distance: {solution['max_route_distance']}")

    for route_data in solution["routes"]:
        vehicle_id = route_data["vehicle"]
        print(f"\nRoute for vehicle {vehicle_id + 1}:")

        route_info = []
        for node, times in zip(route_data["route"], route_data["times"]):
            if node == data["depot"]:
                location_type = "Depot"
            elif node % 2 == 1:
                location_type = f"Pickup {(node + 1) // 2}"
            else:
                location_type = f"Delivery {node // 2}"

            route_info.append(
                f"{node} ({location_type}) - Time: {times[0]}"
            )

        print(" -> ".join(route_info))
        print(f"Distance: {route_data['distance']}")

    solver.visualize_solution()


if __name__ == "__main__":
    main()
