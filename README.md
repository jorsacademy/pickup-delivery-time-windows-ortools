# Pickup and Delivery with Time Windows (PDPTW) — OR-Tools

A Python implementation of the **Pickup and Delivery Problem with Time Windows (PDPTW)** using Google OR-Tools.

The solver models multiple vehicles that must serve pickup-delivery request pairs while respecting route order and time-window constraints. It also provides a simple matplotlib visualization of the resulting routes.

## Features

- Multiple vehicle routing
- Pickup and delivery pairing
- Same-vehicle constraint for each pickup-delivery pair
- Pickup-before-delivery precedence
- Time-window constraints
- Service time at customer locations
- Depot start/end time windows
- Euclidean distance matrix
- Route distance statistics
- Matplotlib route visualization
- Reproducible demo dataset

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- Google OR-Tools

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python pdptw_solver.py
```

The script creates a small demo instance, solves it using OR-Tools, prints each vehicle route and arrival time, and displays the route visualization.

## Problem Data

The solver expects a dictionary with the following structure:

```python
data = {
    "locations": [(50, 50), (10, 20), (30, 40)],
    "num_vehicles": 2,
    "depot": 0,
    "pickups_deliveries": [(1, 2)],
    "time_windows": [(0, 480), (0, 240), (60, 360)],
}
```

### Fields

- `locations`: `(x, y)` coordinates for the depot and service nodes.
- `num_vehicles`: Number of available vehicles.
- `depot`: Index of the depot node.
- `pickups_deliveries`: `(pickup_node, delivery_node)` request pairs.
- `time_windows`: `(earliest, latest)` service window for each node.

## Solver Model

The implementation uses:

- `RoutingIndexManager`
- `RoutingModel`
- a distance dimension
- a time dimension
- `AddPickupAndDelivery`
- vehicle equality constraints for pickup-delivery pairs
- time cumul precedence constraints
- `PARALLEL_CHEAPEST_INSERTION` for the initial solution
- `GUIDED_LOCAL_SEARCH` for local improvement

## Notes

The included demo uses Euclidean coordinates and broad time windows for reproducibility. For real-world applications, replace the generated data with actual road-network distances, travel times, service durations, capacities, and operational time windows.

## License

No license has been added yet.
