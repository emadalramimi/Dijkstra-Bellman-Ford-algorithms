# S.A.E Univesity Project : Algorithmic Exploration of a Problem

## Team Members:
- AL Ramimi Emad
- Miled Willem
- El Kaddouri Amine
- Weter Liam

## Overview
This project focuses on the exploration of algorithmic solutions for finding the shortest paths within weighted graphs. It encompasses the implementation and comparison of notable algorithms such as Dijkstra and Bellman-Ford, alongside functionalities for graph visualization, random graph matrix generation, and connectivity testing.

## Features
- Implementation of the Dijkstra and Bellman-Ford algorithms for shortest path finding.
- Functionality to draw graphs and paths from matrix representations using NetworkX.
- Capability to generate random matrices for weighted graphs with customizable edge proportions.
- Analysis of the efficiency and effectiveness of pathfinding algorithms under various conditions.
- Tests for strong connectivity within graphs.
- Determination and visualization of the threshold for strong connectivity in random graphs.

## Usage
1. **Algorithm Implementation**:
   - `dijkstra_algorithm()`: Finds the shortest paths from a start vertex to all other vertices in a weighted graph without negative weights.
   - `bellman_ford_algorithm()`: Capable of handling graphs with negative weights, this function finds the shortest paths from a single source vertex, detecting negative cycles.

2. **Graph Visualization**:
   - `draw_graph()`: Visualizes a graph with optional path highlighting.
   - `draw_path()`: Highlights the shortest path in the graph visualization.

3. **Random Graph Generation**:
   - `generate_random_graph_matrix()`: Generates a random matrix representation for a weighted graph, with customizable edge density.

4. **Connectivity Testing**:
   - `test_strong_connectivity()`: Checks if a graph is strongly connected, ensuring that there is a path from every vertex to every other vertex.

5. **Experimental Comparison**:
   - Functions to compare the computational complexity and execution time of the Dijkstra and Bellman-Ford algorithms across various graph sizes and conditions.

## Installation
Ensure Python 3.x is installed on your system. This project requires the NetworkX library for graph operations, which can be installed using pip:
```bash
pip install networkx
```
## Running the Program
Execute the main script SAE2.py from your terminal or command prompt:

```bash
python SAE2.py
```
Follow the on-screen instructions to select and run specific functionalities or algorithm comparisons.

## Contributing
This project welcomes contributions from students and educators alike. If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is open-source and available under the MIT License.
