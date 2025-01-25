import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner

        if node not in self.neighbors:

            self.neighbors.append(node)

        if self not in node.neighbors:

            node.neighbors.append(self)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes_dict[(r, c)] = Node((r, c))

    # 2) Link each node with valid neighbors in four directions (undirected)
    for (r, c), node in nodes_dict.items():
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  
            if (nr, nc) in nodes_dict:
                node.add_neighbor(nodes_dict[(nr, nc)])

    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)
    start_node = nodes_dict.get((0, 0))
    goal_node = nodes_dict.get((rows-1, cols-1)) 

    # TODO: Assign start_node and goal_node if they exist in nodes_dict

    return nodes_dict, start_node, goal_node

    # TODO: Implement the logic to build nodes and link neighbors


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS

    queue = deque([start_node])
    visited = set()
    parent_map = {}

    visited.add(start_node)


    while queue:
        current_node = queue.popleft()

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.value)
                current_node = parent_map.get(current_node)
            return path[::-1]

        for neighbor in current_node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current_node
                queue.append(neighbor)

    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    stack = [start_node]
    visited = set()    
    parent_map = {} 

    visited.add(start_node)

    while stack:
        current_node = stack.pop()

        if current_node == goal_node:

            path = []
            while current_node:
                path.append(current_node.value)
                current_node = parent_map.get(current_node)
            return path[::-1] 

        for neighbor in current_node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current_node 
                stack.append(neighbor)

    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*

    if not start_node or not goal_node:
        return None

    # Priority queue for A* (min-heap) - stores (f_score, node)
    open_set = []
    heapq.heappush(open_set, (0, start_node))

    # Dictionaries for g_score, f_score, and parent mapping
    g_score = defaultdict(lambda: float('inf'))  # Cost from start_node to each node
    f_score = defaultdict(lambda: float('inf'))  # Estimated total cost from start_node to goal_node
    g_score[start_node] = 0
    f_score[start_node] = manhattan_distance(start_node, goal_node)

    parent_map = {}  # To reconstruct the path

    while open_set:
        # Expand the node with the smallest f_score
        current_f, current_node = heapq.heappop(open_set)

        # Check if we reached the goal
        if current_node == goal_node:
            # Reconstruct the path
            path = []
            while current_node:
                path.append(current_node.value)
                current_node = parent_map.get(current_node)
            return path[::-1]  # Reverse to get path from start to goal

        # Explore neighbors
        for neighbor in current_node.neighbors:
            tentative_g_score = g_score[current_node] + 1  # Assume uniform cost for edges

            if tentative_g_score < g_score[neighbor]:
                # Better path found
                parent_map[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal_node)

                # Add neighbor to the priority queue if not already there
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # If we exhaust the open_set without finding the goal, return None

    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|

    r1, c1 = node_a.value
    r2, c2 = node_b.value
    return abs(r1 - r2) + abs(c1 - c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

from collections import deque

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    if not start_node or not goal_node:
        return None

    # Two queues and visited sets for forward and backward search
    forward_queue = deque([start_node])
    backward_queue = deque([goal_node])
    forward_visited = {start_node: None}
    backward_visited = {goal_node: None}

    while forward_queue and backward_queue:
        # Expand from start side
        if expand(forward_queue, forward_visited, backward_visited):
            return reconstruct_bidirectional_path(forward_visited, backward_visited)

        # Expand from goal side
        if expand(backward_queue, backward_visited, forward_visited):
            return reconstruct_bidirectional_path(forward_visited, backward_visited)

    return None

def expand(frontier, visited, other_visited):
    """
    Expands a frontier and checks if a meeting node is found.
    """
    if not frontier:
        return False

    current = frontier.popleft()

    for neighbor in current.neighbors:
        if neighbor not in visited:
            visited[neighbor] = current  # Track parent
            frontier.append(neighbor)

        # Check if the neighbor is in the other visited set (i.e., frontiers meet)
        if neighbor in other_visited:
            return True  # Frontiers meet

    return False

def reconstruct_bidirectional_path(forward_visited, backward_visited):
    """
    Reconstruct the path when the frontiers meet.
    """
    # Find the meeting point
    meeting_node = next(node for node in forward_visited if node in backward_visited)

    # Reconstruct forward path
    path = []
    current = meeting_node
    while current:
        path.append(current.value)  # Assuming nodes have a `value` attribute (like (row, col))
        current = forward_visited[current]
    path.reverse()

    # Reconstruct backward path
    current = backward_visited[meeting_node]
    while current:
        path.append(current.value)
        current = backward_visited[current]

    return path



###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    current_node = start_node
    path = [current_node.value]

    while temperature > min_temperature:
        # If we've reached the goal, stop
        if current_node == goal_node:
            return path
        
        # Get neighbors
        neighbors = current_node.neighbors
        if not neighbors:
            break

        # Randomly choose a neighbor
        next_node = random.choice(neighbors)

        # Calculate the 'cost' of moving to the next node (Manhattan distance)
        current_cost = manhattan_distance(current_node, goal_node)
        next_cost = manhattan_distance(next_node, goal_node)

        # Accept the move with probability depending on the temperature
        if next_cost < current_cost or random.random() < math.exp((current_cost - next_cost) / temperature):
            current_node = next_node
            path.append(current_node.value)

        # Cool down the temperature
        temperature *= cooling_rate


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction

    path = []
    current_node = end_node

    while current_node is not None:
        path.append(current_node.value)
        current_node = parent_map.get(current_node)

    return path[::-1]
    return None

def parse_maze_to_graph(maze):
    rows, cols = maze.shape
    nodes_dict = {}

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes_dict[(r, c)] = Node((r, c))

    for (r, c), node in nodes_dict.items():
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if (nr, nc) in nodes_dict:
                node.add_neighbor(nodes_dict[(nr, nc)])

    start_node = nodes_dict.get((0, 0))
    goal_node = nodes_dict.get((rows-1, cols-1))

    return nodes_dict, start_node, goal_node



###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Test DFS
    path_dfs = dfs(start_node, goal_node)
    print("DFS Path:", path_dfs)

    # Test A*
    path_astar = astar(start_node, goal_node)
    print("A* Path:", path_astar)

    # Test Sim Annealing
    path_simulated_annealing = simulated_annealing(start_node, goal_node)
    print("Simulated Annealing Path:", path_simulated_annealing)

    # Test Bidirectional Search
    path_bidirectional_search = bidirectional_search(start_node, goal_node)
    print("Bidirectional Search Path:", path_bidirectional_search)


    # Step 1: Generate a random 5x5 maze (0 = open, 1 = blocked)
np.random.seed(42)  # For reproducibility
maze = np.random.choice([0, 1], size=(5, 5), p=[0.7, 0.3])  # 70% open, 30% blocked

# Make sure start (0,0) and goal (4,4) are open
maze[0][0] = 0
maze[4][4] = 0

print("Generated 5x5 Maze:")
print(maze)

# Step 2: Convert the maze to a graph using your function
nodes_dict, start_node, goal_node = parse_maze_to_graph(maze)

# Step 3: Perform BFS, DFS, or A* search to find a path from start_node to goal_node
if start_node and goal_node:
    print("\nFinding Path Using BFS:")
    bfs_path = bfs(start_node, goal_node)
    print(bfs_path)

    print("\nFinding Path Using DFS:")
    dfs_path = dfs(start_node, goal_node)
    print(dfs_path)

    print("\nFinding Path Using A*:")
    astar_path = astar(start_node, goal_node)
    print(astar_path)
else:
    print("Start or goal is blocked, no path possible.")