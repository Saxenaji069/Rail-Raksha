import numpy as np
import heapq
import random
from typing import List, Tuple, Dict, Set, Optional
import plotly.graph_objects as go
import streamlit as st

class StationGrid:
    """
    Represents a 2D grid of a railway station with crowd density information.
    """
    def __init__(self, width: int, height: int, areas: Dict[str, List[Tuple[int, int]]]):
        """
        Initialize the station grid.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            areas: Dictionary mapping area names to lists of (x, y) coordinates
        """
        self.width = width
        self.height = height
        self.areas = areas
        self.grid = np.zeros((height, width), dtype=float)
        self.crowd_density = np.zeros((height, width), dtype=float)
        self.area_names = np.full((height, width), "", dtype=object)
        
        # Initialize area names
        for area_name, coords in areas.items():
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    self.area_names[y, x] = area_name
    
    def update_crowd_density(self, area_densities: Dict[str, float]):
        """
        Update crowd density for each area.
        
        Args:
            area_densities: Dictionary mapping area names to density values (0-1)
        """
        for area_name, density in area_densities.items():
            if area_name in self.areas:
                for x, y in self.areas[area_name]:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.crowd_density[y, x] = density
    
    def get_crowd_density(self, x: int, y: int) -> float:
        """Get crowd density at a specific position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.crowd_density[y, x]
        return 1.0  # Return high density for out-of-bounds
    
    def get_area_name(self, x: int, y: int) -> str:
        """Get area name at a specific position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.area_names[y, x]
        return "unknown"
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if a position is valid within the grid."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-way movement
        neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors


class PathFinder:
    """
    Implements pathfinding algorithms for finding optimal routes through the station.
    """
    def __init__(self, station_grid: StationGrid):
        """
        Initialize the pathfinder with a station grid.
        
        Args:
            station_grid: StationGrid instance
        """
        self.station_grid = station_grid
    
    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find the optimal path from start to goal using Dijkstra's algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions forming the path from start to goal
        """
        # Initialize data structures
        distances = {start: 0}
        queue = [(0, start)]
        heapq.heapify(queue)
        came_from = {start: None}
        visited = set()
        
        while queue:
            current_distance, current = heapq.heappop(queue)
            
            if current == goal:
                break
                
            if current in visited:
                continue
                
            visited.add(current)
            
            # Explore neighbors
            for next_pos in self.station_grid.get_neighbors(*current):
                if next_pos in visited:
                    continue
                    
                # Calculate edge weight based on crowd density
                # Higher crowd density = higher weight
                crowd_weight = 1.0 + 5.0 * self.station_grid.get_crowd_density(*next_pos)
                distance = current_distance + crowd_weight
                
                if next_pos not in distances or distance < distances[next_pos]:
                    distances[next_pos] = distance
                    came_from[next_pos] = current
                    heapq.heappush(queue, (distance, next_pos))
        
        # Reconstruct path
        if goal not in came_from:
            return []  # No path found
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find the optimal path from start to goal using A* algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of positions forming the path from start to goal
        """
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            """Manhattan distance heuristic."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Initialize data structures
        open_set = {start}
        closed_set = set()
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        queue = [(f_score[start], start)]
        heapq.heapify(queue)
        
        while open_set:
            current_f, current = heapq.heappop(queue)
            
            if current == goal:
                break
            
            # Safely remove from open_set
            if current in open_set:
                open_set.remove(current)
            else:
                print(f"⚠️ Tried removing {current} but it was not in open_set.")
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for next_pos in self.station_grid.get_neighbors(*current):
                if next_pos in closed_set:
                    continue
                    
                # Calculate edge weight based on crowd density
                crowd_weight = 1.0 + 5.0 * self.station_grid.get_crowd_density(*next_pos)
                tentative_g_score = g_score[current] + crowd_weight
                
                if next_pos not in open_set:
                    open_set.add(next_pos)
                elif tentative_g_score >= g_score.get(next_pos, float('inf')):
                    continue
                    
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g_score
                f_score[next_pos] = g_score[next_pos] + heuristic(next_pos, goal)
                heapq.heappush(queue, (f_score[next_pos], next_pos))
        
        # Reconstruct path
        if goal not in came_from:
            return []  # No path found
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def find_safe_route(self, start: Tuple[int, int], algorithm: str = "a_star") -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        """
        Find a safe route from the current position to the nearest less crowded area.
        
        Args:
            start: Starting position (x, y)
            algorithm: Pathfinding algorithm to use ("dijkstra" or "a_star")
            
        Returns:
            Tuple of (path, destination)
        """
        # Find all positions with low crowd density
        safe_positions = []
        for y in range(self.station_grid.height):
            for x in range(self.station_grid.width):
                if self.station_grid.get_crowd_density(x, y) < 0.5:  # Threshold for "safe"
                    safe_positions.append((x, y))
        
        if not safe_positions:
            return [], start  # No safe positions found
        
        # Find the closest safe position
        best_path = []
        best_destination = start
        min_path_length = float('inf')
        
        for goal in safe_positions:
            if algorithm == "dijkstra":
                path = self.dijkstra(start, goal)
            else:  # a_star
                path = self.a_star(start, goal)
                
            if path and len(path) < min_path_length:
                min_path_length = len(path)
                best_path = path
                best_destination = goal
        
        return best_path, best_destination


class StationVisualizer:
    """
    Visualizes the station grid and paths using Plotly.
    """
    def __init__(self, station_grid: StationGrid):
        """
        Initialize the visualizer with a station grid.
        
        Args:
            station_grid: StationGrid instance
        """
        self.station_grid = station_grid
    
    def create_heatmap(self, path: Optional[List[Tuple[int, int]]] = None, 
                      user_pos: Optional[Tuple[int, int]] = None,
                      safe_pos: Optional[Tuple[int, int]] = None) -> go.Figure:
        """
        Create a heatmap visualization of the station with optional path and markers.
        
        Args:
            path: List of positions forming a path
            user_pos: User's current position
            safe_pos: Safe destination position
            
        Returns:
            Plotly figure
        """
        # Create heatmap data
        z = self.station_grid.crowd_density
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale=[
                [0, 'rgb(0, 255, 0)'],      # Green (low density)
                [0.5, 'rgb(255, 255, 0)'],  # Yellow (medium density)
                [1, 'rgb(255, 0, 0)']       # Red (high density)
            ],
            showscale=True,
            colorbar=dict(
                title='Crowd Density',
                title_side='right'
            )
        ))
        
        # Add area labels
        for y in range(self.station_grid.height):
            for x in range(self.station_grid.width):
                area_name = self.station_grid.get_area_name(x, y)
                if area_name:
                    fig.add_annotation(
                        x=x,
                        y=y,
                        text=area_name,
                        showarrow=False,
                        font=dict(color='white', size=10)
                    )
        
        # Add path if provided
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            fig.add_trace(go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Safe Route'
            ))
        
        # Add user position if provided
        if user_pos:
            fig.add_trace(go.Scatter(
                x=[user_pos[0]],
                y=[user_pos[1]],
                mode='markers',
                marker=dict(color='white', size=15, symbol='star'),
                name='Your Location'
            ))
        
        # Add safe position if provided
        if safe_pos:
            fig.add_trace(go.Scatter(
                x=[safe_pos[0]],
                y=[safe_pos[1]],
                mode='markers',
                marker=dict(color='green', size=15, symbol='diamond'),
                name='Safe Zone'
            ))
        
        # Update layout
        fig.update_layout(
            title='Railway Station Crowd Density Map',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        return fig


def create_sample_station() -> StationGrid:
    """
    Create a sample station grid for testing.
    
    Returns:
        StationGrid instance
    """
    # Define station areas
    areas = {
        "Entrance": [(0, 0), (1, 0), (2, 0)],
        "Ticket Counter": [(3, 0), (4, 0), (5, 0)],
        "Waiting Area": [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        "Platform 1": [(0, 2), (1, 2), (2, 2)],
        "Platform 2": [(3, 2), (4, 2), (5, 2)],
        "Corridor": [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3)],
        "Restrooms": [(0, 4), (1, 4)],
        "Food Court": [(2, 4), (3, 4)],
        "Exit": [(4, 4), (5, 4)]
    }
    
    # Create station grid
    station = StationGrid(width=6, height=5, areas=areas)
    
    # Set initial crowd densities
    station.update_crowd_density({
        "Entrance": 0.7,
        "Ticket Counter": 0.8,
        "Waiting Area": 0.9,
        "Platform 1": 0.6,
        "Platform 2": 0.4,
        "Corridor": 0.5,
        "Restrooms": 0.3,
        "Food Court": 0.6,
        "Exit": 0.4
    })
    
    return station


def simulate_crowd_updates(station: StationGrid, update_interval: float = 2.0) -> None:
    """
    Simulate real-time crowd density updates.
    
    Args:
        station: StationGrid instance
        update_interval: Time between updates in seconds
    """
    # Create a placeholder for the heatmap
    heatmap_placeholder = st.empty()
    
    # Create a placeholder for the path
    path_placeholder = st.empty()
    
    # Create a placeholder for the user position
    user_pos_placeholder = st.empty()
    
    # Create a placeholder for the safe position
    safe_pos_placeholder = st.empty()
    
    # Create a placeholder for the crowd density info
    info_placeholder = st.empty()
    
    # Initialize pathfinder and visualizer
    pathfinder = PathFinder(station)
    visualizer = StationVisualizer(station)
    
    # Initialize user position (can be randomly assigned or fixed)
    user_pos = (2, 2)  # Platform 1
    
    # Main update loop
    while True:
        # Update crowd densities with random fluctuations
        new_densities = {}
        for area_name in station.areas.keys():
            current_density = station.crowd_density[station.areas[area_name][0][1], station.areas[area_name][0][0]]
            # Add random fluctuation between -0.1 and 0.1
            fluctuation = random.uniform(-0.1, 0.1)
            new_density = max(0.0, min(1.0, current_density + fluctuation))
            new_densities[area_name] = new_density
        
        station.update_crowd_density(new_densities)
        
        # Find safe route
        path, safe_pos = pathfinder.find_safe_route(user_pos)
        
        # Create heatmap
        fig = visualizer.create_heatmap(path, user_pos, safe_pos)
        
        # Display heatmap
        heatmap_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Display path information
        if path:
            path_placeholder.markdown(f"**Safe Route Found:** {len(path)} steps to safety")
        else:
            path_placeholder.markdown("**No safe route found**")
        
        # Display user position
        user_area = station.get_area_name(*user_pos)
        user_density = station.get_crowd_density(*user_pos)
        user_pos_placeholder.markdown(f"**Your Location:** {user_area} (Density: {user_density:.2f})")
        
        # Display safe position
        if safe_pos != user_pos:
            safe_area = station.get_area_name(*safe_pos)
            safe_density = station.get_crowd_density(*safe_pos)
            safe_pos_placeholder.markdown(f"**Safe Zone:** {safe_area} (Density: {safe_density:.2f})")
        else:
            safe_pos_placeholder.markdown("**You are already in a safe zone**")
        
        # Display crowd density information
        info_placeholder.markdown("""
        **Crowd Density Legend:**
        - 🟢 Green: Low crowd density (0-0.5)
        - 🟡 Yellow: Medium crowd density (0.5-0.7)
        - 🔴 Red: High crowd density (0.7-1.0)
        """)
        
        # Wait for the next update
        time.sleep(update_interval) 