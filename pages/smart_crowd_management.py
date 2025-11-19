import streamlit as st
import time
import random
import numpy as np
import plotly.graph_objects as go
from utils.crowd_pathfinding import (
    StationGrid, 
    PathFinder, 
    StationVisualizer, 
    create_sample_station, 
    simulate_crowd_updates
)

# Set page config
st.set_page_config(
    page_title="Smart Crowd Management - Rail Raksha",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1d4ed8;
        --accent-color: #60a5fa;
        --text-color: #f8fafc;
        --dark-bg: #0f172a;
        --darker-bg: #020617;
        --hover-color: #3b82f6;
        --button-gradient: linear-gradient(135deg, #2563eb, #60a5fa);
        --card-bg: rgba(255, 255, 255, 0.03);
        --sidebar-gradient: linear-gradient(135deg, rgba(37, 99, 235, 0.9), rgba(29, 78, 216, 0.95));
        --menu-hover: rgba(255, 255, 255, 0.08);
        --menu-active: rgba(255, 255, 255, 0.12);
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }

    /* Main content styling */
    .main {
        background: var(--dark-bg) !important;
        color: var(--text-color) !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Configuration section styling */
    .config-section {
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(12px) !important;
    }

    .config-section:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        margin-bottom: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        letter-spacing: -0.025em !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: var(--text-color) !important;
        text-align: center !important;
        margin-bottom: 2.5rem !important;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.5) !important;
        background: linear-gradient(135deg, #f8fafc, #60a5fa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: -0.05em !important;
    }

    /* Button styling */
    .stButton > button {
        background: var(--button-gradient) !important;
        color: var(--text-color) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }

    /* Info text styling */
    .info-text {
        color: var(--text-color) !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        opacity: 0.9 !important;
    }

    /* Highlight box styling */
    .highlight {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 1rem 0 !important;
    }

    /* Status colors */
    .success-text { color: var(--success-color) !important; }
    .warning-text { color: var(--warning-color) !important; }
    .error-text { color: var(--error-color) !important; }

    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the Smart Crowd Management page."""
    st.markdown('<h1 class="main-header">🚂 Smart Crowd Management</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    ## 👋 Welcome to Smart Crowd Management
    
    This feature helps you navigate through crowded areas of the railway station by finding the optimal path 
    to less crowded areas. The system uses advanced pathfinding algorithms to minimize exposure to high-density crowds.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Live Crowd Map", "Pathfinding Demo"])
    
    with tab1:
        st.markdown('<h2 class="section-header">🗺️ Live Crowd Density Map</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.write("""
        The map below shows real-time crowd density across different areas of the station. 
        Areas are color-coded based on crowd density:
        - 🟢 Green: Low crowd density (0-0.5)
        - 🟡 Yellow: Medium crowd density (0.5-0.7)
        - 🔴 Red: High crowd density (0.7-1.0)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a sample station
        station = create_sample_station()
        
        # Create a placeholder for the heatmap
        heatmap_placeholder = st.empty()
        
        # Create a placeholder for the crowd density info
        info_placeholder = st.empty()
        
        # Initialize visualizer
        visualizer = StationVisualizer(station)
        
        # Create initial heatmap
        fig = visualizer.create_heatmap()
        
        # Display heatmap
        heatmap_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Display crowd density information
        info_placeholder.markdown("""
        **Crowd Density Legend:**
        - 🟢 Green: Low crowd density (0-0.5)
        - 🟡 Yellow: Medium crowd density (0.5-0.7)
        - 🔴 Red: High crowd density (0.7-1.0)
        """)
        
        # Add a button to simulate crowd updates
        if st.button("Simulate Crowd Updates"):
            with st.spinner("Updating crowd densities..."):
                # Update crowd densities with random fluctuations
                new_densities = {}
                for area_name in station.areas.keys():
                    current_density = station.crowd_density[station.areas[area_name][0][1], station.areas[area_name][0][0]]
                    # Add random fluctuation between -0.1 and 0.1
                    fluctuation = random.uniform(-0.1, 0.1)
                    new_density = max(0.0, min(1.0, current_density + fluctuation))
                    new_densities[area_name] = new_density
                
                station.update_crowd_density(new_densities)
                
                # Create updated heatmap
                fig = visualizer.create_heatmap()
                
                # Display updated heatmap
                heatmap_placeholder.plotly_chart(fig, use_container_width=True)
                
                st.success("Crowd densities updated successfully!")
    
    with tab2:
        st.markdown('<h2 class="section-header">🛣️ Pathfinding Demo</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.write("""
        This demo shows how the system finds the optimal path from a crowded area to a less crowded area.
        The path is calculated using the A* algorithm, which minimizes exposure to high-density crowds.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a sample station
        station = create_sample_station()
        
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
        
        # Allow user to select their position
        st.markdown('<h3 class="section-header">📍 Select Your Position</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_pos = st.slider("X Position", 0, station.width - 1, 2)
        
        with col2:
            y_pos = st.slider("Y Position", 0, station.height - 1, 2)
        
        user_pos = (x_pos, y_pos)
        
        # Allow user to select the algorithm
        algorithm = st.radio("Select Pathfinding Algorithm", ["A*", "Dijkstra"], index=0)
        algorithm = "a_star" if algorithm == "A*" else "dijkstra"
        
        # Find safe route
        path, safe_pos = pathfinder.find_safe_route(user_pos, algorithm)
        
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
        
        # Add a button to simulate crowd updates
        if st.button("Simulate Crowd Updates and Recalculate Path"):
            with st.spinner("Updating crowd densities and recalculating path..."):
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
                path, safe_pos = pathfinder.find_safe_route(user_pos, algorithm)
                
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
                
                st.success("Crowd densities updated and path recalculated successfully!")

if __name__ == "__main__":
    main() 