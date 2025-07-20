import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_lane_boundaries(locations, rotations, lane_widths):
    """Calculate left and right lane boundaries"""
    left_boundary = []
    right_boundary = []
    
    for i in range(len(locations)):
        # Get waypoint data
        location = locations[i]
        rotation = rotations[i]
        lane_width = lane_widths[i]
        
        # Calculate perpendicular vector (90 degrees from rotation)
        # Assuming rotation is in radians and represents yaw
        yaw = rotation[2] if len(rotation) >= 3 else rotation
        perp_x = -np.sin(yaw)
        perp_y = np.cos(yaw)
        
        # Calculate left and right boundary points
        half_width = lane_width / 2
        left_point = location[:2] + np.array([perp_x, perp_y]) * half_width
        right_point = location[:2] - np.array([perp_x, perp_y]) * half_width
        
        left_boundary.append(left_point)
        right_boundary.append(right_point)
    
    return np.array(left_boundary), np.array(right_boundary)

def plot_waypoints_comparison():
    """Plot waypointsPrimary vs waypointsFinal for comparison"""
    
    # Load original racing line waypoints
    print("Loading original racing line waypoints...")
    original_waypoints = np.load(f"{os.path.dirname(__file__)}/waypoints/waypointsPrimary.npz")
    
    # Load optimized racing line waypoints
    print("Loading optimized racing line waypoints...")
    optimized_waypoints = np.load(f"{os.path.dirname(__file__)}/waypoints/waypointsFinal.npz")
    
    # Load original track waypoints for lane lines
    print("Loading original track waypoints...")
    try:
        track_waypoints = np.load(f"{os.path.dirname(__file__)}/waypoints/Monza Original Waypoints.npz")
        print(f"Available keys in track waypoint file: {list(track_waypoints.keys())}")
    except FileNotFoundError:
        print("Monza Original Waypoints.npz not found, plotting only racing lines")
        track_waypoints = None
    
    # Load original racing line data
    if 'locations' in original_waypoints and 'rotations' in original_waypoints and 'lane_widths' in original_waypoints:
        original_locations = original_waypoints['locations'][35:]  # Skip first 35 waypoints
        original_rotations = original_waypoints['rotations'][35:]
        original_lane_widths = original_waypoints['lane_widths'][35:]
    else:
        print("Could not find required data in original waypoint file")
        return
    
    # Load optimized racing line data
    if 'locations' in optimized_waypoints and 'rotations' in optimized_waypoints and 'lane_widths' in optimized_waypoints:
        optimized_locations = optimized_waypoints['locations'][35:]  # Skip first 35 waypoints
        optimized_rotations = optimized_waypoints['rotations'][35:]
        optimized_lane_widths = optimized_waypoints['lane_widths'][35:]
    else:
        print("Could not find required data in optimized waypoint file")
        return
    
    print(f"Loaded {len(original_locations)} original racing waypoints")
    print(f"Loaded {len(optimized_locations)} optimized racing waypoints")
    
    # Load track data if available
    track_locations = None
    track_rotations = None
    track_lane_widths = None
    
    if track_waypoints is not None:
        if all(key in track_waypoints for key in ['locations', 'rotations', 'lane_widths']):
            track_locations = track_waypoints['locations']
            track_rotations = track_waypoints['rotations']
            track_lane_widths = track_waypoints['lane_widths']
            print(f"Loaded {len(track_locations)} track waypoints")
        else:
            print("Could not find required data in track waypoint file")
    
    # Calculate lane boundaries
    print("Calculating lane boundaries...")
    if track_locations is not None:
        left_boundary, right_boundary = calculate_lane_boundaries(
            track_locations, track_rotations, track_lane_widths
        )
    
    # Extract coordinates
    original_x = original_locations[:, 0]
    original_y = original_locations[:, 1]
    optimized_x = optimized_locations[:, 0]
    optimized_y = optimized_locations[:, 1]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Track overview with both racing lines
    if track_locations is not None:
        # Plot centerline
        track_x = track_locations[:, 0]
        track_y = track_locations[:, 1]
        ax1.plot(track_x, track_y, 'k-', linewidth=2, alpha=0.5, label='Track Centerline')
        
        # Plot lane boundaries
        ax1.plot(left_boundary[:, 0], left_boundary[:, 1], 'k-', linewidth=3, alpha=0.8, label='Lane Boundaries')
        ax1.plot(right_boundary[:, 0], right_boundary[:, 1], 'k-', linewidth=3, alpha=0.8)
        
        # Fill track area
        ax1.fill_between(left_boundary[:, 0], left_boundary[:, 1], right_boundary[:, 1], 
                        alpha=0.1, color='gray', label='Track Area')
    
    # Plot racing lines
    ax1.plot(original_x, original_y, 'b-', linewidth=4, alpha=0.8, label='Original Racing Line')
    ax1.plot(optimized_x, optimized_y, 'r-', linewidth=3, alpha=0.8, label='Optimized Racing Line')
    
    # Mark start point
    ax1.plot(original_x[0], original_y[0], 'go', markersize=15, label='Start Point')
    
    # Mark every 500th waypoint for reference (original)
    for i in range(0, len(original_locations), 500):
        ax1.plot(original_x[i], original_y[i], 'bo', markersize=8, alpha=0.7)
        ax1.annotate(f'{i+35}', (original_x[i], original_y[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Mark every 1500th waypoint for reference (optimized - more dense)
    for i in range(0, len(optimized_locations), 1500):
        ax1.plot(optimized_x[i], optimized_y[i], 'ro', markersize=6, alpha=0.7)
        ax1.annotate(f'{i+35}', (optimized_x[i], optimized_y[i]), 
                    xytext=(5, -15), textcoords='offset points', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Mark section boundaries
    section_locations = [
        [-278, 372],   # Section 0
        [64, 890],     # Section 1
        [511, 1037],   # Section 2
        [762, 908],    # Section 3
        [198, 307],    # Section 4
        [-11, 60],     # Section 5
        [-85, -339],   # Section 6
        [-210, -1060], # Section 7
        [-318, -991],  # Section 8
        [-352, -119],  # Section 9
    ]
    
    for i, (x, y) in enumerate(section_locations):
        ax1.plot(x, y, 's', color='orange', markersize=12, 
                label=f'Section {i}' if i == 0 else "")
        ax1.annotate(f'S{i}', (x, y), xytext=(10, 10), textcoords='offset points', 
                    fontsize=14, fontweight='bold', color='orange',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    ax1.set_title('Racing Line Comparison: Original vs Optimized', fontsize=16)
    ax1.set_xlabel('X Position (meters)', fontsize=12)
    ax1.set_ylabel('Y Position (meters)', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Waypoint density comparison
    # Calculate waypoint spacing for both lines
    original_spacing = []
    for i in range(1, len(original_locations)):
        dist = np.linalg.norm(original_locations[i][:2] - original_locations[i-1][:2])
        original_spacing.append(dist)
    
    optimized_spacing = []
    for i in range(1, len(optimized_locations)):
        dist = np.linalg.norm(optimized_locations[i][:2] - optimized_locations[i-1][:2])
        optimized_spacing.append(dist)
    
    # Plot spacing histograms
    ax2.hist(original_spacing, bins=50, alpha=0.7, color='blue', edgecolor='black', 
             label=f'Original ({len(original_locations)} waypoints)', density=True)
    ax2.hist(optimized_spacing, bins=50, alpha=0.7, color='red', edgecolor='black', 
             label=f'Optimized ({len(optimized_locations)} waypoints)', density=True)
    
    ax2.set_title('Waypoint Spacing Distribution', fontsize=16)
    ax2.set_xlabel('Distance Between Waypoints (meters)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Statistics:
Original: {len(original_locations)} waypoints
Optimized: {len(optimized_locations)} waypoints
Density Increase: {len(optimized_locations)/len(original_locations):.1f}x

Original Avg Spacing: {np.mean(original_spacing):.2f}m
Optimized Avg Spacing: {np.mean(optimized_spacing):.2f}m
Spacing Ratio: {np.mean(original_spacing)/np.mean(optimized_spacing):.1f}x denser"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    print(f"Original waypoints: {len(original_locations)}")
    print(f"Optimized waypoints: {len(optimized_locations)}")
    print(f"Density increase: {len(optimized_locations)/len(original_locations):.1f}x")
    print(f"Original average spacing: {np.mean(original_spacing):.2f} meters")
    print(f"Optimized average spacing: {np.mean(optimized_spacing):.2f} meters")
    print(f"Spacing ratio: {np.mean(original_spacing)/np.mean(optimized_spacing):.1f}x denser")
    
    # Calculate track lengths
    original_length = sum(original_spacing)
    optimized_length = sum(optimized_spacing)
    print(f"Original racing line length: {original_length:.1f} meters")
    print(f"Optimized racing line length: {optimized_length:.1f} meters")
    print(f"Length difference: {abs(optimized_length - original_length):.1f} meters")
    
    # Track bounds
    print(f"\nTrack Bounds:")
    print(f"Original: X [{original_x.min():.1f}, {original_x.max():.1f}], Y [{original_y.min():.1f}, {original_y.max():.1f}]")
    print(f"Optimized: X [{optimized_x.min():.1f}, {optimized_x.max():.1f}], Y [{optimized_y.min():.1f}, {optimized_y.max():.1f}]")

if __name__ == "__main__":
    plot_waypoints_comparison() 