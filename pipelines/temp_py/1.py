from scipy.spatial import KDTree
import numpy as np
import math
import matplotlib.pyplot as plt

def angle_between(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = math.atan2(delta_y, delta_x)
    return angle

def is_within_angle(point, center, direction_point, angle_range):
    angle_to_point = angle_between(center, point)
    angle_to_direction = angle_between(center, direction_point)
    lower_bound = angle_to_direction - angle_range / 2
    upper_bound = angle_to_direction + angle_range / 2
    
    # Normalize angles between -pi and pi
    angle_to_point = (angle_to_point + 2 * math.pi) % (2 * math.pi)
    lower_bound = (lower_bound + 2 * math.pi) % (2 * math.pi)
    upper_bound = (upper_bound + 2 * math.pi) % (2 * math.pi)

    if lower_bound < upper_bound:
        return lower_bound <= angle_to_point <= upper_bound
    else:  # Covers the case where the angle range crosses the -pi/pi boundary
        return angle_to_point >= lower_bound or angle_to_point <= upper_bound

def query_kdtree_within_angle(points, query_point, radius, direction_point, angle_range):
    kdtree = KDTree(points)
    outer_indices = kdtree.query_ball_point(query_point, radius)
    
    if not outer_indices:
        return np.array([])

    annulus_indices = [i for i in outer_indices if is_within_angle(points[i], query_point, direction_point, angle_range)]
    
    if not annulus_indices:
        return np.array([])
    
    return points[annulus_indices]

def simple_locate_segment(segment):
    """Assumes start and end locations of segment are identical."""
    if len(segment) == 0:
        raise ValueError("No legs in segment.")
    elif len(segment) == 1:
        assert segment[0]['from_location'].size > 0 and segment[0]['to_location'].size > 0, "Both start and end locations must be known for a single leg."
        return segment
    # elif len(segment) == 2:
    #     # Handle case with 2 legs
    #     results1 = locate_leg(segment[0])
    #     results2 = locate_leg(segment[1])
    #     return results1 + results2
    # else:
    #     # Find total distance
    #     total_distance = sum([leg['distance'] for leg in segment])
    #     # Process each leg
        located_legs = [locate_leg(leg) for leg in segment]
        return located_legs

def locate_leg(leg):
    points = leg['points']
    query_point = leg['query_point']
    radius = leg['radius']
    direction_point = leg['direction_point']
    angle_range = leg['angle_range']

    located_points = query_kdtree_within_angle(points, query_point, radius, direction_point, angle_range)
    return located_points

def plot_results(segment, located_segment):
    fig, ax = plt.subplots()

    for leg in segment:
        points = leg['points']
        ax.scatter(points[:, 0], points[:, 1], label='Original Points')

    for located_points in located_segment:
        if len(located_points) > 0:
            ax.scatter(located_points[:, 0], located_points[:, 1], label='Located Points', marker='X')

            query_point = leg['query_point']
            ax.plot(query_point[0], query_point[1], 'ro')  # Plot the query point
            for point in located_points:
                ax.plot([query_point[0], point[0]], [query_point[1], point[1]], 'grey', linestyle='--')  # Connections

    ax.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trip Plot')
    plt.show()

# Example usage
segment = [
    {
        'points': np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]),
        'query_point': [5, 5],
        'radius': 5.0,
        'direction_point': [7, 5],  # A point that defines the direction
        'angle_range': math.pi / 4,  # 45 degrees
        'from_location': np.array([2, 3]),
        'to_location': np.array([9, 6]),
        'distance': 5.0
    },
    {
        'points': np.array([[1, 2], [3, 4], [6, 7], [2, 2], [5, 5], [4, 3]]),
        'query_point': [3, 3],
        'radius': 4.0,
        'direction_point': [5, 4],  # Another direction point
        'angle_range': math.pi / 6,  # 30 degrees
        'from_location': np.array([1, 2]),
        'to_location': np.array([6, 7]),
        'distance': 4.0
    }
]

located_segment = simple_locate_segment(segment)

plot_results(segment, located_segment)
