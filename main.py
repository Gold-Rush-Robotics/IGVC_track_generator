import pygame, sys
from pygame.locals import *
import math
import random as rn
import numpy as np
import scipy as sc
from scipy.spatial import ConvexHull
from scipy import interpolate
import argparse
import svgwrite
from shapely.geometry import LineString, Point
import subprocess
import os

from constants import *

####
## logical functions
####
def random_points(min=MIN_POINTS, max=MAX_POINTS, margin=MARGIN, min_distance=MIN_DISTANCE):
    pointCount = rn.randrange(min, max+1, 1)
    points = []
    for i in range(pointCount):
        x = rn.randrange(margin, WIDTH - margin + 1, 1)
        y = rn.randrange(margin, HEIGHT -margin + 1, 1)
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def get_track_points(hull, points):
    # get the original points from the random 
    # set that will be used as the track starting shape
    return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

def make_rand_vector(dims):
    vec = [rn.gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def shape_track(track_points, difficulty=DIFFICULTY, max_displacement=MAX_DISPLACEMENT, margin=MARGIN):
    track_set = [[0,0] for i in range(len(track_points)*2)] 
    for i in range(len(track_points)):
        displacement = math.pow(rn.random(), difficulty) * max_displacement
        disp = [displacement * i for i in make_rand_vector(2)]
        track_set[i*2] = track_points[i]
        track_set[i*2 + 1][0] = int((track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0])
        track_set[i*2 + 1][1] = int((track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1])
    for i in range(3):
        track_set = fix_angles(track_set)
        track_set = push_points_apart(track_set)
    # push any point outside screen limits back again
    final_set = []
    for point in track_set:
        if point[0] < margin:
            point[0] = margin
        elif point[0] > (WIDTH - margin):
            point[0] = WIDTH - margin
        if point[1] < margin:
            point[1] = margin
        elif point[1] > HEIGHT - margin:
            point[1] = HEIGHT - margin
        final_set.append(point)
    return final_set

def push_points_apart(points, distance=DISTANCE_BETWEEN_POINTS):
    # distance might need some tweaking
    distance2 = distance * distance 
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            if p_distance < distance:
                dx = points[j][0] - points[i][0];  
                dy = points[j][1] - points[i][1];  
                dl = math.sqrt(dx*dx + dy*dy);  
                dx /= dl;  
                dy /= dl;  
                dif = distance - dl;  
                dx *= dif;  
                dy *= dif;  
                points[j][0] = int(points[j][0] + dx);  
                points[j][1] = int(points[j][1] + dy);  
                points[i][0] = int(points[i][0] - dx);  
                points[i][1] = int(points[i][1] - dy);  
    return points

def fix_angles(points, max_angle=MAX_ANGLE):
    for i in range(len(points)):
        if i > 0:
            prev_point = i - 1
        else:
            prev_point = len(points)-1
        next_point = (i+1) % len(points)
        px = points[i][0] - points[prev_point][0]
        py = points[i][1] - points[prev_point][1]
        pl = math.sqrt(px*px + py*py)
        px /= pl
        py /= pl
        nx = -(points[i][0] - points[next_point][0])
        ny = -(points[i][1] - points[next_point][1])
        nl = math.sqrt(nx*nx + ny*ny)
        nx /= nl
        ny /= nl  
        a = math.atan2(px * ny - py * nx, px * nx + py * ny)
        if (abs(math.degrees(a)) <= max_angle):
            continue
        diff = math.radians(max_angle * math.copysign(1,a)) - a
        c = math.cos(diff)
        s = math.sin(diff)
        new_x = (nx * c - ny * s) * nl
        new_y = (nx * s + ny * c) * nl
        points[next_point][0] = int(points[i][0] + new_x)
        points[next_point][1] = int(points[i][1] + new_y)
    return points

def smooth_track(track_points):
    x = np.array([p[0] for p in track_points])
    y = np.array([p[1] for p in track_points])

    # append the starting x,y coordinates
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for # points evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, SPLINE_POINTS), tck)
    return [(int(xi[i]), int(yi[i])) for i in range(len(xi))]

def get_checkpoints(track_points, n_checkpoints=N_CHECKPOINTS):
    # get step between checkpoints
    checkpoint_step = len(track_points) // n_checkpoints
    # get checkpoint track points
    checkpoints = []
    for i in range(N_CHECKPOINTS):
        index = i * checkpoint_step
        checkpoints.append(track_points[index])
    return checkpoints

####
## drawing functions
####
def draw_points(surface, color, points):
    for p in points:
        draw_single_point(surface, color, p)

def draw_convex_hull(hull, surface, points, color):
    for i in range(len(hull.vertices)-1):
        draw_single_line(surface, color, points[hull.vertices[i]], points[hull.vertices[i+1]])
        # close the polygon
        if i == len(hull.vertices) - 2:
            draw_single_line(
                surface,
                color,
                points[hull.vertices[0]],
                points[hull.vertices[-1]]
            )

def draw_lines_from_points(surface, color, points):
    for i in range(len(points)-1):
        draw_single_line(surface, color, points[i], points[i+1])
        # close the polygon
        if i == len(points) - 2:
            draw_single_line(
                surface,
                color,
                points[0],
                points[-1]
            )

def draw_single_point(surface, color, pos, radius=2):
    pygame.draw.circle(surface, color, pos, radius)

def draw_single_line(surface, color, init, end):
    pygame.draw.line(surface, color, init, end)

def draw_track(surface, color, points, corners):
    # draw track with white border and black interior
    border_width = int(TRACK_BORDER_FT * PIXELS_PER_FOOT)
    
    # First pass: draw white border (full width)
    for i, point in enumerate(points):
        # Vary track width along the path
        progress = i / len(points)
        width = TRACK_WIDTH_MIN + (TRACK_WIDTH_MAX - TRACK_WIDTH_MIN) * (0.5 + 0.5 * math.sin(progress * 4 * math.pi))
        radius = int(width // 2)
        chunk_dimensions = (radius * 2, radius * 2)
        blit_pos = (point[0] - radius, point[1] - radius)
        track_chunk = pygame.Surface(chunk_dimensions, pygame.SRCALPHA)
        pygame.draw.circle(track_chunk, WHITE, (radius, radius), radius)
        surface.blit(track_chunk, blit_pos)
    
    # Second pass: draw black interior (with border offset)
    for i, point in enumerate(points):
        # Vary track width along the path
        progress = i / len(points)
        width = TRACK_WIDTH_MIN + (TRACK_WIDTH_MAX - TRACK_WIDTH_MIN) * (0.5 + 0.5 * math.sin(progress * 4 * math.pi))
        inner_radius = int((width // 2) - border_width)
        if inner_radius > 0:
            chunk_dimensions = (inner_radius * 2, inner_radius * 2)
            blit_pos = (point[0] - inner_radius, point[1] - inner_radius)
            track_chunk = pygame.Surface(chunk_dimensions, pygame.SRCALPHA)
            pygame.draw.circle(track_chunk, BLACK, (inner_radius, inner_radius), inner_radius)
            surface.blit(track_chunk, blit_pos)

def draw_parking_spots(surface, count=4, start_x=10, margin=10):
    # draw parking spots distributed throughout the course
    spot_w = max(1, int(PARKING_SPOT_WIDTH_FT * PIXELS_PER_FOOT))
    spot_h = max(1, int(PARKING_SPOT_LENGTH_FT * PIXELS_PER_FOOT))
    
    # Draw parking spots in multiple locations around the track
    positions = [
        (start_x, HEIGHT - margin - spot_h),  # bottom-left
        (WIDTH - start_x - spot_w, HEIGHT - margin - spot_h),  # bottom-right
        (start_x, start_x),  # top-left
        (WIDTH - start_x - spot_w, start_x),  # top-right
    ]
    
    for i, (x, y) in enumerate(positions[:count]):
        rect = pygame.Rect(x, y, spot_w, spot_h)
        pygame.draw.rect(surface, PARKING_SPOT_COLOR, rect, 2)
    
    # draw a small label-like scale bar (no font needed)
    bar_w = max(1, int(10 * PIXELS_PER_FOOT))
    bar_h = 4
    bar_x = start_x
    bar_y = HEIGHT - margin - spot_h - margin - bar_h
    pygame.draw.rect(surface, PARKING_SPOT_COLOR, (bar_x, bar_y, bar_w, bar_h))
    # draw bar endpoints
    pygame.draw.line(surface, PARKING_SPOT_COLOR, (bar_x, bar_y - 4), (bar_x, bar_y + bar_h + 4), 1)
    pygame.draw.line(surface, PARKING_SPOT_COLOR, (bar_x + bar_w, bar_y - 4), (bar_x + bar_w, bar_y + bar_h + 4), 1)

def draw_rectangle(dimensions, color, line_thickness=1, fill=False):
    filled = line_thickness
    if fill:
        filled = 0
    rect_surf = pygame.Surface(dimensions, pygame.SRCALPHA)
    pygame.draw.rect(rect_surf, color, (0, 0, dimensions[0], dimensions[1]), filled)
    return rect_surf

def draw_checkpoint(track_surface, points, checkpoint, debug=False):
    # given the main point of a checkpoint, compute and draw the checkpoint box
    margin = CHECKPOINT_MARGIN
    radius = TRACK_WIDTH_MIN // 2 + margin
    offset = CHECKPOINT_POINT_ANGLE_OFFSET
    check_index = points.index(checkpoint)
    vec_p = [points[check_index + offset][1] - points[check_index][1], -(points[check_index+offset][0] - points[check_index][0])]
    n_vec_p = [vec_p[0] / math.hypot(vec_p[0], vec_p[1]), vec_p[1] / math.hypot(vec_p[0], vec_p[1])]
    # compute angle
    angle = math.degrees(math.atan2(n_vec_p[1], n_vec_p[0]))
    # draw checkpoint
    checkpoint = draw_rectangle((radius*2, 5), BLUE, line_thickness=1, fill=False)
    rot_checkpoint = pygame.transform.rotate(checkpoint, -angle)
    if debug:
        rot_checkpoint.fill(RED)
    check_pos = (points[check_index][0] - math.copysign(1, n_vec_p[0])*n_vec_p[0] * radius, points[check_index][1] - math.copysign(1, n_vec_p[1])*n_vec_p[1] * radius)    
    track_surface.blit(rot_checkpoint, check_pos)

def save_track_svg(points, filename="track.svg"):
    # Create new SVG document
    dwg = svgwrite.Drawing(filename, size=(WIDTH, HEIGHT))
    
    # Add black background
    dwg.add(dwg.rect(insert=(0, 0), size=(WIDTH, HEIGHT), fill=f'rgb({BACKGROUND_COLOR[0]},{BACKGROUND_COLOR[1]},{BACKGROUND_COLOR[2]})'))
    
    # Create a path for the track with black fill and white border
    path = dwg.path(fill=f'rgb({BLACK[0]},{BLACK[1]},{BLACK[2]})')
    
    # Move to first point
    path.push(f'M {points[0][0]} {points[0][1]}')
    
    # Add line to each subsequent point
    for point in points[1:]:
        path.push(f'L {point[0]} {point[1]}')
    
    # Close the path
    path.push('Z')
    
    # Add the path with average track width and white stroke
    avg_track_width = (TRACK_WIDTH_MIN + TRACK_WIDTH_MAX) / 2
    path['stroke-width'] = 2
    path['stroke'] = f'rgb({TRACK_COLOR[0]},{TRACK_COLOR[1]},{TRACK_COLOR[2]})'
    path['stroke-linejoin'] = 'round'  # Round the corners
    path['stroke-linecap'] = 'round'   # Round the line ends
    
    dwg.add(path)
    dwg.save()

def save_track_openscad(points, filename="track.scad"):
    # Create a LineString from the track points
    line = LineString(points)
    
    # Buffer the line to create a polygon representing the track area
    # The buffer distance is half the track width
    track_polygon = line.buffer((TRACK_WIDTH_MIN + TRACK_WIDTH_MAX) / 4)
    
    # Get the coordinates of the outer and inner boundaries
    outer_coords = list(track_polygon.exterior.coords)
    inner_coords = []
    if hasattr(track_polygon, 'interiors') and len(track_polygon.interiors) > 0:
        inner_coords = list(track_polygon.interiors[0].coords)
    
    # Create OpenSCAD script
    with open(filename, 'w') as f:
        # Base prism
        f.write(f"// Base rectangular prism\n")
        f.write(f"difference() {{\n")
        f.write(f"    translate([0, 0, 0]) cube([{WIDTH}, {HEIGHT}, {PRISM_DEPTH}]);\n")
        
        # Track cutout
        f.write(f"    // Track cutout\n")
        f.write(f"    translate([0, 0, {PRISM_DEPTH - TRACK_CUTOUT_DEPTH}])\n")
        f.write(f"    linear_extrude(height={TRACK_CUTOUT_DEPTH + 1}, center=false)\n")
        f.write(f"    polygon(points=[")
        
        # Add outer boundary points
        for i, (x, y) in enumerate(outer_coords):
            # Flip y-coordinate as Pygame's origin is top-left, OpenSCAD's is bottom-left
            if i > 0:
                f.write(", ")
            f.write(f"[{x}, {HEIGHT - y}]")
        
        # Add inner boundary points if they exist
        if inner_coords:
            for x, y in inner_coords:
                f.write(f", [{x}, {HEIGHT - y}]")
        
        f.write("], paths=[[")
        
        # Add path indices for outer boundary
        for i in range(len(outer_coords)):
            if i > 0:
                f.write(", ")
            f.write(f"{i}")
        
        f.write("]")
        
        # Add inner boundary path if it exists
        if inner_coords:
            f.write(", [")
            for i in range(len(inner_coords)):
                if i > 0:
                    f.write(", ")
                f.write(f"{i + len(outer_coords)}")
            f.write("]")
        
        f.write("]);\n")
        f.write("}\n")

def convert_scad_to_stl(scad_filename="track.scad", stl_filename="track.stl"):
    """Convert SCAD file to STL using OpenSCAD command-line interface"""
    try:
        # Check if OpenSCAD is installed
        subprocess.run(["openscad", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Run OpenSCAD to convert SCAD to STL
        cmd = ["openscad", "-o", stl_filename, scad_filename]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(stl_filename):
            print(f"Successfully generated {stl_filename}")
            return True
        else:
            print(f"Failed to generate {stl_filename}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"Error converting {scad_filename} to {stl_filename}: {e}")
        print("Please make sure OpenSCAD is installed and in your PATH")
        return False
    except FileNotFoundError:
        print("OpenSCAD not found. Please install OpenSCAD to generate STL files")
        print("Download from: https://openscad.org/downloads.html")
        return False

####
## Main function
####
def main(debug=True, draw_checkpoints_in_track=True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    background_color = BACKGROUND_COLOR
    screen.fill(background_color)

    # generate the track
    points = random_points()
    hull = ConvexHull(points)
    track_points = shape_track(get_track_points(hull, points))
    f_points = smooth_track(track_points)
    # draw the actual track with white outline
    draw_track(screen, TRACK_COLOR, f_points, None)
    # draw checkpoints
    checkpoints = get_checkpoints(f_points)
    if draw_checkpoints_in_track or debug:
        for checkpoint in checkpoints:
            draw_checkpoint(screen, f_points, checkpoint, debug)
    if debug:
        # draw the different elements that end up
        # making the track
        draw_points(screen, WHITE, points)
        draw_convex_hull(hull, screen, points, RED)
        draw_points(screen, BLUE, track_points)
        draw_lines_from_points(screen, BLUE, track_points)    
        draw_points(screen, BLACK, f_points)

    pygame.display.set_caption(TITLE)
    # Save the screen to PNG and SVG files
    pygame.image.save(screen, "track.png")
    save_track_svg(f_points)
    save_track_openscad(f_points)
    # Convert SCAD to STL
    convert_scad_to_stl()
    
    while True: # main loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

def str2bool(v):
    """
    Helper method to parse strings into boolean values
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # rn.seed(rn.choice(COOL_TRACK_SEEDS))
    parser = argparse.ArgumentParser(description="Procedural racetrack generator")
    # Add parser options
    parser.add_argument("--debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Show racetrack generation steps")
    parser.add_argument("--show-checkpoints", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Show checkpoints")
    args = parser.parse_args()
    main(debug=args.debug, draw_checkpoints_in_track=args.show_checkpoints)
