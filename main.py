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
import struct

try:
    from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

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

def get_track_bounds(track_points):
    """Get the bounding box of the track including track width"""
    xs = [p[0] for p in track_points]
    ys = [p[1] for p in track_points]
    
    # Account for track width (use max width to be safe)
    half_width = TRACK_WIDTH_MAX // 2
    
    min_x = min(xs) - half_width
    max_x = max(xs) + half_width
    min_y = min(ys) - half_width
    max_y = max(ys) + half_width
    
    return min_x, min_y, max_x, max_y

def calculate_required_image_size(track_points, margin=MARGIN):
    """Calculate required image size to fit the track with margin"""
    min_x, min_y, max_x, max_y = get_track_bounds(track_points)
    
    # Add margin on all sides
    required_width = int(max_x - min_x + 2 * margin)
    required_height = int(max_y - min_y + 2 * margin)
    
    # Calculate offset to center the track
    offset_x = int(-min_x + margin)
    offset_y = int(-min_y + margin)
    
    return required_width, required_height, offset_x, offset_y

def offset_track_points(track_points, offset_x, offset_y):
    """Offset all track points by the given amounts"""
    return [(p[0] + offset_x, p[1] + offset_y) for p in track_points]

def track_self_intersects(track_points):
    """Check if the track crosses over itself using shapely"""
    # Create a LineString from the track points (closed loop)
    coords = [(p[0], p[1]) for p in track_points]
    coords.append(coords[0])  # Close the loop
    line = LineString(coords)
    
    # A simple (non-self-intersecting) closed curve should be a valid ring
    # is_simple returns False if the line self-intersects
    return not line.is_simple

def get_track_width_at_point(track_points, index):
    """Get the track width at a given point index"""
    progress = index / len(track_points)
    width = TRACK_WIDTH_MIN + (TRACK_WIDTH_MAX - TRACK_WIDTH_MIN) * (0.5 + 0.5 * math.sin(progress * 4 * math.pi))
    return width

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

def save_track_svg(points, img_width=WIDTH, img_height=HEIGHT, filename="track.svg"):
    # Create new SVG document
    dwg = svgwrite.Drawing(filename, size=(img_width, img_height))
    
    # Add black background
    dwg.add(dwg.rect(insert=(0, 0), size=(img_width, img_height), fill=f'rgb({BACKGROUND_COLOR[0]},{BACKGROUND_COLOR[1]},{BACKGROUND_COLOR[2]})'))
    
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

def save_track_openscad(points, img_width=WIDTH, img_height=HEIGHT, filename="track.scad"):
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
        f.write(f"    translate([0, 0, 0]) cube([{img_width}, {img_height}, {PRISM_DEPTH}]);\n")
        
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
            f.write(f"[{x}, {img_height - y}]")
        
        # Add inner boundary points if they exist
        if inner_coords:
            for x, y in inner_coords:
                f.write(f", [{x}, {img_height - y}]")
        
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

def read_stl_file(filepath):
    """Read an STL file and return vertices and face indices"""
    vertices = []
    faces = []
    vertex_map = {}
    
    with open(filepath, 'rb') as f:
        # Read header (80 bytes)
        header = f.read(80)
        
        # Check if it's ASCII or binary
        f.seek(0)
        first_line = f.read(80)
        is_ascii = first_line.startswith(b'solid') and not b'\x00' in first_line
        f.seek(0)
        
        if is_ascii:
            # ASCII STL
            f.seek(0)
            content = f.read().decode('utf-8')
            lines = content.split('\n')
            current_face = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('vertex'):
                    parts = line.split()
                    v = (float(parts[1]), float(parts[2]), float(parts[3]))
                    
                    if v not in vertex_map:
                        vertex_map[v] = len(vertices)
                        vertices.append(v)
                    current_face.append(vertex_map[v])
                    
                    if len(current_face) == 3:
                        faces.extend(current_face)
                        current_face = []
        else:
            # Binary STL
            f.seek(80)  # Skip header
            num_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(num_triangles):
                # Read normal (12 bytes) - skip it
                f.read(12)
                
                face_indices = []
                for _ in range(3):
                    # Read vertex (12 bytes)
                    vx, vy, vz = struct.unpack('<fff', f.read(12))
                    v = (vx, vy, vz)
                    
                    if v not in vertex_map:
                        vertex_map[v] = len(vertices)
                        vertices.append(v)
                    face_indices.append(vertex_map[v])
                
                faces.extend(face_indices)
                
                # Read attribute byte count (2 bytes) - skip it
                f.read(2)
    
    return vertices, faces

def save_track_usd(points, floor_image="track.png", filename="field.usd", num_obstacles=10, img_width=WIDTH, img_height=HEIGHT):
    """Generate a USD file with the track as a floor and obstacles from assets using USD Python API"""
    
    if not USD_AVAILABLE:
        print("Warning: USD Python API not available. Install Omniverse or pixar USD to use this feature.")
        return
    
    # Available obstacle models in assets folder
    obstacle_models = [
        "SmallTrafficEasel.stl",
        "TrafficBarrel.stl",
        "TrafficBarrel2.stl",
        "TrafficCone.stl",
        "TrashCan.stl",
        "TrashCan2.stl"
    ]
    
    # Scale factor for obstacles (STL files are often in mm, we need them bigger)
    OBSTACLE_SCALE = 10.0
    
    # Get absolute path to assets directory and floor image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(script_dir, "assets")
    
    # Use relative path for texture (relative to USD file location)
    # Since USD file is saved in script_dir, just use the filename
    floor_image_rel = "./" + floor_image
    
    # Pre-load all STL meshes
    stl_meshes = {}
    for model in obstacle_models:
        model_path = os.path.join(assets_path, model)
        if os.path.exists(model_path):
            try:
                vertices, faces = read_stl_file(model_path)
                stl_meshes[model] = (vertices, faces)
            except Exception as e:
                print(f"Warning: Could not load {model}: {e}")
    
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(filename)
    
    # Set up axis and units for Isaac Sim
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # cm to meters
    
    # Create the World prim
    world = UsdGeom.Xform.Define(stage, "/World")
    
    # Create Looks scope for materials
    looks_scope = UsdGeom.Scope.Define(stage, "/World/Looks")
    
    # Create floor material with texture
    floor_material = UsdShade.Material.Define(stage, "/World/Looks/FloorMaterial")
    
    # Create PBR shader
    pbr_shader = UsdShade.Shader.Define(stage, "/World/Looks/FloorMaterial/PBRShader")
    pbr_shader.CreateIdAttr("UsdPreviewSurface")
    pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    
    # Create texture reader for diffuse color
    texture_reader = UsdShade.Shader.Define(stage, "/World/Looks/FloorMaterial/DiffuseTexture")
    texture_reader.CreateIdAttr("UsdUVTexture")
    texture_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(floor_image_rel))
    texture_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture_reader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    
    # Create UV coordinate reader
    uv_reader = UsdShade.Shader.Define(stage, "/World/Looks/FloorMaterial/UVReader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    
    # Connect UV reader to texture reader
    texture_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    
    # Connect texture to shader diffuse color
    pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_reader.ConnectableAPI(), "rgb")
    
    # Also connect to emissive color so white lines are always visible
    pbr_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_reader.ConnectableAPI(), "rgb")
    
    # Connect shader to material output
    floor_material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
    
    # Create the floor mesh
    floor_mesh = UsdGeom.Mesh.Define(stage, "/World/Floor")
    
    # Define floor vertices (quad plane) - use two triangles for better compatibility
    floor_vertices = [
        Gf.Vec3f(-img_width/2, -img_height/2, 0),
        Gf.Vec3f(img_width/2, -img_height/2, 0),
        Gf.Vec3f(img_width/2, img_height/2, 0),
        Gf.Vec3f(-img_width/2, img_height/2, 0)
    ]
    
    floor_mesh.GetPointsAttr().Set(floor_vertices)
    # Use two triangles instead of one quad for better compatibility
    floor_mesh.GetFaceVertexCountsAttr().Set([3, 3])
    floor_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    
    # Add normals pointing up
    floor_mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
    floor_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    
    # Make double-sided
    floor_mesh.GetDoubleSidedAttr().Set(True)
    
    # Add UV coordinates for the floor (texture mapping)
    # Flip V coordinate (1-v) because pygame uses top-left origin, USD uses bottom-left
    texCoords = UsdGeom.PrimvarsAPI(floor_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    texCoords.Set([(0, 1), (1, 1), (1, 0), (0, 0)])
    
    # Bind material to floor
    UsdShade.MaterialBindingAPI(floor_mesh).Bind(floor_material)
    
    # Generate random obstacle positions ON the track (within white borders)
    obstacle_count = 0
    border_width = int(TRACK_BORDER_FT * PIXELS_PER_FOOT)
    
    # Distribute obstacles evenly along the track with some randomness
    num_points = len(points)
    step = num_points // (num_obstacles + 1)  # Evenly space obstacles
    
    for i in range(num_obstacles):
        # Pick a point along the track with some randomness
        base_idx = (i + 1) * step
        idx = (base_idx + rn.randint(-step//4, step//4)) % num_points
        point = points[idx]
        
        # Get adjacent points to calculate track direction
        if idx > 0:
            prev_point = points[idx - 1]
        else:
            prev_point = points[-1]
        
        if idx < num_points - 1:
            next_point = points[idx + 1]
        else:
            next_point = points[0]
        
        # Calculate perpendicular direction (to offset within track width)
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Perpendicular vector (rotated 90 degrees)
            perp_x = -dy / length
            perp_y = dx / length
            
            # Get track width at this point and calculate inner track area
            track_width = get_track_width_at_point(points, idx)
            inner_radius = (track_width / 2) - border_width - 10  # Stay inside white lines with margin
            
            # Random offset WITHIN the track (between the white borders)
            offset = rn.uniform(-inner_radius, inner_radius)
            obs_x = point[0] + perp_x * offset
            obs_y = point[1] + perp_y * offset
            
            # Random rotation around Z axis
            rotation = rn.uniform(0, 360)
            rotation_rad = math.radians(rotation)
            
            # Randomly select an obstacle model
            model = rn.choice(obstacle_models)
            
            if model not in stl_meshes:
                continue
            
            vertices, faces = stl_meshes[model]
            
            # Create obstacle xform
            obstacle_xform = UsdGeom.Xform.Define(stage, f"/World/Obstacle_{obstacle_count}")
            
            # Set translation - convert from pygame coords to USD coords
            # pygame: origin top-left, Y down
            # USD floor: centered, with texture UV flipped
            usd_x = obs_x - img_width/2
            usd_y = obs_y - img_height/2
            obstacle_xform.AddTranslateOp().Set(Gf.Vec3f(usd_x, usd_y, 0))
            
            # Set rotation (around Z axis)
            rotation_quat = Gf.Quatf(math.cos(rotation_rad/2), 0, 0, math.sin(rotation_rad/2))
            obstacle_xform.AddOrientOp().Set(rotation_quat)
            
            # Set scale
            obstacle_xform.AddScaleOp().Set(Gf.Vec3f(OBSTACLE_SCALE, OBSTACLE_SCALE, OBSTACLE_SCALE))
            
            # Create the mesh under the xform
            obstacle_mesh = UsdGeom.Mesh.Define(stage, f"/World/Obstacle_{obstacle_count}/Mesh")
            
            # Set mesh data
            usd_vertices = [Gf.Vec3f(v[0], v[1], v[2]) for v in vertices]
            obstacle_mesh.GetPointsAttr().Set(usd_vertices)
            
            # All faces are triangles (3 vertices each)
            num_faces = len(faces) // 3
            obstacle_mesh.GetFaceVertexCountsAttr().Set([3] * num_faces)
            obstacle_mesh.GetFaceVertexIndicesAttr().Set(faces)
            
            obstacle_count += 1
    
    # Set the default prim
    stage.SetDefaultPrim(world.GetPrim())
    
    # Save the stage
    stage.Save()
    
    print(f"Successfully generated {filename} with {obstacle_count} obstacles from assets/")
    print(f"Note: Make sure '{floor_image}' is in the same directory as '{filename}'")

####
## Main function
####
def main(debug=True, draw_checkpoints_in_track=True, generate_scad=True, generate_stl=True, generate_usd=False, num_obstacles=10):
    pygame.init()
    
    # generate the track - retry if we don't get enough points or if track self-intersects
    max_attempts = 100
    attempts = 0
    f_points = None
    
    while attempts < max_attempts:
        attempts += 1
        points = random_points()
        
        if points is None or len(points) < 3:
            continue
        
        hull = ConvexHull(points)
        track_points = shape_track(get_track_points(hull, points))
        f_points = smooth_track(track_points)
        
        # Check if track self-intersects
        if not track_self_intersects(f_points):
            break
        
        if attempts % 10 == 0:
            print(f"Regenerating track (attempt {attempts})...")
    
    if f_points is None or track_self_intersects(f_points):
        print("Warning: Could not generate non-self-intersecting track after {max_attempts} attempts")
    
    # Calculate required image size to fit the entire track
    required_width, required_height, offset_x, offset_y = calculate_required_image_size(f_points)
    
    # Use the larger of default size or required size
    image_width = max(WIDTH, required_width)
    image_height = max(HEIGHT, required_height)
    
    # If we need a larger image, offset the track to center it
    if image_width > WIDTH or image_height > HEIGHT:
        print(f"Adjusting image size from {WIDTH}x{HEIGHT} to {image_width}x{image_height} to fit track")
        f_points = offset_track_points(f_points, offset_x, offset_y)
        # Also offset the debug points if needed
        if debug:
            points = np.array([(p[0] + offset_x, p[1] + offset_y) for p in points])
            track_points = [[p[0] + offset_x, p[1] + offset_y] for p in track_points]
    
    screen = pygame.display.set_mode((image_width, image_height))
    background_color = BACKGROUND_COLOR
    screen.fill(background_color)
    
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
    save_track_svg(f_points, image_width, image_height)
    
    # Conditionally generate SCAD and STL files
    if generate_scad:
        save_track_openscad(f_points, image_width, image_height)
    
    if generate_stl and generate_scad:
        convert_scad_to_stl()
    elif generate_stl and not generate_scad:
        print("Warning: SCAD generation is disabled. STL generation requires SCAD files. Skipping STL generation.")
    
    # Conditionally generate USD file
    if generate_usd:
        save_track_usd(f_points, filename="field.usd", num_obstacles=num_obstacles, img_width=image_width, img_height=image_height)
    
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
    parser.add_argument("--generate-scad", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Generate OpenSCAD file (default: True)")
    parser.add_argument("--generate-stl", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Generate STL file from OpenSCAD (default: True)")
    parser.add_argument("--generate-usd", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Generate USD field file with obstacles (default: False)")
    parser.add_argument("--num-obstacles", type=int, default=10,
                        help="Number of random obstacles to add in USD file (default: 10)")
    args = parser.parse_args()
    main(debug=args.debug, draw_checkpoints_in_track=args.show_checkpoints, 
         generate_scad=args.generate_scad, generate_stl=args.generate_stl,
         generate_usd=args.generate_usd, num_obstacles=args.num_obstacles)
