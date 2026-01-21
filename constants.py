# Screen dimensions
WIDTH = 1000 
HEIGHT = 1000

###
# Drawing
###
TITLE = 'Procedural Race Track'

# Colors
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
BLUE = [0, 0, 255]
GRASS_GREEN = [58, 156, 53]
GREY = [186, 182, 168]

CHECKPOINT_POINT_ANGLE_OFFSET = 3
CHECKPOINT_MARGIN = 5

###
# Track parameters
###

# Boundaries for the numbers of points that will be randomly 
# generated to define the initial polygon used to build the track
MIN_POINTS = 20
MAX_POINTS = 30

SPLINE_POINTS = 1000

# Margin between screen limits and any of the points that shape the
# initial polygon
MARGIN = 50
# minimum distance between points that form the track skeleton
MIN_DISTANCE = 20
# Maximum midpoint displacement for points placed after obtaining the initial polygon
MAX_DISPLACEMENT = 80
# Track difficulty
DIFFICULTY = 0.1
# min distance between two points that are part of thr track skeleton
DISTANCE_BETWEEN_POINTS = 20
# Maximum corner allowed angle
MAX_ANGLE = 90

TRACK_WIDTH = 40

###
# 3D Model parameters
###
PRISM_DEPTH = 100        # Depth/height of the base material in mm
TRACK_CUTOUT_DEPTH = 20  # Depth of the track cutout in mm

###
# Game parameters
###
N_CHECKPOINTS = 10

###
# Some seeds I find cool or interesting
###
COOL_TRACK_SEEDS = [
    911, 
    639620465, 
    666574559, 
    689001243, 
    608068482, 
    1546, 
    8, 
    83, 
    945, 
    633, 
    10, 
    23, 
    17, 
    123, 
    1217, 
    12, 
    5644, 
    5562, 
    2317, 
    1964, 
    95894, 
    95521
]

###
# IGVC-specific parameters
###
# Course dimensions (approximate, in feet)
COURSE_LENGTH_FT = 500
COURSE_AREA_WIDTH_FT = 120
COURSE_AREA_DEPTH_FT = 100

# Track width limits (feet)
TRACK_WIDTH_MIN_FT = 10
TRACK_WIDTH_MAX_FT = 20

# Minimum turning radius (feet)
TURN_RADIUS_MIN_FT = 5

# Maximum ramp gradient (%)
RAMP_MAX_GRADE_PERCENT = 15

# Speed limits (mph)
MIN_SPEED_MPH = 1
MAX_SPEED_MPH = 5

# Run time limit (minutes)
RUN_TIME_MINUTES = 6

# A simple conversion from canvas pixels to feet based on the IGVC area width
# This can be used to draw a scale or elements sized in feet (eg. parking spots)
PIXELS_PER_FOOT = WIDTH / COURSE_AREA_WIDTH_FT

# Parking spot (typical) dimensions in feet (width x length)
PARKING_SPOT_WIDTH_FT = 9
PARKING_SPOT_LENGTH_FT = 18
PARKING_SPOT_COLOR = WHITE
