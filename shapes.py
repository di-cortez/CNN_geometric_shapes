import numpy as np
import random

def draw_rectangle(draw, img_size, color, **kwargs):
    """draw a rectangle with random coordinates."""
    x0, y0 = np.random.randint(0, img_size - 3, size=2)
    # ensure the rectangle has 3 pixels
    x1 = np.random.randint(x0 + 3, img_size)
    y1 = np.random.randint(y0 + 3, img_size)
    draw.rectangle([x0, y0, x1, y1], fill=color)
    return "rectangle"

def draw_ellipse(draw, img_size, color, **kwargs):
    """draw an eplipse with random coordinates."""
    x0, y0 = np.random.randint(0, img_size - 3, size=2)
    x1 = np.random.randint(x0 + 3, img_size)
    y1 = np.random.randint(y0 + 3, img_size)
    draw.ellipse([x0, y0, x1, y1], fill=color)
    return "ellipse"

def get_angle(p1, p2, p3):
    a = np.linalg.norm(np.array(p3) - np.array(p2))
    b = np.linalg.norm(np.array(p1) - np.array(p2))
    c = np.linalg.norm(np.array(p3) - np.array(p1))
    
    if a*b == 0: return 0
    cos_angle = (a**2 + b**2 - c**2)/(2*a*b)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def get_height(p1, p2, p3):
    sides = sorted([
        np.linalg.norm(np.array(p3) - np.array(p2)),
        np.linalg.norm(np.array(p1) - np.array(p2)),
        np.linalg.norm(np.array(p3) - np.array(p1))
    ])

    base = sides[2]
    area = 0.5*abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    if base == 0: return 0
    return 2*area/base

def draw_triangle(draw, img_size, color, **kwargs):
    """draw a triangle with random coordinates."""
    # generates 3 random points
    min_angle = 25
    min_side = min(img_size*0.3, 6)
    min_height = min(img_size*0.1, 3)

    attempts = 0
    while attempts < 500:
        p1 = tuple(np.random.randint(0, img_size, size=2))
        p2 = tuple(np.random.randint(0, img_size, size=2))
        p3 = tuple(np.random.randint(0, img_size, size=2))

        angle1 = get_angle(p3, p1, p2)
        angle2 = get_angle(p1, p2, p3)
        angle3 = get_angle(p1, p3, p2)
        if angle1 < min_angle or angle2 < min_angle or angle3 < min_angle:
            attempts +=1
            continue

        side1 = np.linalg.norm(np.array(p1) - np.array(p2))
        side2 = np.linalg.norm(np.array(p2) - np.array(p3))
        side3 = np.linalg.norm(np.array(p3) - np.array(p1))

        if side1 < min_side or side2 < min_side or side3 < min_side:
            attempts += 1
            continue

        height = get_height(p1, p2, p3)
        if height < min_height:
            attempts += 1
            continue
        
        draw.polygon([p1, p2, p3], fill=color)
        return "triangle"

    from. import draw_rectangle
    return draw_rectangle(draw, img_size, color)

def draw_rhombus(draw, img_size, color, **kwargs):
    """draw an rhombus with random coordinates."""
    margin = int(img_size*0.2)
    # rhombus center
    center_x, center_y = np.random.randint(margin, img_size - margin, size=2)
    
    max_hf_w = min(center_x, img_size - center_x) - 1
    max_hf_h = min(center_y, img_size - center_y) - 1

    if max_hf_w < 3 or max_hf_h < 3:
        from . import draw_rectangle
        return draw_rectangle(draw, img_size, color)
    # half width and height
    half_w = np.random.randint(3, max_hf_w + 1)
    half_h = np.random.randint(3, max_hf_h + 1)

    # rhombus points
    p1 = (center_x, center_y - half_h) # top
    p2 = (center_x + half_w, center_y) # right
    p3 = (center_x, center_y + half_h) # bottom
    p4 = (center_x - half_w, center_y) # left
    
    draw.polygon([p1, p2, p3, p4], fill=color)
    return "rhombus"

def draw_star(draw, img_size, color, num_points=5, **kwargs):
    """draw a star with random coordinates."""
    center_x, center_y = np.random.randint(10, img_size - 10, size=2)
    outer_radius = np.random.randint(8, min(center_x, center_y, img_size-center_x, img_size-center_y))
    inner_radius = outer_radius / 2.0

    points = []
    angle = -np.pi / 2 
    angle_step = np.pi / num_points

    for i in range(num_points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((x, y))
        angle += angle_step

    draw.polygon(points, fill=color)
    return "star"

def draw_crescent(draw, img_size, color, **kwargs):
    """draw a half-moon with random coordinates."""

    background_color = kwargs.get('background_color', (0,0,0))

    center_x, center_y = np.random.randint(10, img_size - 10, size=2)
    radius = np.random.randint(8, min(center_x, center_y, img_size-center_x, img_size-center_y))
    
    # extern circle
    bbox_outer = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    draw.ellipse(bbox_outer, fill=color)
    
    # intern circle that cut the external circle
    offset = radius / 2

    if random.choice([True, False]):
        bbox_inner = [center_x - radius + offset, center_y - radius, center_x + radius + offset, center_y + radius]
    else:
        bbox_inner = [center_x - radius - offset, center_y - radius, center_x + radius - offset, center_y + radius]
    # intern circle color
    draw.ellipse(bbox_inner, fill=background_color)
    
    return "crescent"

# dict to map shape name to the draw function
SHAPE_FUNCTIONS = {
    "rectangle": draw_rectangle,
    "ellipse": draw_ellipse,
    "triangle": draw_triangle,
    "rhombus": draw_rhombus,
    "star": draw_star,
    "crescent": draw_crescent,
}

# IDs list to Labels
SHAPE_IDS = {name: i for i, name in enumerate(SHAPE_FUNCTIONS.keys())}

def get_random_shape_function():
    """returns a random drawing function with its name."""
    shape_name = random.choice(list(SHAPE_FUNCTIONS.keys()))
    return SHAPE_FUNCTIONS[shape_name], shape_name