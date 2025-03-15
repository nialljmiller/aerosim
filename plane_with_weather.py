import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
import random
import terrain as plane_terrain
import model as plane_model
import tree_system
import bird_system
import weather_effects as cloud_system  # Import the cloud and weather system
  # New import for bird flocking system

def rotate_vector(v, axis, angle):
    """Rotate vector v around a given axis by a given angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))




def draw_detailed_plane(plane):
    """Draw a slightly more complex wireframe plane model in local space.
       Local axes: +X = forward, +Y = up, +Z = right.
    """
    # Set color based on damage
    damage_factor = plane.damage
    undamaged_color = (1.0, 1.0, 0.0)  # Yellow
    damaged_color = (1.0, 0.0, 0.0)    # Red
    color = (
        undamaged_color[0] * (1-damage_factor) + damaged_color[0] * damage_factor,
        undamaged_color[1] * (1-damage_factor) + damaged_color[1] * damage_factor,
        undamaged_color[2] * (1-damage_factor) + damaged_color[2] * damage_factor
    )
    
    # Draw fuselage
    glColor3f(*color)
    glBegin(GL_LINE_LOOP)
    glVertex3f(1, 0, 0)  # Nose
    glVertex3f(-1, 0.5, 0)  # Top back
    glVertex3f(-1, -0.5, 0)  # Bottom back
    glEnd()

    # Draw fuselage (side view)
    glBegin(GL_LINE_LOOP)
    glVertex3f(1, 0, 0)  # Nose
    glVertex3f(-1, 0, 0.5)  # Right back
    glVertex3f(-1, 0, -0.5)  # Left back
    glEnd()

    # Draw wings (a quad) - BIGGER WINGS
    glColor3f(0, 1, 1)  # cyan
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, -3)  # Left wing tip
    glVertex3f(0, 0, 3)   # Right wing tip
    glVertex3f(-0.5, 0, 3)  # Right wing back
    glVertex3f(-0.5, 0, -3)  # Left wing back
    glEnd()
    
    # Draw wing details (ribs and struts) for visual effect
    glBegin(GL_LINES)
    # Main spar
    glVertex3f(-0.25, 0, -3)
    glVertex3f(-0.25, 0, 3)
    
    # Wing ribs
    for i in range(-2, 3):
        glVertex3f(0, 0, i)
        glVertex3f(-0.5, 0, i)
    glEnd()

    # Draw vertical tail (a small triangle)
    glColor3f(1, 0, 1)  # magenta
    glBegin(GL_LINE_LOOP)
    glVertex3f(-0.8, 0, 0)    # Bottom
    glVertex3f(-0.8, 1, 0)    # Top
    glVertex3f(-1, 0, 0)      # Back
    glEnd()
    
    # Draw horizontal stabilizer (tail wing)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-0.9, 0, -1)  # Left tip
    glVertex3f(-0.9, 0, 1)   # Right tip
    glVertex3f(-1, 0, 1)     # Right back
    glVertex3f(-1, 0, -1)    # Left back
    glEnd()

def draw_plane(plane):
    """Draw the plane using the detailed model."""
    glPushMatrix()
    # First translate to the plane's position
    glTranslatef(plane.position[0], plane.position[1], plane.position[2])
    
    # Build a rotation matrix from the plane's basis vectors (COLUMN-MAJOR ORDER for OpenGL)
    # Each column represents one of the plane's basis vectors
    matrix = [
        plane.forward[0], plane.forward[1], plane.forward[2], 0,
        plane.up[0], plane.up[1], plane.up[2], 0,
        plane.right[0], plane.right[1], plane.right[2], 0,
        0, 0, 0, 1
    ]
    glMultMatrixf(matrix)
    
    # Draw the plane model
    draw_detailed_plane(plane)
    
    # Optional: Draw velocity vector for debugging
    glColor3f(1, 0, 0)  # Red for velocity
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    # Scale velocity vector for better visualization
    vel_scale = 0.1
    glVertex3f(
        plane.velocity[0] * vel_scale,
        plane.velocity[1] * vel_scale,
        plane.velocity[2] * vel_scale
    )
    glEnd()
    
    # Draw collision points for debugging (useful to see where the plane is touching the ground)
    for collision in plane.collision_points:
        world_point = collision['world_point']
        normal = collision['normal']
        
        # Draw a small sphere at collision point
        glPushMatrix()
        glTranslatef(world_point[0] - plane.position[0], 
                    world_point[1] - plane.position[1], 
                    world_point[2] - plane.position[2])
        glColor3f(1, 0, 0)  # Red
        glutSolidSphere(0.2, 8, 8)  # Small sphere
        
        # Draw the terrain normal
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(normal[0], normal[1], normal[2])
        glEnd()
        glPopMatrix()
    
    glPopMatrix()

def draw_wireframe_text(text, x, y, z, scale=1.0, color=(0.0, 1.0, 0.0), rotate_to_face_camera=False):
    """
    Draw 3D wireframe text at the specified position in world space.
    
    Args:
        text: String to display
        x, y, z: Position in world space
        scale: Size of the text (default: 1.0)
        color: RGB tuple for text color (default: green)
        rotate_to_face_camera: Whether to rotate text to always face camera
    """
    glPushMatrix()
    
    # Move to the text position
    glTranslatef(x, y, z)
    
    # If enabled, make text billboard to always face camera
    if rotate_to_face_camera:
        # Get the current modelview matrix
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)
        
        # Extract the rotation part of the modelview matrix
        # We take the inverse of the upper 3x3 portion to cancel out camera rotation
        rotation = np.array([
            [modelview[0][0], modelview[1][0], modelview[2][0]],
            [modelview[0][1], modelview[1][1], modelview[2][1]],
            [modelview[0][2], modelview[1][2], modelview[2][2]]
        ])
        
        # Transpose rotation matrix (inverse of orthogonal matrix is its transpose)
        # This cancels the camera rotation
        rotation = rotation.transpose()
        
        # Create a 4x4 rotation matrix for OpenGL
        rot_matrix = [
            rotation[0][0], rotation[0][1], rotation[0][2], 0.0,
            rotation[1][0], rotation[1][1], rotation[1][2], 0.0,
            rotation[2][0], rotation[2][1], rotation[2][2], 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        
        # Apply the rotation to make text face camera
        glMultMatrixf(rot_matrix)
    
    # Scale the text
    glScalef(scale, scale, scale)
    
    # Set color for wireframe text
    glColor3f(color[0], color[1], color[2])
    
    # Draw each character
    for character in text:
        glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, ord(character))
    
    glPopMatrix()

def draw_sky_text(texts, fixed_position=None):
    """
    Draw multiple text lines in the sky, at a fixed position in world space.
    
    Args:
        texts: List of strings to display
        fixed_position: Optional tuple (x, y, z) for position in world space.
                        If None, uses a default position.
    """
    # Set fixed position in world space
    if fixed_position is None:
        # Default static position if none provided
        base_x = 100.0  # Fixed X coordinate in world space
        base_y = 50.0   # Fixed Y coordinate in world space (height)
        base_z = 0.0    # Fixed Z coordinate in world space
    else:
        base_x, base_y, base_z = fixed_position
    
    line_spacing = 15
    
    for i, text in enumerate(texts):
        # Position each line with proper spacing
        text_y = base_y - (i * line_spacing)
        draw_wireframe_text(text, base_x, text_y, base_z, scale=0.1, rotate_to_face_camera=False)





# Modified update function for Plane3D to use the infinite terrain
def update_with_terrain(plane, dt, terrain):
    """Physics update with terrain collision."""
    # Calculate thrust along the forward direction
    actual_thrust = plane.max_thrust * plane.throttle
    thrust_force = actual_thrust * plane.forward
    
    # Calculate current speed
    v = np.linalg.norm(plane.velocity)
    
    # Aerodynamics calculations
    if v > 0.1:
        airflow_direction = -plane.velocity / v
        angle_of_attack_cos = np.dot(airflow_direction, plane.forward)
        angle_of_attack = np.arccos(np.clip(angle_of_attack_cos, -1.0, 1.0))
        angle_factor = 1.0 - abs(angle_of_attack - 0.26) * 2.0
        effective_lift_coefficient = plane.params['lift_coefficient'] * max(0.1, min(1.2, angle_factor))
    else:
        effective_lift_coefficient = plane.params['lift_coefficient']
    
    # Drag calculation
    if v > 0.01:
        induced_drag_coefficient = (effective_lift_coefficient ** 2) / (3.14159 * 6.0)
        total_drag_coefficient = plane.params['drag_coefficient'] + induced_drag_coefficient
        drag_mag = 0.5 * plane.params['air_density'] * v**2 * total_drag_coefficient * plane.params['wing_area']
        drag_force = -drag_mag * (plane.velocity / v)
    else:
        drag_force = np.zeros(3)
    
    # Lift calculation
    if v > 1.0:
        airflow_direction = -plane.velocity / v
        lift_direction = np.cross(plane.right, airflow_direction)
        lift_direction = lift_direction / np.linalg.norm(lift_direction)
        lift_mag = 0.5 * plane.params['air_density'] * v**2 * plane.params['wing_area'] * effective_lift_coefficient
        lift_force = lift_mag * lift_direction
    else:
        lift_force = np.zeros(3)
    
    # Gravity force
    gravity_force = np.array([0.0, -plane.params['mass'] * plane.params['gravity'], 0.0])
    
    # Calculate total force and acceleration
    total_force = thrust_force + drag_force + lift_force + gravity_force
    acceleration = total_force / plane.params['mass']
    
    # Update velocity with physics forces
    plane.velocity += acceleration * dt
    
    # Physics for aircraft alignment with velocity
    if v > 1.0:
        forward_component = np.dot(plane.velocity, plane.forward)
        side_velocity = plane.velocity - (plane.forward * forward_component)
        side_speed = np.linalg.norm(side_velocity)
        
        if side_speed > 0.01:
            side_resistance = 0.25 * side_speed * side_speed
            plane.velocity -= (side_velocity / side_speed) * side_resistance * dt
            
        if v > 5.0 and not plane.is_grounded:
            alignment = np.dot(plane.forward, plane.velocity) / (np.linalg.norm(plane.forward) * v)
            alignment_factor = 0.8 * (1.0 - plane.throttle * 0.5)
            
            if alignment < 0.98:
                vel_normalized = plane.velocity / v
                rotation_axis = np.cross(plane.forward, vel_normalized)
                
                if np.linalg.norm(rotation_axis) > 0.01:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation_angle = alignment_factor * (1.0 - alignment) * dt * 5.0
                    plane.forward = rotate_vector(plane.forward, rotation_axis, rotation_angle)
                    plane.up = rotate_vector(plane.up, rotation_axis, rotation_angle)
                    
                    # Ensure vectors stay orthonormal
                    plane.forward = plane.forward / np.linalg.norm(plane.forward)
                    plane.right = np.cross(plane.forward, plane.up)
                    plane.right = plane.right / np.linalg.norm(plane.right)
                    plane.up = np.cross(plane.right, plane.forward)
                    plane.up = plane.up / np.linalg.norm(plane.up)
                    
        if plane.throttle < 0.2 and plane.velocity[1] > -5.0 and not plane.is_grounded:
            pitch_down_factor = (0.2 - plane.throttle) * 0.5
            pitch_down_axis = plane.right
            pitch_down_angle = pitch_down_factor * dt
            
            plane.forward = rotate_vector(plane.forward, pitch_down_axis, -pitch_down_angle)
            plane.up = rotate_vector(plane.up, pitch_down_axis, -pitch_down_angle)
    
    # Gliding physics
    if plane.throttle < 0.1 and v > 5.0 and plane.velocity[1] < 0 and not plane.is_grounded:
        horizontal_velocity = np.array([plane.velocity[0], 0, plane.velocity[2]])
        horizontal_speed = np.linalg.norm(horizontal_velocity)
        vertical_speed = abs(plane.velocity[1])
        
        current_glide_ratio = horizontal_speed / max(0.1, vertical_speed)
        target_glide_ratio = plane.glide_efficiency
        
        if current_glide_ratio < target_glide_ratio:
            glide_correction = min(0.05, (target_glide_ratio - current_glide_ratio) * 0.02)
            
            forward_horizontal = np.array([plane.forward[0], 0, plane.forward[2]])
            if np.linalg.norm(forward_horizontal) > 0.01:
                forward_horizontal = forward_horizontal / np.linalg.norm(forward_horizontal)
                correction_force = glide_correction * forward_horizontal * dt * v
                plane.velocity += correction_force
        
        elif current_glide_ratio > target_glide_ratio * 1.2:
            glide_correction = min(0.05, (current_glide_ratio - target_glide_ratio) * 0.01)
            plane.velocity[1] -= glide_correction * v * dt
    
    # Update position
    plane.position += plane.velocity * dt
    
    # Speed limits
    max_speed = 100.0
    speed = np.linalg.norm(plane.velocity)
    if speed > max_speed:
        plane.velocity = (plane.velocity / speed) * max_speed
    
    # Check collision with terrain using the new terrain system
    check_terrain_collisions(plane, terrain, dt)
    
def check_terrain_collisions(plane, terrain, dt):
    """Check if any part of the plane is colliding with the terrain."""
    plane.collision_points = []
    plane.is_grounded = False
    
    # Check each wing point for collision
    for point_idx, local_point in enumerate(plane.wing_points):
        # Convert to world coordinates
        world_point = plane.get_world_point(local_point)
        
        # Get terrain height at this point
        terrain_height = terrain.get_height(world_point[0], world_point[2])
        
        # Check if point is below terrain
        if world_point[1] < terrain_height:
            # Record collision point and penetration depth
            penetration = terrain_height - world_point[1]
            normal = terrain.get_terrain_normal(world_point[0], world_point[2])
            
            plane.collision_points.append({
                'point_idx': point_idx,
                'world_point': world_point,
                'penetration': penetration,
                'normal': normal
            })
            
            # If nose or central body collides, mark as grounded
            if point_idx in [4, 5]:  # Nose or tail
                plane.is_grounded = True
                plane.ground_normal = normal
    
    # Handle collisions if any
    if plane.collision_points:
        resolve_collisions(plane, dt)
        return True
    
    return False
    
def resolve_collisions(plane, dt):
    """Resolve terrain collisions with appropriate physics response."""
    if not plane.collision_points:
        return
    
    # Get current time for collision damage calculation
    current_time = pygame.time.get_ticks() / 1000.0
    collision_response = np.zeros(3)
    collision_count = len(plane.collision_points)
    
    for collision in plane.collision_points:
        penetration = collision['penetration']
        normal = collision['normal']
        point_idx = collision['point_idx']
        
        # Add collision response force
        response_strength = penetration * 50.0  # Proportional to penetration depth
        collision_response += normal * response_strength / collision_count
        
        # Calculate collision speed (dot product of velocity and normal)
        impact_speed = -np.dot(plane.velocity, normal)
        
        # Apply damage if impact is significant and time from last collision is sufficient
        if impact_speed > 10.0 and current_time - plane.last_collision_time > 0.5:
            damage_factor = (impact_speed - 10.0) / 40.0  # Scale damage (max at 50 m/s)
            plane.damage += min(0.2, damage_factor)  # Cap single collision damage
            plane.damage = min(1.0, plane.damage)  # Cap total damage
            plane.last_collision_time = current_time
            
            # Add angular impulse for realistic crash behavior
            if point_idx in [0, 1]:  # Wing tips
                # Add roll effect when wing hits ground
                roll_dir = 1 if point_idx == 0 else -1  # Left or right wing
                plane.up = rotate_vector(plane.up, plane.forward, roll_dir * 0.1 * impact_speed * dt)
                plane.right = np.cross(plane.forward, plane.up)
            elif point_idx == 4:  # Nose
                # Add pitch effect when nose hits ground
                plane.forward = rotate_vector(plane.forward, plane.right, -0.1 * impact_speed * dt)
                plane.up = rotate_vector(plane.up, plane.right, -0.1 * impact_speed * dt)
    
    # Apply position correction to move plane above ground
    if collision_response.any():
        # Move the plane out of the ground
        plane.position += collision_response * dt * 10
        
        # Reflect velocity for a bounce effect, with damping
        if plane.is_grounded:
            # If on wheels (grounded), apply rolling friction
            ground_speed = np.linalg.norm(plane.velocity - np.array([0, plane.velocity[1], 0]))
            friction = min(1.0, 0.01 * ground_speed)  # Friction increases with speed, up to a limit
            
            # Apply friction to horizontal velocity
            plane.velocity[0] *= (1.0 - friction)
            plane.velocity[2] *= (1.0 - friction)
            
            # Zero out vertical velocity if pointing down
            if plane.velocity[1] < 0:
                plane.velocity[1] = 0
        else:
            # For wing/tail collisions, apply a bounce with energy loss
            normal_vel = np.dot(plane.velocity, collision_response)
            if normal_vel < 0:  # Only bounce if moving toward the surface
                # Reflect velocity with damping
                reflection = plane.velocity - 1.8 * normal_vel * collision_response / np.linalg.norm(collision_response)
                energy_loss = 0.3  # 30% energy lost in collision
                plane.velocity = reflection * (1.0 - energy_loss)












    # Integration of infinite terrain system with the flight simulator
# Replace the existing terrain-related code with the new implementation
import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import celestial_system  # Import the celestial system module

def integrate_day_night_system(main_function_code):
    """
    This function modifies the main() function to integrate the celestial system.
    The approach preserves all existing functionality while adding the day/night cycle.
    """
    # Initialize the celestial system in the initialization section of main
    initialization_code = """
    # Initialize celestial system for day/night cycle
    print("Initializing celestial system...")
    celestial = celestial_system.CelestialSystem()
    celestial.initialize()
    # Set time progression to be faster than real-time for better gameplay
    celestial.set_time_progression(100.0)  # 100x faster than real time
    # Start at mid-morning for good lighting
    celestial.set_time(10)  # 10 AM
    """
    
    # Replace the fixed lighting setup with dynamic lighting from celestial system
    lighting_replacement = """
    # Lighting is now handled by the celestial system, but we still need base setup
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    # Light properties will be updated dynamically by celestial system
    diffuse_material = [0.8, 0.8, 0.8, 1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)
    """
    
    # Add celestial update to the main game loop
    game_loop_update = """
        # Update celestial system
        celestial.update(dt)
        
        # Update fog color based on time of day
        sun_altitude = np.degrees(np.atan2(celestial.sun.position[1], 
                                             np.sqrt(celestial.sun.position[0]**2 + celestial.sun.position[2]**2)))
        
        # Adjust fog color based on time of day
        zenith, horizon = celestial.sky_dome.update_colors(sun_altitude)
        
        # Update fog to match sky color at horizon
        fog_color = [horizon[0], horizon[1], horizon[2], 1.0]
        glFogfv(GL_FOG_COLOR, fog_color)
        
        # Adjust fog distance based on time of day (further during day, closer at night)
        day_factor = max(0, min(1, (sun_altitude + 10) / 20))  # 0 at night, 1 in day
        night_fog_start = 200.0
        day_fog_start = 500.0
        night_fog_end = 800.0
        day_fog_end = 1500.0
        
        glFogf(GL_FOG_START, night_fog_start + (day_fog_start - night_fog_start) * day_factor)
        glFogf(GL_FOG_END, night_fog_end + (day_fog_end - night_fog_end) * day_factor)
    """
    
    # Add celestial rendering to the rendering section (before terrain rendering)
    rendering_code = """
        # Draw celestial system (sky, sun, moon, stars)
        celestial.draw(cam_position)

        # Draw cloud and weather systems
        clouds.draw(cam_position)
    """
    
    # Add time information to the HUD
    hud_time_info = """
            f"TIME: {celestial.get_time_of_day_string()}",

            f"WEATHER: {clouds.get_weather_status()}",
    """
    
    # Add time controls to keyboard handling
    time_controls = """
                elif event.key == K_t:
                    # Toggle time progression speed
                    if celestial.time_scale == 100.0:
                        celestial.set_time_progression(300.0)  # Very fast
                    elif celestial.time_scale == 300.0:
                        celestial.set_time_progression(10.0)   # Slow
                    else:
                        celestial.set_time_progression(100.0)  # Normal
                elif event.key == K_PAGEUP:
                    # Advance time by 1 hour
                    celestial.current_time += 3600
                elif event.key == K_PAGEDOWN:
                    # Go back in time by 1 hour
                    celestial.current_time -= 3600

                elif event.key == K_w and pygame.key.get_mods() & KMOD_CTRL:
                    # Cycle through weather types
                    weather_types = ["clear", "fair", "overcast", "stormy"]
                    current_idx = weather_types.index(clouds.weather_type)
                    next_idx = (current_idx + 1) % len(weather_types)
                    clouds.start_weather_transition(weather_types[next_idx])
    """
    
    # Add new control instructions to the HUD
    new_controls = """
            "T: TOGGLE TIME SPEED",
            "PGUP/PGDN: ADV/REW TIME",

            "CTRL+W: CYCLE WEATHER",
    """
    
    return {
        'initialization': initialization_code,
        'lighting': lighting_replacement,
        'game_loop': game_loop_update,
        'rendering': rendering_code,
        'hud_time': hud_time_info,
        'time_controls': time_controls,
        'new_controls': new_controls
    }



import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time

def main():
    """Main function to demonstrate the enhanced terrain system."""
    # Initialize GLUT for text rendering
    glutInit()
    
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Enhanced Terrain Demo")
    
    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, display[0] / display[1], 0.1, 2000.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Set initial clear color (sky blue)
    glClearColor(0.7, 0.85, 1.0, 1.0)
    
    # Enable fog for distance fading
    glEnable(GL_FOG)
    fog_color = [0.7, 0.85, 1.0, 1.0]
    glFogfv(GL_FOG_COLOR, fog_color)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 500.0)
    glFogf(GL_FOG_END, 1500.0)
    
    # Basic lighting setup
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    # Light position (like the sun)
    light_position = [500.0, 1000.0, 500.0, 0.0]  # Directional light
    light_ambient = [0.3, 0.3, 0.3, 1.0]
    light_diffuse = [1.0, 1.0, 0.9, 1.0]
    
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    
    # Create the terrain system
    print("Initializing Enhanced Terrain System...")
    terrain = enhanced_terrain.EnhancedTerrain(chunk_size=100, resolution=10, view_distance=600)
    
    # Camera settings
    camera_pos = np.array([0.0, 100.0, 0.0])  # Start position
    camera_yaw = 0.0  # Horizontal rotation (around Y axis)
    camera_pitch = -30.0  # Vertical rotation (around X axis)
    camera_speed = 20.0  # Units per second
    mouse_sensitivity = 0.2
    
    # Simulation settings
    clock = pygame.time.Clock()
    wireframe_mode = False
    running = True
    
    # Initialize mouse for camera control
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    # For smooth frame rate and delta time calculation
    last_time = time.time()
    
    # Helper function to convert spherical to Cartesian coordinates
    def get_direction():
        # Convert degrees to radians
        yaw_rad = np.radians(camera_yaw)
        pitch_rad = np.radians(camera_pitch)
        
        # Calculate direction vector
        x = np.cos(yaw_rad) * np.cos(pitch_rad)
        y = np.sin(pitch_rad)
        z = np.sin(yaw_rad) * np.cos(pitch_rad)
        
        return np.array([x, y, z])
    
    # Function to draw text on screen (2D overlay)
    def draw_text(x, y, text, color=(255, 255, 255)):
        # Switch to 2D orthographic projection for rendering text
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Render text using Pygame's font system
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        # Create texture
        text_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, text_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_surface.get_width(), text_surface.get_height(), 
                    0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Enable blending for transparent background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable texture
        glEnable(GL_TEXTURE_2D)
        
        # Draw textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + text_surface.get_width(), y)
        glTexCoord2f(1, 1); glVertex2f(x + text_surface.get_width(), y + text_surface.get_height())
        glTexCoord2f(0, 1); glVertex2f(x, y + text_surface.get_height())
        glEnd()
        
        # Clean up
        glDeleteTextures(1, [text_texture])
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Restore states
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    # Main game loop
    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Process events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_TAB:
                    # Toggle wireframe mode
                    wireframe_mode = not wireframe_mode
                elif event.key == K_w and pygame.key.get_mods() & KMOD_CTRL:
                    # Toggle water rendering
                    water_enabled = terrain.toggle_water_rendering()
                    print(f"Water rendering: {'Enabled' if water_enabled else 'Disabled'}")
            elif event.type == MOUSEMOTION:
                # Update camera angles based on mouse movement
                dx, dy = event.rel
                camera_yaw += dx * mouse_sensitivity
                camera_pitch -= dy * mouse_sensitivity
                
                # Clamp pitch to prevent flipping
                camera_pitch = max(-89.0, min(89.0, camera_pitch))
        
        # Process keyboard input for movement
        keys = pygame.key.get_pressed()
        move_dir = np.zeros(3)
        
        # Forward/backward movement along the camera's forward vector
        direction = get_direction()
        if keys[K_w] and not pygame.key.get_mods() & KMOD_CTRL:
            move_dir += direction
        if keys[K_s]:
            move_dir -= direction
        
        # Strafe left/right perpendicular to camera direction
        if keys[K_a]:
            # Calculate right vector from direction (cross product with up)
            right = np.cross(direction, [0, 1, 0])
            right = right / np.linalg.norm(right)
            move_dir -= right
        if keys[K_d]:
            right = np.cross(direction, [0, 1, 0])
            right = right / np.linalg.norm(right)
            move_dir += right
        
        # Up/down movement
        if keys[K_SPACE]:
            move_dir[1] += 1.0  # Up
        if keys[K_LSHIFT]:
            move_dir[1] -= 1.0  # Down
        
        # Normalize and apply movement
        if np.linalg.norm(move_dir) > 0.0:
            move_dir = move_dir / np.linalg.norm(move_dir)
            camera_pos += move_dir * camera_speed * dt
        
        # Ensure camera doesn't go below terrain
        terrain_height = terrain.get_height(camera_pos[0], camera_pos[2])
        water_level = terrain.get_water_level(camera_pos[0], camera_pos[2])
        min_height = max(terrain_height, water_level) + 2.0  # Stay above terrain or water
        
        if camera_pos[1] < min_height:
            camera_pos[1] = min_height
        
        # Update terrain chunks based on camera position
        terrain.update_chunks(camera_pos)
        
        # Update water animation
        terrain.update_water_animation(dt)
        
        # Get biome information at camera position
        current_biome = terrain.get_biome_at(camera_pos[0], camera_pos[2])
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera position and orientation
        look_at = camera_pos + get_direction()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
                 look_at[0], look_at[1], look_at[2],
                 0, 1, 0)
        
        # Draw terrain
        terrain.draw(wireframe_mode)
        
        # Draw on-screen help text
        draw_text(5, 5, f"Current Biome: {current_biome}")
        draw_text(5, 25, f"Position: {camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}")
        draw_text(5, 45, f"Controls: WASD=Move, Space/Shift=Up/Down, Tab=Wireframe, Ctrl+W=Toggle Water")
        
        # Update window caption with information
        pygame.display.set_caption(
            f"Enhanced Terrain Demo | "
            f"Position: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}) | "
            f"Biome: {current_biome} | "
            f"FPS: {clock.get_fps():.1f}"
        )
        
        # Swap buffers and tick the clock
        pygame.display.flip()
        clock.tick(60)
    
    # Clean up resources
    terrain.cleanup()
    pygame.quit()

if __name__ == "__main__":
    main()