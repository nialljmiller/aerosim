import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
import random
import sys
import time


try:
    import celestial_system
    import terrain as plane_terrain
    import model as plane_model
    import tree_system
    import bird_system
    import weather_effects as cloud_system  # Import the cloud and weather system
    import water_system
    import plane_model_renderer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the current directory.")
    sys.exit(1)

def rotate_vector(v, axis, angle):
    """Rotate vector v around a given axis by a given angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))

def create_enhanced_plane(params, initial_position, initial_velocity):
    """Create an instance of the enhanced plane model with the specified parameters."""
    # Convert existing parameters to work with the enhanced plane model
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    
    # Create the enhanced plane model
    plane = Plane3D(initial_position, initial_velocity, forward, up, params)
    
    # Set initial state
    plane.throttle = 0.3  # Start at 30% throttle
    plane.gear_state = 1.0  # Start with landing gear down
    plane.gear_target = 1.0
    plane.flaps = 0.0  # Start with flaps retracted
    plane.flaps_target = 0.0
    
    return plane



def draw_wireframe_text(text, x, y, z, scale=1.0, color=(0.0, 1.0, 0.0), rotate_to_face_camera=False):
    """
    Draw 3D wireframe text at the specified position in world space.
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






def main():
    """Main function for the flight simulator."""
    # Initialize GLUT for text rendering
    print("Initializing GLUT...")
    glutInit()
    
    # Simulation parameters
    params = {
        'gravity': 9.81,
        'air_density': 1.225,
        'mass': 500.0,
        'wing_area': 50.0,        # Increased for better gliding
        'drag_coefficient': 0.2,  # Reduced for better gliding
        'lift_coefficient': 1.2,  # Increased for better gliding 
        'propeller_thrust': 6000.0,
        'glide_ratio': 15.0,      # Glide ratio (distance:height)
    }

    print("Initializing Pygame...")
    pygame.init()
    display = (800, 600)
    try:
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Flight Simulator")
    except pygame.error as e:
        print(f"Error setting up display: {e}")
        sys.exit(1)

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, display[0] / display[1], 0.1, 2000.0)  # Extended far plane
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.5, 0.7, 1.0, 1.0)  # Sky blue background
    
    # Enable fog for distance fading
    print("Setting up fog...")
    glEnable(GL_FOG)
    fog_color = [0.5, 0.7, 1.0, 1.0]  # Match sky color
    glFogfv(GL_FOG_COLOR, fog_color)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 500.0)
    glFogf(GL_FOG_END, 1500.0)
    
    # Initialize celestial system for day/night cycle
    print("Initializing celestial system...")
    try:
        celestial = celestial_system.CelestialSystem()
        celestial.initialize()
        # Set time progression to be faster than real-time for better gameplay
        celestial.set_time_progression(100.0)  # 100x faster than real time
        # Start at mid-morning for good lighting
        celestial.set_time(10)  # 10 AM
    except Exception as e:
        print(f"Error initializing celestial system: {e}")
        celestial = None  # Continue without celestial system

    # Initialize infinite terrain system
    print("Initializing terrain system...")
    try:
        terrain = plane_terrain.InfiniteTerrain(chunk_size=100, resolution=50, view_distance=600)
    except Exception as e:
        print(f"Error initializing terrain: {e}")
        sys.exit(1)
    
    # Initialize tree system
    print("Initializing tree system...")
    try:
        trees = tree_system.TreeSystem(terrain, density=0.008, max_trees_per_chunk=200)
    except Exception as e:
        print(f"Error initializing tree system: {e}")
        trees = None  # Continue without trees
    
    # Initialize bird flocking system
    print("Initializing bird system...")
    try:
        birds = bird_system.BirdSystem(terrain, max_flocks=15)
    except Exception as e:
        print(f"Error initializing bird system: {e}")
        birds = None  # Continue without birds
    
    # Initialize improved cloud and weather system
    print("Initializing enhanced cloud and weather system...")
    clouds = cloud_system.CloudSystem(terrain)
    
    # Start with overcast weather for more visible clouds 
    if random.random() < 0.5:
        clouds.weather_type = "fair"
    else:
        clouds.weather_type = "overcast"
    
    # Generate initial clouds with the selected weather type
    clouds.generate_initial_clouds()
    print(f"Weather initialized to: {clouds.weather_type}")
    
    # Force texture loading immediately 
    clouds.load_textures()
    print(f"Cloud texture ID: {clouds.cloud_texture}")
    

    # Initialize water system
    print("Initializing water system...")
    try:
        water = water_system.WaterSystem(water_level=5.0, size=10000.0)
    except Exception as e:
        print(f"Error initializing water system: {e}")
        water = None  # Continue without water



    # Create plane at a safe starting position
    print("Creating plane...")
    initial_position = [0.0, 50.0, 0.0]  # Start higher for safety
    initial_velocity = [10.0, 0.0, 0.0]  # Start with some forward velocity
    forward = [1.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    try:
        plane = plane_model.Plane3D(initial_position, initial_velocity, forward, up, params)
    except Exception as e:
        print(f"Error creating plane: {e}")
        sys.exit(1)

    # Camera follow variables
    cam_smoothness = 0.1
    cam_position = np.array(initial_position)
    
    # Setup lighting
    print("Setting up lighting...")
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    # Basic light properties (will be dynamically updated)
    diffuse_material = [0.8, 0.8, 0.8, 1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)

    print("Setting up game clock...")
    clock = pygame.time.Clock()
    running = True
    wireframe_mode = False  # Toggle for wireframe view
    
    # Game state tracking
    game_time = 0.0
    
    # Initial terrain update
    try:
        terrain.update_chunks(initial_position)
    except Exception as e:
        print(f"Error updating terrain chunks: {e}")
    
    print("Entering main game loop...")
    while running:
        try:
            dt = clock.tick(60) / 1000.0
            game_time += dt
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_TAB:
                        # Toggle wireframe mode
                        wireframe_mode = not wireframe_mode
                    elif event.key == K_r:
                        # Reset plane position (for crashes)
                        plane.position = np.array([0.0, 50.0, 0.0])
                        plane.velocity = np.array([10.0, 0.0, 0.0])
                        plane.forward = np.array([1.0, 0.0, 0.0])
                        plane.up = np.array([0.0, 1.0, 0.0])
                        plane.right = np.array([0.0, 0.0, 1.0])
                        plane.throttle = 0.3
                        plane.damage = 0.0
                        
                        # Reset gear and flaps to default state
                        if hasattr(plane, 'gear_state'):
                            plane.gear_state = 1.0
                            plane.gear_target = 1.0
                        if hasattr(plane, 'flaps'):
                            plane.flaps = 0.0
                            plane.flaps_target = 0.0
                            
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
                    elif event.key == K_g:
                        # Toggle landing gear
                        if hasattr(plane, 'toggle_landing_gear'):
                            plane.toggle_landing_gear()
                            print(f"Landing gear {'lowering' if plane.gear_target > 0.5 else 'retracting'}")
                    elif event.key == K_f:
                        # Cycle through flap settings (0, 0.5, 1.0)
                        if hasattr(plane, 'set_flaps'):
                            current_flaps = round(plane.flaps_target * 2) / 2  # Round to nearest 0.5
                            next_flaps = (current_flaps + 0.5) % 1.5  # Cycle 0 -> 0.5 -> 1.0 -> 0
                            plane.set_flaps(next_flaps)
                            print(f"Flaps set to {int(next_flaps * 100)}%")
                    # Weather control hotkeys
                    elif event.key == K_w and pygame.key.get_mods() & KMOD_CTRL:
                        # Cycle through weather types
                        weather_types = ["clear", "fair", "overcast", "stormy", "extreme"]
                        current_idx = weather_types.index(clouds.weather_type) if clouds.weather_type in weather_types else 0
                        next_idx = (current_idx + 1) % len(weather_types)
                        clouds.start_weather_transition(weather_types[next_idx])
                        print(f"Weather transitioning to: {weather_types[next_idx]}")
                    elif event.key == K_x and pygame.key.get_mods() & KMOD_CTRL:
                        # Force extreme weather
                        clouds.start_weather_transition("extreme")
                        print("Extreme weather incoming!")
                    elif event.key == K_c and pygame.key.get_mods() & KMOD_CTRL:
                        # Force clear weather
                        clouds.start_weather_transition("clear")
                        print("Clearing up the skies!")

            # Process controls
            # Mapping: Up/Down control pitch, Left/Right control roll
            keys = pygame.key.get_pressed()

            # Flight controls
            pitch = roll = yaw = 0.0
            control_speed = np.radians(45)  # 45 deg/sec
            if keys[pygame.K_UP]:
                pitch = control_speed     # Nose up
            if keys[pygame.K_DOWN]:
                pitch = -control_speed    # Nose down
            if keys[pygame.K_LEFT]:
                roll = control_speed      # Bank left
            if keys[pygame.K_RIGHT]:
                roll = -control_speed     # Bank right
            if keys[pygame.K_a]:
                yaw = control_speed       # Yaw left
            if keys[pygame.K_d]:
                yaw = -control_speed      # Yaw right
                
            # Throttle control
            throttle_increment = 0.05  # 5% throttle change per key press
            throttle_change = False    # Track if throttle has changed
            
            old_throttle = plane.throttle  # Store old throttle value to detect changes
            
            if keys[pygame.K_w]:
                plane.throttle = min(1.0, plane.throttle + throttle_increment * dt * 10)
                throttle_change = plane.throttle != old_throttle
            if keys[pygame.K_s]:
                plane.throttle = max(0.0, plane.throttle - throttle_increment * dt * 10)
                throttle_change = plane.throttle != old_throttle
                
            # Number keys for quick throttle settings
            if keys[pygame.K_0]:
                throttle_change = plane.throttle != 0.0
                plane.throttle = 0.0      # Idle
            if keys[pygame.K_1]:
                throttle_change = plane.throttle != 0.1
                plane.throttle = 0.1      # 10%
            if keys[pygame.K_2]:
                throttle_change = plane.throttle != 0.25
                plane.throttle = 0.25     # 25%
            if keys[pygame.K_3]:
                throttle_change = plane.throttle != 0.5
                plane.throttle = 0.5      # 50%
            if keys[pygame.K_4]:
                throttle_change = plane.throttle != 0.75
                plane.throttle = 0.75     # 75%
            if keys[pygame.K_5]:
                throttle_change = plane.throttle != 1.0
                plane.throttle = 1.0      # 100% (Full throttle)
                
            # Apply a small velocity reduction when throttle is decreased significantly
            if throttle_change and old_throttle > plane.throttle:
                throttle_reduction = (old_throttle - plane.throttle) * 0.1
                # Reduce velocity proportionally to throttle reduction
                plane.velocity *= (1.0 - throttle_reduction)
                
            # Update celestial system
            if celestial is not None:
                celestial.update(dt)
                
                # Update fog color based on time of day
                sun_altitude = np.degrees(np.arctan2(celestial.sun.position[1], 
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

            # Update terrain chunks based on aircraft position
            try:
                terrain.update_chunks(plane.position)
            except Exception as e:
                print(f"Error updating terrain: {e}")

            # Update water system
            if water is not None:
                try:
                    water.update(dt, cam_position)
                except Exception as e:
                    print(f"Error updating water: {e}")



            # Update additional systems
            if trees is not None:
                try:
                    trees.update(plane.position)
                except Exception as e:
                    print(f"Error updating trees: {e}")
                    
            if birds is not None:
                try:
                    birds.update(dt, plane.position, game_time)
                except Exception as e:
                    print(f"Error updating birds: {e}")
                 
            # Update cloud and weather system
            clouds.update(dt, celestial.current_time / (24 * 60 * 60), plane.position)
            
            # Apply controls with roll inversion for correct visual feedback
            plane.apply_controls(pitch, -roll, yaw, dt)  # Notice the negated roll input
            
            # Update plane with custom terrain-aware physics
            update_with_terrain(plane, dt, terrain)
            
            # Get current speed for display and calculations
            speed = np.linalg.norm(plane.velocity)
            altitude = plane.position[1]
            ground_height = terrain.get_height(plane.position[0], plane.position[2])
            height_above_ground = altitude - ground_height
            
            # Add debug info about current glide status
            glide_info = ""
            if plane.throttle < 0.1 and speed > 5.0:
                horizontal_velocity = np.array([plane.velocity[0], 0, plane.velocity[2]])
                horizontal_speed = np.linalg.norm(horizontal_velocity)
                vertical_speed = abs(plane.velocity[1]) if plane.velocity[1] < 0 else 0.1
                current_glide_ratio = horizontal_speed / vertical_speed
                glide_info = f" | GLIDING: {current_glide_ratio:.1f}:1"
            

            # Update window caption with flight information
            time_str = ""
            if celestial is not None:
                time_str = f" | Time: {celestial.get_time_of_day_string()}"
                
            pygame.display.set_caption(
                f"Speed: {speed:.1f} m/s | Alt: {height_above_ground:.1f}m AGL | "
                f"Throttle: {plane.throttle*100:.0f}%{glide_info} | "
                f"Damage: {plane.damage*100:.0f}%{time_str}"
            )

            # Enhanced Camera System with Strict Terrain Collision Prevention
            # Calculate desired camera position (15 units behind, 3 units above)
            camera_offset = plane.forward * -15 + plane.up * 3
            target_cam_pos = plane.position + camera_offset
            
            # Check terrain height at desired camera position BEFORE applying smoothing
            terrain_height_at_target = terrain.get_height(target_cam_pos[0], target_cam_pos[2])
            min_camera_height = terrain_height_at_target + 2.0  # Increased clearance to 2.0 units
            
            # Apply height constraint to target position first
            if target_cam_pos[1] < min_camera_height:
                height_correction = min_camera_height - target_cam_pos[1]
                target_cam_pos[1] = min_camera_height
                
                # Optional: Adjust look target slightly upward when camera is terrain-constrained
                look_adjustment = height_correction * 0.2
                target_look_point = plane.position + plane.forward * 5 + np.array([0, look_adjustment, 0])
            else:
                # Normal look target slightly ahead of the plane
                target_look_point = plane.position + plane.forward * 5
            
            # Apply camera smoothing AFTER terrain constraint
            cam_smoothness = 0.2
            cam_position += (target_cam_pos - cam_position) * cam_smoothness
            
            # Apply a SECOND terrain check after smoothing for absolute prevention
            terrain_height_at_camera = terrain.get_height(cam_position[0], cam_position[2])
            absolute_min_height = terrain_height_at_camera + 1.0
            if cam_position[1] < absolute_min_height:
                cam_position[1] = absolute_min_height
                
            # Clear screen and prepare for rendering
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Position camera
            gluLookAt(
                cam_position[0], cam_position[1], cam_position[2],
                target_look_point[0], target_look_point[1], target_look_point[2],
                0, 1, 0
            )

            if celestial is not None:
                try:
                    celestial.draw(cam_position)
                except Exception as e:
                    print(f"Error drawing celestial system: {e}")

            # Add this block to draw the clouds immediately after the celestial system
            try:
                clouds.draw(cam_position)
            except Exception as e:
                print(f"Error drawing clouds: {e}")

            # Disable lighting for wireframe elements
            glDisable(GL_LIGHTING)
                        
            # Draw terrain with or without wireframe
            try:
                terrain.draw(wireframe=wireframe_mode)
            except Exception as e:
                print(f"Error drawing terrain: {e}")

            try:
                water.draw(cam_position)
            except Exception as e:
                print(f"Error drawing water: {e}")
            
            # Draw trees (only in solid mode)
            if not wireframe_mode and trees is not None:
                # Enable lighting for trees
                glEnable(GL_LIGHTING)
                try:
                    trees.draw(cam_position)
                except Exception as e:
                    print(f"Error drawing trees: {e}")
                glDisable(GL_LIGHTING)
                
               # Draw bird flocks
                if birds is not None:
                    glEnable(GL_LIGHTING)
                    try:
                        birds.draw(cam_position)
                    except Exception as e:
                        print(f"Error drawing birds: {e}")
                    glDisable(GL_LIGHTING)
            
            # Draw plane
            try:
                plane_model_renderer.draw_plane(plane)
            except Exception as e:
                print(f"Error drawing plane: {e}")
            
            # Draw sky text with flight information and controls
            is_night = celestial is not None and celestial.is_night()
            if is_night:
                text_color = (0.8, 0.8, 0.2)  # Yellow for better visibility at night
            else:
                text_color = (0.0, 1.0, 0.0)  # Green for day
                
            time_info = f"TIME: {celestial.get_time_of_day_string()}" if celestial is not None else ""
            

            sky_texts = [
                "FLIGHT SIMULATOR",
                f"SPEED: {speed:.1f} m/s",
                f"ALTITUDE: {height_above_ground:.1f}m AGL",
                f"THROTTLE: {plane.throttle*100:.0f}%",
                f"DAMAGE: {plane.damage*100:.0f}%",
                time_info,
            ]


            sky_texts.extend([
                f"POSITION: X:{plane.position[0]:.0f} Y:{plane.position[1]:.0f} Z:{plane.position[2]:.0f}",
                "",
                "CONTROLS:",
                "UP/DOWN: PITCH",
                "LEFT/RIGHT: ROLL",
                "A/D: YAW",
                "W/S: THROTTLE UP/DOWN",
                "0-5: THROTTLE PRESETS",
                "G: TOGGLE LANDING GEAR",
                "F: CYCLE FLAPS (0/50/100%)",
                "TAB: TOGGLE WIREFRAME",
                "R: RESET POSITION",
                "",
                "TERRAIN: PROCEDURAL INFINITE"
            ])
            
            # Only add time controls if celestial system is available
            if celestial is not None:
                sky_texts.extend([
                    "T: TOGGLE TIME SPEED",
                    "PGUP/PGDN: ADV/REW TIME"
                ])
            
            # Draw status messages
            fixed_text_position = (initial_position[0] + 50.0, initial_position[1] + 50.0, initial_position[2])
            try:
                draw_sky_text(sky_texts, fixed_text_position)
            except Exception as e:
                print(f"Error drawing sky text: {e}")
            
            # If plane is severely damaged, show warning
            if plane.damage > 0.7:
                warning_texts = [
                    "WARNING: AIRCRAFT SEVERELY DAMAGED",
                    "PRESS R TO RESET"
                ]
                try:
                    draw_sky_text(warning_texts, (plane.position[0], plane.position[1] + 10, plane.position[2]))
                except Exception as e:
                    print(f"Error drawing warning text: {e}")
            
            # Display night flying warning when appropriate
            if is_night and not wireframe_mode:
                night_text = [
                    "NIGHT FLIGHT IN PROGRESS",
                    "REDUCED VISIBILITY"
                ]
                try:
                    draw_sky_text(night_text, (plane.position[0], plane.position[1] + 15, plane.position[2] - 5))
                except Exception as e:
                    print(f"Error drawing night text: {e}")
            

            # Update the display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            # Continue the loop instead of crashing
    
    # Clean up resources
    print("Cleaning up resources...")
    try:
        terrain.cleanup()
    except Exception as e:
        print(f"Error cleaning up terrain: {e}")
        
    if trees is not None:
        try:
            trees.cleanup()
        except Exception as e:
            print(f"Error cleaning up trees: {e}")
            
    if birds is not None:
        try:
            birds.cleanup()
        except Exception as e:
            print(f"Error cleaning up birds: {e}")


    if water is not None:
        try:
            water.cleanup()
        except Exception as e:
            print(f"Error cleaning up water: {e}")

    clouds.cleanup()            
    pygame.quit()
    print("Flight simulator terminated.")

if __name__ == "__main__":
    main()






