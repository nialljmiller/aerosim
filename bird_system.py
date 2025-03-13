"""
Bird Flocking System for Flight Simulator
Implements multiple species of birds with distinct flocking behaviors including
V-formations and amorphous murmuration-style clouds.

The implementation utilizes a modified Boids algorithm with species-specific
parameters to govern emergent flocking behaviors and dynamic state transitions.
"""

import numpy as np
import random
import math
from OpenGL.GL import *
from OpenGL.GLU import *

class BirdFlock:
    """
    Manages a flock of birds with species-specific behaviors and rendering.
    Each flock follows modified Boids algorithms with additional parameters
    for formation transitions and environmental response.
    """
    def __init__(self, species_type, initial_position, flock_size=30, terrain=None):
        self.species_type = species_type
        self.terrain = terrain
        self.center_position = np.array(initial_position, dtype=float)
        
        # Initialize flock behaviors
        self.birds = []
        self.flock_size = flock_size
        
        # Define current behavioral state (initialized BEFORE initialize_flock is called)
        self.current_formation = "loose" if random.random() < 0.5 else "v_formation"
        self.formation_transition = 0.0  # Transition value between formations
        self.formation_change_time = 0.0
        
        # Display list for bird models
        self.display_list = None
        self.compiled = False
        
        # Flapping animation
        self.flap_offset = 0.0
        
        # Activation state (only render if within view distance)
        self.active = True
        
        # Bird species parameters
        if species_type == "migratory":  # V-formation birds (geese, cranes)
            self.bird_scale = 0.7
            self.formation_tendency = 0.85
            self.v_formation_strength = 0.9
            self.speed_range = (15.0, 25.0)
            self.altitude_range = (50.0, 120.0)
            self.flock_spacing = 6.0
            self.bird_colors = [(0.6, 0.6, 0.6), (0.7, 0.7, 0.7), (0.75, 0.75, 0.75)]
            self.wing_span_ratio = 3.5
            self.body_len_ratio = 2.0
            self.flap_speed = 2.0
            self.transition_probability = 0.001  # Chance to switch formations
        
        elif species_type == "murmuration":  # Starling-like murmurations
            self.bird_scale = 0.4
            self.formation_tendency = 0.3
            self.v_formation_strength = 0.1
            self.speed_range = (12.0, 20.0)
            self.altitude_range = (30.0, 90.0)
            self.flock_spacing = 2.5
            self.bird_colors = [(0.1, 0.1, 0.1), (0.15, 0.15, 0.15), (0.2, 0.2, 0.2)]
            self.wing_span_ratio = 2.5
            self.body_len_ratio = 1.5
            self.flap_speed = 3.0
            self.transition_probability = 0.002  # Higher chance for dynamic changes
        
        elif species_type == "predator":  # Hawks, eagles
            self.bird_scale = 0.9
            self.formation_tendency = 0.1
            self.v_formation_strength = 0.0
            self.speed_range = (18.0, 30.0)
            self.altitude_range = (40.0, 150.0)
            self.flock_spacing = 15.0
            self.bird_colors = [(0.4, 0.25, 0.1), (0.45, 0.3, 0.15), (0.5, 0.35, 0.2)]
            self.wing_span_ratio = 4.0
            self.body_len_ratio = 2.2
            self.flap_speed = 1.5
            self.transition_probability = 0.01  # Solitary hunters occasionally group up
        
        else:  # Default - small birds (sparrows, finches)
            self.bird_scale = 0.3
            self.formation_tendency = 0.5
            self.v_formation_strength = 0.2
            self.speed_range = (8.0, 15.0)
            self.altitude_range = (20.0, 60.0)
            self.flock_spacing = 2.0
            self.bird_colors = [(0.6, 0.5, 0.3), (0.7, 0.6, 0.4), (0.65, 0.55, 0.35)]
            self.wing_span_ratio = 2.0
            self.body_len_ratio = 1.2
            self.flap_speed = 4.0
            self.transition_probability = 0.003
        
        # Flocking behavior parameters
        self.separation_factor = 1.5
        self.alignment_factor = 1.0
        self.cohesion_factor = 1.0
        self.obstacle_avoidance_factor = 2.0
        self.noise_factor = 0.1
        
        # Initialize bird positions, velocities, and states
        self.initialize_flock()
    
    def initialize_flock(self):
        """Initialize birds with positions and velocities."""
        # Random initial heading for the flock
        heading = random.uniform(0, 2 * math.pi)
        self.flock_direction = np.array([
            math.cos(heading),
            random.uniform(-0.1, 0.1),  # Slight vertical variation
            math.sin(heading)
        ])
        self.flock_direction /= np.linalg.norm(self.flock_direction)
        
        # Initial speed
        self.flock_speed = random.uniform(*self.speed_range)
        
        # Set initial altitude based on species preferences
        target_altitude = random.uniform(*self.altitude_range)
        if self.terrain:
            # Adjust based on terrain height
            terrain_height = self.terrain.get_height(
                self.center_position[0], 
                self.center_position[2]
            )
            self.center_position[1] = max(
                self.center_position[1],
                terrain_height + target_altitude
            )
        else:
            self.center_position[1] = target_altitude
        
        # Create birds
        for i in range(self.flock_size):
            if self.current_formation == "v_formation" and self.formation_tendency > 0.5:
                # Position in V-formation
                offset = self.calculate_v_formation_position(i)
            else:
                # Position in loose formation (random cluster)
                spread = self.flock_spacing * 3
                offset = np.array([
                    random.uniform(-spread, spread),
                    random.uniform(-spread/2, spread/2),
                    random.uniform(-spread, spread)
                ])
            
            # Initial position is center + offset
            position = self.center_position + offset
            
            # Initial velocity based on flock direction and speed
            # Add slight variation
            velocity = self.flock_direction * self.flock_speed
            velocity += np.array([
                random.uniform(-2, 2),
                random.uniform(-1, 1),
                random.uniform(-2, 2)
            ])
            
            # Select random color from available bird colors
            color = random.choice(self.bird_colors)
            
            # Random flap phase offset
            flap_phase = random.uniform(0, 2 * math.pi)
            
            # Add bird to flock
            self.birds.append({
                'position': position,
                'velocity': velocity,
                'acceleration': np.zeros(3),
                'color': color,
                'flap_phase': flap_phase,
                'wing_angle': 0.0
            })
    
    def calculate_v_formation_position(self, index):
        """Calculate a position in a V-formation based on index."""
        if index == 0:
            # Leader at the front
            return np.zeros(3)
        
        # Alternating left-right positions
        side = 1 if index % 2 == 1 else -1
        row = (index + 1) // 2
        
        # Calculate position in V-formation
        spacing = self.flock_spacing
        
        # Forward vector (negative because V opens backward from leader)
        fwd = -self.flock_direction * spacing * row * 1.2
        
        # Side vector (perpendicular to forward and up)
        up = np.array([0, 1, 0])
        side_vec = np.cross(self.flock_direction, up)
        side_vec /= np.linalg.norm(side_vec)
        side_offset = side_vec * side * spacing * row
        
        # Small random variation
        variation = np.array([
            random.uniform(-0.2, 0.2),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.2, 0.2)
        ]) * spacing
        
        return fwd + side_offset + variation
    
    def update(self, dt, player_position, current_time):
        """Update flock behavior and positions."""
        if not self.active:
            return
        
        # Check for formation transitions
        if random.random() < self.transition_probability * dt * 60:
            # Transition between formations
            if self.current_formation == "loose":
                self.current_formation = "v_formation"
            elif self.current_formation == "v_formation":
                self.current_formation = "loose"
            
            self.formation_change_time = current_time
            self.formation_transition = 0.0
        
        # Update formation transition progress
        if self.formation_change_time > 0:
            transition_duration = 10.0  # seconds to complete transition
            elapsed = current_time - self.formation_change_time
            self.formation_transition = min(1.0, elapsed / transition_duration)
        
        # Update flock center position
        self.center_position += self.flock_direction * self.flock_speed * dt
        
        # Gradual course corrections for the flock as a whole
        # Target altitude based on species preference
        target_altitude = random.uniform(*self.altitude_range)
        
        # Get terrain height if available
        terrain_height = 0
        if self.terrain:
            terrain_height = self.terrain.get_height(
                self.center_position[0], 
                self.center_position[2]
            )
        
        # Calculate altitude adjustment
        current_altitude = self.center_position[1] - terrain_height
        altitude_error = target_altitude - current_altitude
        
        # Compute steering vector for the whole flock
        steering = np.zeros(3)
        
        # Altitude adjustment
        steering[1] = altitude_error * 0.1
        
        # Random course variations
        if random.random() < 0.02:
            # Apply small random steering occasionally
            steering[0] += random.uniform(-0.5, 0.5)
            steering[2] += random.uniform(-0.5, 0.5)
        
        # Avoid player aircraft (treat as predator)
        to_player = player_position - self.center_position
        dist_to_player = np.linalg.norm(to_player)
        avoid_radius = 100.0  # Distance to start avoiding player
        
        if dist_to_player < avoid_radius:
            # Calculate avoidance strength based on distance
            avoidance_strength = 1.0 - (dist_to_player / avoid_radius)
            avoidance_strength = min(3.0, avoidance_strength * 5.0)  # Cap and scale
            
            # Direction away from player
            away_dir = -to_player / max(0.1, dist_to_player)
            
            # Apply stronger vertical avoidance to fly above/below the aircraft
            away_dir[1] *= 2.0
            
            # Add avoidance to steering
            steering += away_dir * avoidance_strength
        
        # Apply steering to gradually change flock direction
        steering_strength = 0.5 * dt
        self.flock_direction += steering * steering_strength
        self.flock_direction /= np.linalg.norm(self.flock_direction)
        
        # Randomly adjust flock speed
        if random.random() < 0.05:
            # Gradually change speed
            target_speed = random.uniform(*self.speed_range)
            self.flock_speed += (target_speed - self.flock_speed) * 0.05
        
        # Update individual bird positions and velocities using boid algorithm
        for i, bird in enumerate(self.birds):
            # Calculate target position based on current formation
            if self.current_formation == "v_formation":
                target_offset = self.calculate_v_formation_position(i)
                target_position = self.center_position + target_offset
                formation_factor = self.v_formation_strength * (0.5 + 0.5 * self.formation_transition)
            else:  # Loose formation
                # Looser formation uses boid rules
                formation_factor = 0.2 * (1.0 - self.formation_transition)
                # Still provide a loose target to maintain flock coherence
                spread = self.flock_spacing * 5
                target_offset = np.array([
                    random.uniform(-spread, spread),
                    random.uniform(-spread/2, spread/2),
                    random.uniform(-spread, spread)
                ])
                target_position = self.center_position + target_offset
            
            # Vector to target position
            to_target = target_position - bird['position']
            target_dist = np.linalg.norm(to_target)
            
            # Calculate steering forces
            steering = np.zeros(3)
            
            # 1. Formation steering - pull toward formation position
            if target_dist > 0.1:
                formation_steering = to_target / target_dist
                formation_steering *= min(target_dist * 2.0, self.flock_speed) * formation_factor
                steering += formation_steering
            
            # Apply standard boid rules for the remaining steering
            # 2. Separation - avoid collision with flockmates
            separation = np.zeros(3)
            separation_count = 0
            
            for j, other in enumerate(self.birds):
                if i != j:
                    to_other = bird['position'] - other['position']
                    dist = np.linalg.norm(to_other)
                    min_dist = self.flock_spacing * self.separation_factor
                    
                    if dist < min_dist and dist > 0:
                        # Strength increases as distance decreases
                        strength = (min_dist - dist) / min_dist
                        separation += (to_other / dist) * strength
                        separation_count += 1
            
            if separation_count > 0:
                separation /= separation_count
                separation *= self.flock_speed
                steering += separation * self.separation_factor * (1.0 - formation_factor * 0.5)
            
            # 3. Alignment - align with flockmates' velocity
            alignment = np.zeros(3)
            alignment_count = 0
            alignment_radius = self.flock_spacing * 3
            
            for j, other in enumerate(self.birds):
                if i != j:
                    dist = np.linalg.norm(bird['position'] - other['position'])
                    if dist < alignment_radius:
                        alignment += other['velocity']
                        alignment_count += 1
            
            if alignment_count > 0:
                alignment /= alignment_count
                alignment = alignment - bird['velocity']
                steering += alignment * self.alignment_factor * (1.0 - formation_factor)
            
            # 4. Cohesion - stay with the flock
            cohesion = self.center_position - bird['position']
            dist_to_center = np.linalg.norm(cohesion)
            
            if dist_to_center > self.flock_spacing * 4:
                cohesion = cohesion / dist_to_center * self.flock_speed
                cohesion = cohesion - bird['velocity']
                steering += cohesion * self.cohesion_factor * (1.0 - formation_factor * 0.5)
            
            # 5. Avoid terrain if we're getting too close
            if self.terrain:
                ground_height = self.terrain.get_height(
                    bird['position'][0], 
                    bird['position'][2]
                )
                height_above_ground = bird['position'][1] - ground_height
                min_safe_height = 10.0
                
                if height_above_ground < min_safe_height:
                    # Strong upward steering to avoid ground
                    ground_avoidance = np.array([0, min_safe_height - height_above_ground, 0])
                    steering += ground_avoidance * self.obstacle_avoidance_factor
            
            # 6. Add some random noise
            noise = np.array([
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ])
            steering += noise * self.noise_factor
            
            # Apply steering with smoothing
            max_force = 20.0  # Maximum steering force
            steering = np.clip(steering, -max_force, max_force)
            
            # Apply acceleration
            bird['acceleration'] = steering
            
            # Update velocity
            bird['velocity'] += bird['acceleration'] * dt
            
            # Limit speed
            speed = np.linalg.norm(bird['velocity'])
            if speed > self.flock_speed:
                bird['velocity'] = bird['velocity'] / speed * self.flock_speed
            
            # Update position
            bird['position'] += bird['velocity'] * dt
            
            # Update wing flapping
            bird['wing_angle'] = math.sin(current_time * self.flap_speed + bird['flap_phase']) * 0.5
        
        # Update flock center based on actual bird positions
        if len(self.birds) > 0:
            new_center = np.zeros(3)
            for bird in self.birds:
                new_center += bird['position']
            new_center /= len(self.birds)
            
            # Smooth transition to calculated center
            self.center_position = self.center_position * 0.95 + new_center * 0.05
    
    def compile_display_list(self):
        """Compile bird model display list for efficient rendering."""
        if self.compiled:
            return
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Draw bird model at origin, aligned along X-axis
        # Scale based on species
        scale = self.bird_scale
        wing_span = scale * self.wing_span_ratio
        body_length = scale * self.body_len_ratio
        
        # Bird body (elongated ellipsoid)
        glPushMatrix()
        glScalef(body_length, scale * 0.6, scale * 0.6)
        self._draw_ellipsoid(16, 8)
        glPopMatrix()
        
        # Wings represented as triangular shapes
        glBegin(GL_TRIANGLES)
        # Right wing
        glVertex3f(0, 0, 0)  # Wing root
        glVertex3f(-body_length * 0.3, 0, wing_span)  # Wing tip
        glVertex3f(-body_length * 0.6, 0, 0)  # Wing trailing edge
        
        # Left wing
        glVertex3f(0, 0, 0)  # Wing root
        glVertex3f(-body_length * 0.3, 0, -wing_span)  # Wing tip
        glVertex3f(-body_length * 0.6, 0, 0)  # Wing trailing edge
        glEnd()
        
        # Tail
        glBegin(GL_TRIANGLES)
        glVertex3f(-body_length, 0, 0)  # Tail root
        glVertex3f(-body_length * 1.3, 0, scale * 0.5)  # Right tail
        glVertex3f(-body_length * 1.3, 0, -scale * 0.5)  # Left tail
        glEnd()
        
        # Head (small sphere)
        glPushMatrix()
        glTranslatef(body_length * 0.8, scale * 0.1, 0)
        glScalef(scale * 0.5, scale * 0.5, scale * 0.5)
        self._draw_ellipsoid(8, 4)
        glPopMatrix()
        
        glEndList()
        self.compiled = True
    
    def _draw_ellipsoid(self, slices, stacks):
        """Helper to draw an ellipsoid."""
        for i in range(stacks):
            phi1 = (i / stacks) * math.pi
            phi2 = ((i + 1) / stacks) * math.pi
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                theta = (j / slices) * 2 * math.pi
                
                # First vertex
                x = math.sin(phi1) * math.cos(theta)
                y = math.cos(phi1)
                z = math.sin(phi1) * math.sin(theta)
                glVertex3f(x, y, z)
                
                # Second vertex
                x = math.sin(phi2) * math.cos(theta)
                y = math.cos(phi2)
                z = math.sin(phi2) * math.sin(theta)
                glVertex3f(x, y, z)
            glEnd()
    
    def draw(self, camera_position):
        """Render all birds in the flock."""
        if not self.active:
            return
        
        # Distance-based culling
        dist_to_camera = np.linalg.norm(self.center_position - camera_position)
        if dist_to_camera > 1000:
            return
        
        # Ensure display list is compiled
        if not self.compiled:
            self.compile_display_list()
        
        # Draw each bird
        for bird in self.birds:
            # Distance-based culling for individual birds
            bird_to_camera = np.linalg.norm(bird['position'] - camera_position)
            if bird_to_camera > 800:
                continue
            
            # Simplified LOD - only draw every 2nd or 3rd bird at distance
            if bird_to_camera > 400 and random.random() < 0.5:
                continue
            
            # Setup rendering
            glPushMatrix()
            
            # Position
            glTranslatef(bird['position'][0], bird['position'][1], bird['position'][2])
            
            # Orientation - align with velocity direction
            # Calculate rotation to align bird with its velocity
            forward = bird['velocity'] / max(0.1, np.linalg.norm(bird['velocity']))
            
            # Calculate rotation axis and angle to rotate from default X-axis to velocity vector
            default_dir = np.array([1.0, 0.0, 0.0])
            rotation_axis = np.cross(default_dir, forward)
            
            # Avoid zero-length rotation axis
            if np.linalg.norm(rotation_axis) > 0.001:
                rotation_axis /= np.linalg.norm(rotation_axis)
                dot = np.dot(default_dir, forward)
                angle = math.acos(max(-1.0, min(1.0, dot))) * 180.0 / math.pi
                
                glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            
            # Wing flapping - rotate around Z-axis
            glPushMatrix()
            # Right wing
            glRotatef(bird['wing_angle'] * 45.0, 0, 0, 1)
            
            # Set color
            glColor3f(*bird['color'])
            
            # Draw bird model
            glCallList(self.display_list)
            glPopMatrix()
            
            glPopMatrix()
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.compiled and self.display_list:
            glDeleteLists(self.display_list, 1)
            self.compiled = False


class BirdSystem:
    """
    Manages multiple flocks of birds with different behaviors and species.
    Handles spawning, updating, and rendering of all bird flocks.
    """
    def __init__(self, terrain, max_flocks=15):
        self.terrain = terrain
        self.max_flocks = max_flocks
        self.flocks = []
        
        # Bird species and their probabilities
        self.species = {
            "migratory": 0.3,     # V-formation geese/cranes
            "murmuration": 0.25,   # Starling-like swarms
            "predator": 0.15,      # Hawks, eagles (small groups or solo)
            "default": 0.3         # Small birds in loose flocks
        }
        
        # Flock size ranges by species
        self.flock_sizes = {
            "migratory": (15, 35),
            "murmuration": (40, 120),
            "predator": (1, 4),
            "default": (10, 25)
        }
        
        # Active area for bird spawning/despawning
        self.active_radius = 1500.0
        self.spawn_cooldown = 0.0
        
        # Initialize with some flocks
        self.spawn_initial_flocks()
    
    def spawn_initial_flocks(self):
        """Create initial flocks distributed around the map."""
        initial_count = self.max_flocks // 2
        
        for _ in range(initial_count):
            # Random position within a large area
            radius = random.uniform(500, 1000)
            angle = random.uniform(0, 2 * math.pi)
            
            pos_x = radius * math.cos(angle)
            pos_z = radius * math.sin(angle)
            
            # Get terrain height as base altitude
            ground_y = 0
            if self.terrain:
                ground_y = self.terrain.get_height(pos_x, pos_z)
            
            # Position above ground
            min_altitude = 50
            max_altitude = 150
            pos_y = ground_y + random.uniform(min_altitude, max_altitude)
            
            self.spawn_flock((pos_x, pos_y, pos_z))
    
    def spawn_flock(self, position):
        """Spawn a new random flock at the given position."""
        if len(self.flocks) >= self.max_flocks:
            return None
        
        # Choose species based on probability distribution
        species_choice = random.random()
        cumulative_prob = 0
        selected_species = "default"
        
        for species, prob in self.species.items():
            cumulative_prob += prob
            if species_choice <= cumulative_prob:
                selected_species = species
                break
        
        # Determine flock size based on species
        min_size, max_size = self.flock_sizes[selected_species]
        flock_size = random.randint(min_size, max_size)
        
        # Create the flock
        new_flock = BirdFlock(selected_species, position, flock_size, self.terrain)
        self.flocks.append(new_flock)
        return new_flock
    
    def update(self, dt, player_position, current_time):
        """Update all bird flocks."""
        # Update spawn cooldown
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= dt
        
        # Check for flocks that have moved too far away and remove them
        flocks_to_remove = []
        
        for flock in self.flocks:
            distance = np.linalg.norm(np.array(flock.center_position) - np.array(player_position))
            
            if distance > self.active_radius * 1.5:
                flocks_to_remove.append(flock)
            else:
                # Activate/deactivate based on distance for performance
                flock.active = distance < self.active_radius
        
        # Remove far-away flocks
        for flock in flocks_to_remove:
            if flock in self.flocks:
                self.flocks.remove(flock)
        
        # Spawn new flocks occasionally
        if (len(self.flocks) < self.max_flocks and 
            self.spawn_cooldown <= 0 and 
            random.random() < 0.01):
            
            # Spawn at edge of active area in random direction
            spawn_angle = random.uniform(0, 2 * math.pi)
            spawn_x = player_position[0] + math.cos(spawn_angle) * self.active_radius
            spawn_z = player_position[2] + math.sin(spawn_angle) * self.active_radius
            
            # Get terrain height
            ground_y = 0
            if self.terrain:
                ground_y = self.terrain.get_height(spawn_x, spawn_z)
            
            # Random altitude based on terrain
            altitude = ground_y + random.uniform(80, 200)
            
            # Spawn new flock
            self.spawn_flock((spawn_x, altitude, spawn_z))
            
            # Set cooldown to prevent spawning too many flocks at once
            self.spawn_cooldown = random.uniform(5.0, 15.0)
        
        # Update active flocks
        for flock in self.flocks:
            if flock.active:
                flock.update(dt, player_position, current_time)
    
    def draw(self, camera_position):
        """Render all active bird flocks."""
        for flock in self.flocks:
            if flock.active:
                flock.draw(camera_position)
    
    def cleanup(self):
        """Free OpenGL resources."""
        for flock in self.flocks:
            flock.cleanup()
        self.flocks.clear()