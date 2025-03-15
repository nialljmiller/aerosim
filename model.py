import numpy as np
import math

class Plane3D:
    def __init__(self, position, velocity, forward, up, params):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.forward = np.array(forward, dtype=float)  # Plane's forward direction
        self.forward /= np.linalg.norm(self.forward)
        self.up = np.array(up, dtype=float)            # Plane's up direction
        self.up /= np.linalg.norm(self.up)
        self.right = np.cross(self.forward, self.up)     # Right vector
        self.right /= np.linalg.norm(self.right)
        self.params = params
        
        # Add throttle control (0.0 to 1.0)
        self.throttle = 0.3  # Start at 30% throttle
        # Maximum thrust when throttle is at 100%
        self.max_thrust = params['propeller_thrust']
        
        # Energy and glide characteristics
        self.energy = 100.0  # Energy reserve for more realistic gliding (100%)
        self.glide_efficiency = params['glide_ratio']  # How far the plane can glide horizontally per unit of altitude lost
        
        # Wing geometry factors - derived from the wing_area parameter
        self.wing_span = math.sqrt(params['wing_area'] * 5)  # Width of the wings
        self.wing_chord = params['wing_area'] / self.wing_span  # Length of the wings back from leading edge
        
        # Fuselage dimensions based on wing size
        self.fuselage_length = self.wing_chord * 3.5
        self.fuselage_height = self.wing_chord * 0.8
        self.fuselage_width = self.wing_chord * 0.6
        
        # Landing gear state (0: retracted, 1: extended)
        self.gear_state = 1.0
        self.gear_transition_speed = 1.0  # Speed of gear extension/retraction
        self.gear_target = 1.0  # Target gear state
        
        # Landing gear dimensions
        self.wheel_radius = self.fuselage_height * 0.25
        self.gear_height = self.wheel_radius * 1.5
        self.gear_compression = 0.0  # Landing gear shock absorption (0-1)
        
        # Track gear compression for each wheel independently
        self.wheel_compression = {
            'left': 0.0, 
            'right': 0.0, 
            'nose': 0.0,
            'tail': 0.0  # Some small planes have a tail wheel
        }
        
        # Flaps state (0: retracted, 1: fully extended)
        self.flaps = 0.0
        self.flaps_transition_speed = 0.5  # Speed of flap extension/retraction
        self.flaps_target = 0.0  # Target flaps state
        
        # Collision state tracking
        self.is_grounded = False
        self.collision_points = []
        self.ground_normal = np.array([0, 1, 0])
        
        # Damage tracking
        self.damage = 0.0  # From 0 (no damage) to 1.0 (destroyed)
        self.last_collision_time = 0.0
        
        # Define points for collision detection including wheels
        self.update_collision_points()

    def update_collision_points(self):
        """Update all collision detection points based on current plane dimensions."""
        # Wing tips and control surfaces
        wing_tip_dist = self.wing_span / 2
        tail_width = self.wing_span * 0.4
        self.wing_points = [
            np.array([0, 0, -wing_tip_dist]),                      # Left wing tip
            np.array([0, 0, wing_tip_dist]),                       # Right wing tip
            np.array([-self.fuselage_length * 0.8, 0, -tail_width/2]), # Left horizontal stabilizer
            np.array([-self.fuselage_length * 0.8, 0, tail_width/2]),  # Right horizontal stabilizer
            np.array([self.fuselage_length/2, 0, 0]),              # Nose
            np.array([-self.fuselage_length/2, 0, 0]),             # Tail
        ]
        
        # Landing gear positions (only for collision when extended)
        self.wheel_points = {
            'left': np.array([0, -self.gear_height, -self.wing_span * 0.3]),  # Left wheel
            'right': np.array([0, -self.gear_height, self.wing_span * 0.3]),  # Right wheel
            'nose': np.array([self.fuselage_length * 0.3, -self.gear_height, 0]),  # Nose wheel
            'tail': np.array([-self.fuselage_length * 0.7, -self.gear_height * 0.5, 0])  # Tail wheel (smaller)
        }

    def toggle_landing_gear(self):
        """Toggle landing gear between retracted and extended states."""
        if self.gear_target < 0.5:
            self.gear_target = 1.0
        else:
            self.gear_target = 0.0

    def set_flaps(self, position):
        """Set flaps position (0.0 to 1.0)."""
        self.flaps_target = max(0.0, min(1.0, position))

    def apply_controls(self, pitch, roll, yaw, dt):
        """Apply pitch, roll, and yaw controls with ground constraints."""
        # If grounded, restrict movements
        if self.is_grounded:
            # Prevent rolling when on ground
            roll = 0.0
            
            # Limit pitch when on ground
            if pitch < 0:  # Prevent nose-down when on ground
                pitch = 0.0
                
            # Modify yaw effect when on ground (simulate wheel steering)
            # More effective steering with the nose wheel
            yaw *= 0.5 * (1.0 + self.wheel_compression['nose'] * 2.0)
        
        # Apply pitch: rotate around right vector
        self.forward = rotate_vector(self.forward, self.right, pitch * dt)
        self.up = rotate_vector(self.up, self.right, pitch * dt)
        
        # Apply roll: rotate around forward vector
        self.up = rotate_vector(self.up, self.forward, roll * dt)
        self.right = rotate_vector(self.right, self.forward, roll * dt)
        
        # Apply yaw: rotate around up vector
        self.forward = rotate_vector(self.forward, self.up, yaw * dt)
        self.right = rotate_vector(self.right, self.up, yaw * dt)
        
        # Add coordinated turn effect: banking naturally causes turning
        if not self.is_grounded:  # Only apply in air
            world_up = np.array([0, 1, 0])
            bank_angle = np.arccos(np.clip(np.dot(self.up, world_up), -1.0, 1.0))
            bank_direction = np.sign(np.dot(np.cross(world_up, self.up), self.forward))
            
            # Coordinated turn effect is stronger with flaps extended
            flaps_turn_factor = 1.0 + self.flaps * 0.2
            turn_rate = 0.5 * bank_angle * bank_direction * flaps_turn_factor
            
            self.forward = rotate_vector(self.forward, self.up, turn_rate * dt)
            self.right = rotate_vector(self.right, self.up, turn_rate * dt)
        
        # Re-orthonormalize basis vectors
        self.forward /= np.linalg.norm(self.forward)
        self.up /= np.linalg.norm(self.up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)

    def get_world_point(self, local_point):
        """Convert a point from local plane coordinates to world coordinates."""
        rotation_matrix = np.array([
            self.forward,
            self.up,
            self.right
        ]).T
        
        return self.position + rotation_matrix @ local_point
        
    def update_landing_gear(self, dt):
        """Update landing gear state and animation."""
        # Update gear state
        if self.gear_state != self.gear_target:
            if self.gear_state < self.gear_target:
                self.gear_state = min(self.gear_state + self.gear_transition_speed * dt, self.gear_target)
            else:
                self.gear_state = max(self.gear_state - self.gear_transition_speed * dt, self.gear_target)
        
        # Update flaps state
        if self.flaps != self.flaps_target:
            if self.flaps < self.flaps_target:
                self.flaps = min(self.flaps + self.flaps_transition_speed * dt, self.flaps_target)
            else:
                self.flaps = max(self.flaps - self.flaps_transition_speed * dt, self.flaps_target)
        
        # Gradually decompress landing gear when airborne
        if not self.is_grounded:
            for wheel in self.wheel_compression:
                self.wheel_compression[wheel] *= 0.95  # Gradual decompression

    def check_wheel_collisions(self, terrain, dt):
        """Check for collisions specifically with the landing gear."""
        # Reset wheel compressions if we're not already grounded
        if not self.is_grounded:
            for wheel in self.wheel_compression:
                self.wheel_compression[wheel] *= 0.9
        
        # Only check for wheel collisions if gear is at least partially down
        if self.gear_state < 0.1:
            return False
        
        any_wheel_contact = False
        
        # Check each wheel for collision
        for wheel_name, local_point in self.wheel_points.items():
            # Adjust wheel position based on gear state (retracted/extended)
            adjusted_point = local_point.copy()
            if wheel_name == 'nose' or wheel_name == 'tail':
                # Nose and tail wheels retract differently
                adjusted_point[1] = local_point[1] * self.gear_state - (1.0 - self.gear_state) * self.fuselage_height * 0.3
            else:
                # Main gear retracts into wings/fuselage
                adjusted_point[1] = local_point[1] * self.gear_state - (1.0 - self.gear_state) * self.fuselage_height * 0.2
            
            # Convert to world coordinates
            world_point = self.get_world_point(adjusted_point)
            
            # Get terrain height at this point
            terrain_height = terrain.get_height(world_point[0], world_point[2])
            
            # Check if wheel is below terrain
            wheel_height = world_point[1] - self.wheel_radius + self.wheel_compression[wheel_name] * self.wheel_radius
            if wheel_height < terrain_height:
                # Record collision point and calculate compression
                penetration = terrain_height - wheel_height
                normal = terrain.get_terrain_normal(world_point[0], world_point[2])
                
                # Maximum compression is the wheel radius
                max_compression = self.wheel_radius * 1.5
                compression = min(penetration / max_compression, 1.0)
                
                # Update wheel compression (with damping to prevent oscillation)
                self.wheel_compression[wheel_name] = max(self.wheel_compression[wheel_name], compression)
                
                # Mark as grounded when any wheel has contact
                any_wheel_contact = True
                
                # Add to collision points with special wheel flag
                self.collision_points.append({
                    'wheel': wheel_name,
                    'world_point': world_point,
                    'penetration': penetration,
                    'normal': normal,
                    'compression': compression
                })
                
                # Set ground normal from wheel contact (average if multiple wheels)
                if self.is_grounded:
                    self.ground_normal = (self.ground_normal + normal) / 2
                    self.ground_normal /= np.linalg.norm(self.ground_normal)
                else:
                    self.ground_normal = normal
        
        # Update grounded state based on wheel contact
        self.is_grounded = any_wheel_contact or self.is_grounded
        
        return any_wheel_contact

    def check_terrain_collisions(self, terrain, dt):
        """Check if any part of the plane is colliding with the terrain."""
        self.collision_points = []
        old_grounded = self.is_grounded
        self.is_grounded = False
        
        # First check wheel collisions which update self.is_grounded separately
        wheel_collision = self.check_wheel_collisions(terrain, dt)
        
        # Check each wing point for collision
        for point_idx, local_point in enumerate(self.wing_points):
            # Convert to world coordinates
            world_point = self.get_world_point(local_point)
            
            # Get terrain height at this point
            terrain_height = terrain.get_height(world_point[0], world_point[2])
            
            # Check if point is below terrain
            if world_point[1] < terrain_height:
                # Record collision point and penetration depth
                penetration = terrain_height - world_point[1]
                normal = terrain.get_terrain_normal(world_point[0], world_point[2])
                
                self.collision_points.append({
                    'point_idx': point_idx,
                    'world_point': world_point,
                    'penetration': penetration,
                    'normal': normal
                })
                
                # If nose or central body collides, mark as grounded and damaged
                if point_idx in [4, 5]:  # Nose or tail
                    self.is_grounded = True
                    self.ground_normal = normal
        
        # Handle collisions if any
        if self.collision_points:
            self.resolve_collisions(dt)
            return True
        
        return wheel_collision

    def resolve_collisions(self, dt):
        """Resolve terrain collisions with appropriate physics response."""
        if not self.collision_points:
            return
        
        # Get current time for collision damage calculation
        try:
            import pygame
            current_time = pygame.time.get_ticks() / 1000.0
        except:
            current_time = 0
            
        collision_response = np.zeros(3)
        collision_count = 0
        wheel_collision_count = 0
        
        # Process wheel collisions separately
        for collision in self.collision_points:
            if 'wheel' in collision:
                # This is a wheel collision, handle with wheel physics
                wheel_name = collision['wheel']
                normal = collision['normal']
                penetration = collision['penetration']
                compression = collision['compression']
                
                # Add collision response force with wheel suspension characteristics
                suspension_strength = 100.0  # Spring constant
                damping = 5.0  # Damping factor
                
                # Calculate suspension force - stronger for higher compression
                suspension_force = penetration * suspension_strength
                
                # Apply the suspension force along the normal direction
                wheel_response = normal * suspension_force / max(1, wheel_collision_count)
                
                # Add wheel response to total collision response
                collision_response += wheel_response
                wheel_collision_count += 1
                
                # Apply friction in the horizontal plane for wheel physics
                impact_speed = -np.dot(self.velocity, normal)
                
                # Calculate how much of the velocity is perpendicular to the wheel
                perpendicular_vel = self.velocity - normal * impact_speed
                perp_speed = np.linalg.norm(perpendicular_vel)
                
                if perp_speed > 0.1:
                    # Apply rolling resistance based on wheel compression
                    rolling_friction = 0.05 * compression
                    
                    # For nose wheel, apply stronger braking to simulate nose wheel steering
                    if wheel_name == 'nose':
                        rolling_friction *= 2.0
                        
                    # Apply friction to horizontal velocity
                    friction_decel = min(perp_speed, rolling_friction * perp_speed)
                    self.velocity -= (perpendicular_vel / perp_speed) * friction_decel * dt
            else:
                # Handle non-wheel collisions (body parts hitting ground)
                penetration = collision['penetration']
                normal = collision['normal']
                point_idx = collision['point_idx']
                
                # Add collision response force
                response_strength = penetration * 50.0  # Proportional to penetration depth
                collision_response += normal * response_strength
                collision_count += 1
                
                # Calculate collision speed (dot product of velocity and normal)
                impact_speed = -np.dot(self.velocity, normal)
                
                # Apply damage if impact is significant and time from last collision is sufficient
                if impact_speed > 10.0 and current_time - self.last_collision_time > 0.5:
                    damage_factor = (impact_speed - 10.0) / 40.0  # Scale damage (max at 50 m/s)
                    self.damage += min(0.2, damage_factor)  # Cap single collision damage
                    self.damage = min(1.0, self.damage)  # Cap total damage
                    self.last_collision_time = current_time
                    
                    # Add angular impulse for realistic crash behavior
                    if point_idx in [0, 1]:  # Wing tips
                        # Add roll effect when wing hits ground
                        roll_dir = 1 if point_idx == 0 else -1  # Left or right wing
                        self.up = rotate_vector(self.up, self.forward, roll_dir * 0.1 * impact_speed * dt)
                        self.right = np.cross(self.forward, self.up)
                    elif point_idx == 4:  # Nose
                        # Add pitch effect when nose hits ground
                        self.forward = rotate_vector(self.forward, self.right, -0.1 * impact_speed * dt)
                        self.up = rotate_vector(self.up, self.right, -0.1 * impact_speed * dt)
        
        # Apply position correction to move plane above ground
        if collision_response.any():
            # Different response for wheels vs body collisions
            if wheel_collision_count > 0:
                # Wheels touching - gentler correction to simulate suspension
                self.position += collision_response * dt * 2 / wheel_collision_count
                
                # For wheels, only damp vertical velocity when it's downward
                if self.velocity[1] < 0:
                    # Ground friction only applied to horizontal components when wheels are down
                    self.velocity[0] *= 0.95
                    self.velocity[2] *= 0.95
                    
                    # Apply suspension damping to vertical velocity
                    damping_factor = 0.8  # Higher = more damping
                    self.velocity[1] *= (1.0 - damping_factor * dt * 10)
                    
                    # If very low vertical velocity, zero it out to prevent bouncing
                    if abs(self.velocity[1]) < 0.1:
                        self.velocity[1] = 0
            else:
                # Body parts hitting - stronger correction to prevent clipping
                self.position += collision_response * dt * 10 / max(1, collision_count)
                
                # For non-wheel collisions, apply a bounce with energy loss
                normal_vel = np.dot(self.velocity, collision_response)
                if normal_vel < 0:  # Only bounce if moving toward the surface
                    # Reflect velocity with damping
                    norm_dir = collision_response / np.linalg.norm(collision_response)
                    reflection = self.velocity - 1.8 * normal_vel * norm_dir
                    energy_loss = 0.5  # 50% energy lost in crash collision
                    self.velocity = reflection * (1.0 - energy_loss)

    def update(self, dt, terrain=None):
        """Physics update with improved flight dynamics and terrain collision."""
        # Update landing gear and flaps
        self.update_landing_gear(dt)
        
        # Calculate thrust along the forward direction
        actual_thrust = self.max_thrust * self.throttle
        thrust_force = actual_thrust * self.forward
        
        # Calculate current speed
        v = np.linalg.norm(self.velocity)
        
        # Adjust lift coefficient based on flaps setting
        flaps_lift_factor = 1.0 + self.flaps * 0.5  # Up to 50% more lift with full flaps
        flaps_drag_factor = 1.0 + self.flaps * 0.8  # Up to 80% more drag with full flaps
        
        # Landing gear drag
        gear_drag_factor = 1.0 + self.gear_state * 0.25  # Up to 25% more drag with gear down
        
        # Improved aerodynamics for better gliding
        # Adjust the lift coefficient based on angle of attack
        if v > 0.1:  # Only calculate if we have meaningful velocity
            airflow_direction = -self.velocity / v
            
            # Calculate angle of attack (angle between airflow and forward direction)
            angle_of_attack_cos = np.dot(airflow_direction, self.forward)
            angle_of_attack = np.arccos(np.clip(angle_of_attack_cos, -1.0, 1.0))
            
            # Adjust lift coefficient based on angle of attack
            # With flaps, peak angle of attack is higher (improved slow-speed performance)
            peak_aoa = 0.26 + self.flaps * 0.1  # radians (15 degrees + up to ~6 more degrees with flaps)
            angle_factor = 1.0 - abs(angle_of_attack - peak_aoa) * (2.0 / (1.0 + self.flaps * 0.5))
            
            # Apply both flaps and angle of attack effects to lift
            effective_lift_coefficient = self.params['lift_coefficient'] * max(0.1, min(1.5, angle_factor)) * flaps_lift_factor
        else:
            effective_lift_coefficient = self.params['lift_coefficient'] * flaps_lift_factor
        
        # Calculate drag with improved model for gliding
        if v > 0.01:
            # Calculate induced drag (due to lift) and parasitic drag (due to shape)
            # Higher lift = higher induced drag
            induced_drag_coefficient = (effective_lift_coefficient ** 2) / (3.14159 * 6.0)
            total_drag_coefficient = self.params['drag_coefficient'] * flaps_drag_factor * gear_drag_factor + induced_drag_coefficient
            
            drag_mag = 0.5 * self.params['air_density'] * v**2 * total_drag_coefficient * self.params['wing_area']
            drag_force = -drag_mag * (self.velocity / v)
        else:
            drag_force = np.zeros(3)
        
        # Calculate lift perpendicular to forward direction
        if v > 1.0:  # Only calculate meaningful lift when we have speed
            airflow_direction = -self.velocity / v
            
            # Calculate what lift direction should be (perpendicular to airflow and right vector)
            lift_direction = np.cross(self.right, airflow_direction)
            lift_direction = lift_direction / np.linalg.norm(lift_direction)
            
            # Calculate lift force magnitude
            lift_mag = 0.5 * self.params['air_density'] * v**2 * self.params['wing_area'] * effective_lift_coefficient
            
            # Apply lift in the correct direction
            lift_force = lift_mag * lift_direction
        else:
            lift_force = np.zeros(3)
        
        # Gravity force
        gravity_force = np.array([0.0, -self.params['mass'] * self.params['gravity'], 0.0])
        
        # Calculate total force and acceleration
        total_force = thrust_force + drag_force + lift_force + gravity_force
        acceleration = total_force / self.params['mass']
        
        # Update velocity with physics forces
        self.velocity += acceleration * dt
        
        # Enhanced aircraft alignment physics with stronger effect
        if v > 1.0:  # Only apply when moving at a reasonable speed
            # Calculate how aligned velocity is with forward direction
            forward_component = np.dot(self.velocity, self.forward)
            
            # Calculate the side component (velocity perpendicular to forward direction)
            side_velocity = self.velocity - (self.forward * forward_component)
            side_speed = np.linalg.norm(side_velocity)
            
            # Add aerodynamic resistance to sideways motion
            side_resistance = 0.25 * side_speed * side_speed
            
            # Apply side resistance (stronger at higher speeds)
            if side_speed > 0.01:
                self.velocity -= (side_velocity / side_speed) * side_resistance * dt
                
            # Automatic aircraft orientation alignment with velocity (less with flaps down)
            if v > 5.0 and not self.is_grounded:
                # Reduce alignment with flaps extended (more manual control in landing config)
                flaps_alignment_reduction = 1.0 - self.flaps * 0.5
                
                # Calculate how misaligned the plane is with its velocity
                alignment = np.dot(self.forward, self.velocity) / (np.linalg.norm(self.forward) * v)
                
                # Factor controlling how quickly the plane aligns with velocity
                alignment_factor = 0.8 * (1.0 - self.throttle * 0.5) * flaps_alignment_reduction
                
                if alignment < 0.98:  # Only apply if not already well-aligned
                    # Calculate rotation axis (perpendicular to both forward and velocity)
                    vel_normalized = self.velocity / v
                    rotation_axis = np.cross(self.forward, vel_normalized)
                    
                    if np.linalg.norm(rotation_axis) > 0.01:  # Check if axis is valid
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        
                        # Calculate rotation angle based on misalignment
                        rotation_angle = alignment_factor * (1.0 - alignment) * dt * 5.0
                        
                        # Apply rotation to both forward and up vectors
                        self.forward = rotate_vector(self.forward, rotation_axis, rotation_angle)
                        self.up = rotate_vector(self.up, rotation_axis, rotation_angle)
                        
                        # Ensure vectors stay orthonormal
                        self.forward = self.forward / np.linalg.norm(self.forward)
                        self.right = np.cross(self.forward, self.up)
                        self.right = self.right / np.linalg.norm(self.right)
                        self.up = np.cross(self.right, self.forward)
                        self.up = self.up / np.linalg.norm(self.up)
                        
            # Additional pitch-down effect during low throttle (reduced with flaps)
            if self.throttle < 0.2 and self.velocity[1] > -5.0 and not self.is_grounded:
                # Flaps help maintain lift at low speeds, reducing pitch-down tendency
                flaps_pitch_reduction = 1.0 - self.flaps * 0.8
                
                pitch_down_factor = (0.2 - self.throttle) * 0.5 * flaps_pitch_reduction
                pitch_down_axis = self.right  # Pitch around right vector (wings)
                pitch_down_angle = pitch_down_factor * dt
                
                # Apply subtle pitch down rotation
                self.forward = rotate_vector(self.forward, pitch_down_axis, -pitch_down_angle)
                self.up = rotate_vector(self.up, pitch_down_axis, -pitch_down_angle)
        
        # Enhanced gliding physics, modified by flaps
        if self.throttle < 0.1 and v > 5.0 and self.velocity[1] < 0 and not self.is_grounded:
            # Calculate current glide ratio (horizontal distance / vertical drop)
            horizontal_velocity = np.array([self.velocity[0], 0, self.velocity[2]])
            horizontal_speed = np.linalg.norm(horizontal_velocity)
            vertical_speed = abs(self.velocity[1])  # Should be negative when descending
            
            # Calculate current and target glide ratio
            # Flaps decrease glide ratio for steeper but more controlled descent
            flaps_glide_reduction = 1.0 - self.flaps * 0.3
            current_glide_ratio = horizontal_speed / max(0.1, vertical_speed)
            target_glide_ratio = self.glide_efficiency * flaps_glide_reduction
            
            # Use a gentler correction that doesn't create sudden speed boosts
            if current_glide_ratio < target_glide_ratio:
                # We're descending too steeply - add a bit of forward force to flatten glide
                glide_correction = min(0.05, (target_glide_ratio - current_glide_ratio) * 0.02)
                
                # Use the forward direction for the correction
                forward_horizontal = np.array([self.forward[0], 0, self.forward[2]])
                if np.linalg.norm(forward_horizontal) > 0.01:
                    forward_horizontal = forward_horizontal / np.linalg.norm(forward_horizontal)
                    # Add a gentle corrective force in the forward direction
                    correction_force = glide_correction * forward_horizontal * dt * v
                    self.velocity += correction_force
            
            # If we're gliding too flat, increase descent rate
            elif current_glide_ratio > target_glide_ratio * 1.2:
                # We're gliding too flat - add a bit of downward force
                glide_correction = min(0.05, (current_glide_ratio - target_glide_ratio) * 0.01)
                # Apply a gentle downward force
                self.velocity[1] -= glide_correction * v * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Speed limits
        max_speed = 100.0
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed
        
        # Check collision with terrain if provided
        if terrain is not None:
            self.check_terrain_collisions(terrain, dt)

def draw_detailed_plane(plane, wire_mode=True):
    """Draw the plane model with enhanced detail, including landing gear.
    This function should be called from your rendering code after importing OpenGL.
    """
    try:
        from OpenGL.GL import glColor3f, glBegin, glEnd, glVertex3f, glLineWidth, glPushMatrix, glPopMatrix
        from OpenGL.GL import GL_LINES, GL_LINE_LOOP, GL_LINE_STRIP, GL_TRIANGLES, GL_QUADS, GL_POLYGON
        from OpenGL.GLUT import glutSolidSphere, glutWireSphere
        
        # Set color based on damage
        damage_factor = plane.damage
        undamaged_color = (1.0, 1.0, 0.0)  # Yellow
        damaged_color = (1.0, 0.0, 0.0)    # Red
        color = (
            undamaged_color[0] * (1-damage_factor) + damaged_color[0] * damage_factor,
            undamaged_color[1] * (1-damage_factor) + damaged_color[1] * damage_factor,
            undamaged_color[2] * (1-damage_factor) + damaged_color[2] * damage_factor
        )
        
        # Store original line width to restore later
        original_line_width = glGetFloat(GL_LINE_WIDTH)
        
        # Use thicker lines for better visibility
        glLineWidth(2.0)
        
        # ---------- Draw fuselage ----------
        glColor3f(*color)
        
        # Main fuselage (3D shape)
        fuselage_length = plane.fuselage_length
        fuselage_height = plane.fuselage_height
        fuselage_width = plane.fuselage_width
        
        # Top profile
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, fuselage_height/4, 0)  # Nose
        glVertex3f(fuselage_length/4, fuselage_height/2, 0)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/2, 0)  # Top middle
        glVertex3f(-fuselage_length/2, fuselage_height/4, 0)  # Top rear
        glEnd()
        
        # Bottom profile
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, -fuselage_height/4, 0)  # Nose bottom
        glVertex3f(fuselage_length/4, -fuselage_height/2, 0)  # Bottom front
        glVertex3f(-fuselage_length/3, -fuselage_height/2, 0)  # Bottom middle
        glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)  # Bottom rear
        glEnd()
        
        # Side struts to connect top and bottom
        glBegin(GL_LINES)
        # Front
        glVertex3f(fuselage_length/2, fuselage_height/4, 0)
        glVertex3f(fuselage_length/2, -fuselage_height/4, 0)
        
        # Mid front
        glVertex3f(fuselage_length/4, fuselage_height/2, 0)
        glVertex3f(fuselage_length/4, -fuselage_height/2, 0)
        
        # Mid rear
        glVertex3f(-fuselage_length/3, fuselage_height/2, 0)
        glVertex3f(-fuselage_length/3, -fuselage_height/2, 0)
        
        # Rear
        glVertex3f(-fuselage_length/2, fuselage_height/4, 0)
        glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)
        glEnd()
        
        # Side profiles (port side)
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, 0, -fuselage_width/2)  # Nose side
        glVertex3f(fuselage_length/4, fuselage_height/3, -fuselage_width/2)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/3, -fuselage_width/2)  # Top rear
        glVertex3f(-fuselage_length/2, 0, -fuselage_width/2)  # Tail
        glVertex3f(-fuselage_length/3, -fuselage_height/3, -fuselage_width/2)  # Bottom rear
        glVertex3f(fuselage_length/4, -fuselage_height/3, -fuselage_width/2)  # Bottom front
        glEnd()
        
        # Side profile (starboard side)
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, 0, fuselage_width/2)  # Nose side
        glVertex3f(fuselage_length/4, fuselage_height/3, fuselage_width/2)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/3, fuselage_width/2)  # Top rear
        glVertex3f(-fuselage_length/2, 0, fuselage_width/2)  # Tail
        glVertex3f(-fuselage_length/3, -fuselage_height/3, fuselage_width/2)  # Bottom rear
        glVertex3f(fuselage_length/4, -fuselage_height/3, fuselage_width/2)  # Bottom front
        glEnd()
        
        # Cockpit lines
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/5, fuselage_height/2.2, -fuselage_width/4)
        glVertex3f(fuselage_length/5, fuselage_height/2.2, fuselage_width/4)
        glVertex3f(-fuselage_length/8, fuselage_height/2.2, fuselage_width/4)
        glVertex3f(-fuselage_length/8, fuselage_height/2.2, -fuselage_width/4)
        glEnd()
        
        # ---------- Draw wings ----------
        wing_span = plane.wing_span
        wing_chord = plane.wing_chord
        
        # Wing color - blue with damage tint
        wing_color = (0.1, 0.3 * (1-damage_factor), 0.8 * (1-damage_factor))
        glColor3f(*wing_color)
        
        # Main wings - a quad with sweep
        glBegin(GL_LINE_LOOP)
        glVertex3f(wing_chord * 0.1, 0, -wing_span/2)  # Left wing tip leading edge
        glVertex3f(wing_chord * 0.1, 0, wing_span/2)   # Right wing tip leading edge
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2)  # Right wing tip trailing edge
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2)  # Left wing tip trailing edge
        glEnd()
        
        # Wing details (ribs and spars) for visual effect
        glBegin(GL_LINES)
        # Main spar
        glVertex3f(-wing_chord * 0.3, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.3, 0, wing_span/2)
        
        # Secondary spar
        glVertex3f(-wing_chord * 0.7, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.7, 0, wing_span/2)
        
        # Wing ribs
        sections = 10
        for i in range(sections + 1):
            z_pos = -wing_span/2 + i * wing_span/sections
            glVertex3f(wing_chord * 0.1, 0, z_pos)
            glVertex3f(-wing_chord * 0.9, 0, z_pos)
        glEnd()
        
        # Draw control surfaces (ailerons) with animation
        aileron_width = wing_span * 0.3
        aileron_depth = wing_chord * 0.2
        
        # Left aileron
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, -wing_span/2 + aileron_width)
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2 + aileron_width)
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2)
        glEnd()
        
        # Right aileron
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, wing_span/2 - aileron_width)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, wing_span/2)
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2)
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2 - aileron_width)
        glEnd()
        
        # Flaps (animated based on flap position)
        flap_extend = plane.flaps * wing_chord * 0.15  # How far flaps extend down
        flap_width = wing_span * 0.4
        flap_start = wing_span * 0.05  # Distance from fuselage
        
        # Flap connection lines
        glBegin(GL_LINES)
        # Left flap
        glVertex3f(-wing_chord * 0.9, 0, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start - flap_width)
        
        glVertex3f(-wing_chord * 0.9, 0, -flap_start)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start)
        
        # Right flap
        glVertex3f(-wing_chord * 0.9, 0, flap_start)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start)
        
        glVertex3f(-wing_chord * 0.9, 0, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start + flap_width)
        glEnd()
        
        # Left flap surface
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9, 0, -flap_start)
        glVertex3f(-wing_chord * 0.9, 0, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start)
        glEnd()
        
        # Right flap surface
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9, 0, flap_start)
        glVertex3f(-wing_chord * 0.9, 0, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start)
        glEnd()
        
        # ---------- Draw tail surfaces ----------
        tail_color = (0.8, 0.4, 0.9)  # Purple
        glColor3f(*tail_color)
        
        # Horizontal stabilizer
        h_stab_span = wing_span * 0.4
        h_stab_chord = wing_chord * 0.5
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.8, 0, -h_stab_span/2)  # Left tip
        glVertex3f(-fuselage_length * 0.8, 0, h_stab_span/2)   # Right tip
        glVertex3f(-fuselage_length * 0.95, 0, h_stab_span/2)  # Right rear
        glVertex3f(-fuselage_length * 0.95, 0, -h_stab_span/2)  # Left rear
        glEnd()
        
        # Vertical stabilizer
        v_stab_height = fuselage_height * 1.5
        v_stab_chord = wing_chord * 0.6
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.7, 0, 0)  # Bottom front
        glVertex3f(-fuselage_length * 0.7, v_stab_height, 0)  # Top front
        glVertex3f(-fuselage_length * 0.95, v_stab_height * 0.8, 0)  # Top rear
        glVertex3f(-fuselage_length * 0.95, 0, 0)  # Bottom rear
        glEnd()
        
        # Elevator (animated)
        elevator_deflection = 0  # Add control for this if needed
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.9, elevator_deflection, -h_stab_span/2)
        glVertex3f(-fuselage_length * 0.9, elevator_deflection, h_stab_span/2)
        glVertex3f(-fuselage_length * 0.95, elevator_deflection, h_stab_span/2)
        glVertex3f(-fuselage_length * 0.95, elevator_deflection, -h_stab_span/2)
        glEnd()
        
        # Rudder (animated)
        rudder_deflection = 0  # Add control for this if needed
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.85, v_stab_height * 0.2, rudder_deflection)
        glVertex3f(-fuselage_length * 0.85, v_stab_height * 0.9, rudder_deflection)
        glVertex3f(-fuselage_length * 0.95, v_stab_height * 0.7, rudder_deflection)
        glVertex3f(-fuselage_length * 0.95, 0, rudder_deflection)
        glEnd()
        
        # ---------- Draw landing gear ----------
        # Only draw if partially or fully extended
        if plane.gear_state > 0.01:
            gear_color = (0.5, 0.5, 0.5)  # Gray for landing gear
            glColor3f(*gear_color)
            
            wheel_radius = plane.wheel_radius
            gear_height = plane.gear_height
            
            # Left main gear
            left_wheel_pos = plane.wheel_points['left'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                retract_angle = (1.0 - plane.gear_state) * 90  # Degrees
                retract_angle_rad = retract_angle * math.pi / 180.0
                
                # Calculate retracted position
                retracted_y = -plane.fuselage_height * 0.2
                retracted_z = left_wheel_pos[2] * 0.5
                
                # Interpolate
                left_wheel_pos[1] = left_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                left_wheel_pos[2] = left_wheel_pos[2] * plane.gear_state + retracted_z * (1.0 - plane.gear_state)
            
            # Adjust for compression
            left_wheel_pos[1] += plane.wheel_compression['left'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(0, 0, left_wheel_pos[2])  # Gear attachment point
            glVertex3f(left_wheel_pos[0], left_wheel_pos[1], left_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(left_wheel_pos[0], left_wheel_pos[1], left_wheel_pos[2])
            draw_wheel(wheel_radius)
            glPopMatrix()
            
            # Right main gear
            right_wheel_pos = plane.wheel_points['right'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                retract_angle = (1.0 - plane.gear_state) * 90  # Degrees
                retract_angle_rad = retract_angle * math.pi / 180.0
                
                # Calculate retracted position
                retracted_y = -plane.fuselage_height * 0.2
                retracted_z = right_wheel_pos[2] * 0.5
                
                # Interpolate
                right_wheel_pos[1] = right_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                right_wheel_pos[2] = right_wheel_pos[2] * plane.gear_state + retracted_z * (1.0 - plane.gear_state)
            
            # Adjust for compression
            right_wheel_pos[1] += plane.wheel_compression['right'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(0, 0, right_wheel_pos[2])  # Gear attachment point
            glVertex3f(right_wheel_pos[0], right_wheel_pos[1], right_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(right_wheel_pos[0], right_wheel_pos[1], right_wheel_pos[2])
            draw_wheel(wheel_radius)
            glPopMatrix()
            
            # Nose gear
            nose_wheel_pos = plane.wheel_points['nose'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                # Calculate retracted position (different for nose gear - retracts forward)
                retracted_y = -plane.fuselage_height * 0.2
                retracted_x = nose_wheel_pos[0] * 0.5
                
                # Interpolate
                nose_wheel_pos[1] = nose_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                nose_wheel_pos[0] = nose_wheel_pos[0] * plane.gear_state + retracted_x * (1.0 - plane.gear_state)
            
            # Adjust for compression
            nose_wheel_pos[1] += plane.wheel_compression['nose'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(fuselage_length/4, -fuselage_height/3, 0)  # Gear attachment point
            glVertex3f(nose_wheel_pos[0], nose_wheel_pos[1], nose_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(nose_wheel_pos[0], nose_wheel_pos[1], nose_wheel_pos[2])
            draw_wheel(wheel_radius * 0.8)  # Smaller nose wheel
            glPopMatrix()
            
            # Tail wheel (if plane has one)
            if 'tail' in plane.wheel_points:
                tail_wheel_pos = plane.wheel_points['tail'].copy()
                # Adjust for gear state (minimal retraction for tail wheel)
                if plane.gear_state < 1.0:
                    # Calculate retracted position (tail wheel retracts less)
                    retracted_y = -plane.fuselage_height * 0.1
                    
                    # Interpolate
                    tail_wheel_pos[1] = tail_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                
                # Adjust for compression
                tail_wheel_pos[1] += plane.wheel_compression['tail'] * wheel_radius
                
                # Draw strut
                glBegin(GL_LINES)
                glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)  # Gear attachment point
                glVertex3f(tail_wheel_pos[0], tail_wheel_pos[1], tail_wheel_pos[2])  # Wheel center
                glEnd()
                
                # Draw wheel using a circle of lines
                glPushMatrix()
                glTranslatef(tail_wheel_pos[0], tail_wheel_pos[1], tail_wheel_pos[2])
                draw_wheel(wheel_radius * 0.5)  # Smaller tail wheel
                glPopMatrix()
        
        # Draw propeller
        prop_color = (0.8, 0.8, 0.8)  # Light gray
        glColor3f(*prop_color)
        
        # Propeller spinner
        glPushMatrix()
        glTranslatef(fuselage_length/2 + 0.2, 0, 0)
        if wire_mode:
            glutWireSphere(plane.fuselage_width * 0.15, 8, 8)
        else:
            glutSolidSphere(plane.fuselage_width * 0.15, 8, 8)
        glPopMatrix()
        
        # Propeller blades (rotating)
        # Could animate rotation based on throttle
        prop_radius = wing_span * 0.15
        
        glBegin(GL_LINES)
        # Vertical blade
        glVertex3f(fuselage_length/2 + 0.2, prop_radius, 0)
        glVertex3f(fuselage_length/2 + 0.2, -prop_radius, 0)
        
        # Horizontal blade
        glVertex3f(fuselage_length/2 + 0.2, 0, prop_radius)
        glVertex3f(fuselage_length/2 + 0.2, 0, -prop_radius)
        glEnd()
        
        # Restore original line width
        glLineWidth(original_line_width)
        
    except Exception as e:
        # Fall back to basic drawing if there's an error
        print(f"Error rendering plane: {e}")
        basic_draw_plane(plane)

def draw_wheel(radius):
    """Draw a wheel as a circle of lines."""
    try:
        from OpenGL.GL import glBegin, glEnd, glVertex3f
        from OpenGL.GL import GL_LINE_LOOP
        
        segments = 12
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(0, math.sin(angle) * radius, math.cos(angle) * radius)
        glEnd()
        
        # Add wheel hub details
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(0, math.sin(angle) * radius * 0.3, math.cos(angle) * radius * 0.3)
        glEnd()
        
        # Add spokes
        glBegin(GL_LINES)
        for i in range(6):  # 6 spokes
            angle = 2 * math.pi * i / 6
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            
            # Inner point
            glVertex3f(0, sin_a * radius * 0.3, cos_a * radius * 0.3)
            # Outer point
            glVertex3f(0, sin_a * radius, cos_a * radius)
        glEnd()
    except Exception as e:
        print(f"Error drawing wheel: {e}")

def basic_draw_plane(plane):
    """Simple fallback drawing function if the detailed one fails."""
    try:
        from OpenGL.GL import glColor3f, glBegin, glEnd, glVertex3f
        from OpenGL.GL import GL_LINE_LOOP, GL_LINES
        
        # Set color based on damage
        damage_factor = plane.damage
        color = (1.0, 1.0 * (1-damage_factor), 0.0)
        glColor3f(*color)
        
        # Simple fuselage
        glBegin(GL_LINE_LOOP)
        glVertex3f(1, 0, 0)  # Nose
        glVertex3f(-1, 0.5, 0)  # Top back
        glVertex3f(-1, -0.5, 0)  # Bottom back
        glEnd()
        
        # Wings
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, -plane.wing_span/2)  # Left wing tip
        glVertex3f(0, 0, plane.wing_span/2)   # Right wing tip
        glVertex3f(-0.5, 0, plane.wing_span/2)  # Right wing back
        glVertex3f(-0.5, 0, -plane.wing_span/2)  # Left wing back
        glEnd()
        
        # Tail
        glBegin(GL_LINE_LOOP)
        glVertex3f(-0.8, 0, 0)    # Bottom
        glVertex3f(-0.8, 0.5, 0)  # Top
        glVertex3f(-1, 0, 0)      # Back
        glEnd()
        
    except Exception as e:
        print(f"Error in basic plane rendering: {e}")

def rotate_vector(v, axis, angle):
    """Rotate vector v around a given axis by a given angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))