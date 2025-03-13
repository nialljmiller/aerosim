import numpy as np

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
        
        # Collision state tracking
        self.is_grounded = False
        self.collision_points = []
        self.ground_normal = np.array([0, 1, 0])
        
        # Damage tracking
        self.damage = 0.0  # From 0 (no damage) to 1.0 (destroyed)
        self.last_collision_time = 0.0
        
        # Wing endpoints in local coordinates (for collision detection)
        self.wing_points = [
            np.array([0, 0, -3]),  # Left wing tip
            np.array([0, 0, 3]),   # Right wing tip
            np.array([-1, 0, -1]), # Left horizontal stabilizer
            np.array([-1, 0, 1]),  # Right horizontal stabilizer
            np.array([1, 0, 0]),   # Nose
            np.array([-1, 0, 0])   # Tail
        ]

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
            yaw *= 0.3  # Reduce yaw effectiveness on ground
        
        # Apply pitch: rotate around right vector
        self.forward = rotate_vector(self.forward, self.right, pitch * dt)
        self.up = rotate_vector(self.up, self.right, pitch * dt)
        
        # Apply roll: rotate around forward vector
        # Note: The roll direction correction is now handled at the input level
        self.up = rotate_vector(self.up, self.forward, roll * dt)
        self.right = rotate_vector(self.right, self.forward, roll * dt)
        
        # Apply yaw: rotate around up vector
        self.forward = rotate_vector(self.forward, self.up, yaw * dt)
        self.right = rotate_vector(self.right, self.up, yaw * dt)
        
        # Add coordinated turn effect: banking (roll) naturally causes turning (yaw)
        if not self.is_grounded:  # Only apply in air
            # Calculate bank angle by measuring angle between world up and plane up
            world_up = np.array([0, 1, 0])
            bank_angle = np.arccos(np.clip(np.dot(self.up, world_up), -1.0, 1.0))
            # Determine bank direction (left/right) using cross product
            bank_direction = np.sign(np.dot(np.cross(world_up, self.up), self.forward))
            # Apply automatic yaw based on bank angle (coordinated turn)
            turn_rate = 0.5 * bank_angle * bank_direction
            self.forward = rotate_vector(self.forward, self.up, turn_rate * dt)
            self.right = rotate_vector(self.right, self.up, turn_rate * dt)
        
        # Re-orthonormalize basis vectors
        self.forward /= np.linalg.norm(self.forward)
        self.up /= np.linalg.norm(self.up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)

    def get_world_point(self, local_point):
        """Convert a point from local plane coordinates to world coordinates."""
        # Create rotation matrix from the plane's basis vectors
        rotation_matrix = np.array([
            self.forward,
            self.up,
            self.right
        ]).T  # Transpose for correct multiplication
        
        # Rotate the local point and add the plane's position
        return self.position + rotation_matrix @ local_point

    def check_terrain_collisions(self, terrain, dt):
        """Check if any part of the plane is colliding with the terrain."""
        self.collision_points = []
        self.is_grounded = False
        
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
                
                # If nose or central body collides, mark as grounded
                if point_idx in [4, 5]:  # Nose or tail
                    self.is_grounded = True
                    self.ground_normal = normal
        
        # Handle collisions if any
        if self.collision_points:
            self.resolve_collisions(dt)
            return True
        
        return False

    def resolve_collisions(self, dt):
        """Resolve terrain collisions with appropriate physics response."""
        if not self.collision_points:
            return
        
        # Get current time for collision damage calculation
        current_time = pygame.time.get_ticks() / 1000.0
        collision_response = np.zeros(3)
        collision_count = len(self.collision_points)
        
        for collision in self.collision_points:
            penetration = collision['penetration']
            normal = collision['normal']
            point_idx = collision['point_idx']
            
            # Add collision response force
            response_strength = penetration * 50.0  # Proportional to penetration depth
            collision_response += normal * response_strength / collision_count
            
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
            # Move the plane out of the ground
            self.position += collision_response * dt * 10
            
            # Reflect velocity for a bounce effect, with damping
            if self.is_grounded:
                # If on wheels (grounded), apply rolling friction
                ground_speed = np.linalg.norm(self.velocity - np.array([0, self.velocity[1], 0]))
                friction = min(1.0, 0.01 * ground_speed)  # Friction increases with speed, up to a limit
                
                # Apply friction to horizontal velocity
                self.velocity[0] *= (1.0 - friction)
                self.velocity[2] *= (1.0 - friction)
                
                # Zero out vertical velocity if pointing down
                if self.velocity[1] < 0:
                    self.velocity[1] = 0
            else:
                # For wing/tail collisions, apply a bounce with energy loss
                normal_vel = np.dot(self.velocity, collision_response)
                if normal_vel < 0:  # Only bounce if moving toward the surface
                    # Reflect velocity with damping
                    reflection = self.velocity - 1.8 * normal_vel * collision_response / np.linalg.norm(collision_response)
                    energy_loss = 0.3  # 30% energy lost in collision
                    self.velocity = reflection * (1.0 - energy_loss)

    def update(self, dt, terrain):
        """Physics update with improved flight dynamics and terrain collision."""
        # Calculate thrust along the forward direction
        actual_thrust = self.max_thrust * self.throttle
        thrust_force = actual_thrust * self.forward
        
        # Calculate current speed
        v = np.linalg.norm(self.velocity)
        
        # Improved aerodynamics for better gliding
        # Adjust the lift coefficient based on angle of attack
        # Get direction of airflow relative to plane orientation
        if v > 0.1:  # Only calculate if we have meaningful velocity
            airflow_direction = -self.velocity / v  # Normalized and inverted (airflow comes from opposite direction of velocity)
            
            # Calculate angle of attack (angle between airflow and forward direction)
            # dot product of two unit vectors gives cosine of angle between them
            angle_of_attack_cos = np.dot(airflow_direction, self.forward)
            angle_of_attack = np.arccos(np.clip(angle_of_attack_cos, -1.0, 1.0))
            
            # Adjust lift coefficient based on angle of attack
            # Lift peaks at around 15 degrees (0.26 radians) and then decreases
            angle_factor = 1.0 - abs(angle_of_attack - 0.26) * 2.0
            effective_lift_coefficient = self.params['lift_coefficient'] * max(0.1, min(1.2, angle_factor))
        else:
            effective_lift_coefficient = self.params['lift_coefficient']
        
        # Calculate drag with improved model for gliding
        if v > 0.01:
            # Calculate induced drag (due to lift) and parasitic drag (due to shape)
            # Higher lift = higher induced drag
            induced_drag_coefficient = (effective_lift_coefficient ** 2) / (3.14159 * 6.0)  # Based on wing aspect ratio
            total_drag_coefficient = self.params['drag_coefficient'] + induced_drag_coefficient
            
            drag_mag = 0.5 * self.params['air_density'] * v**2 * total_drag_coefficient * self.params['wing_area']
            drag_force = -drag_mag * (self.velocity / v)
        else:
            drag_force = np.zeros(3)
        
        # Calculate lift perpendicular to forward direction
        # Lift is perpendicular to airflow, not just 'up'
        if v > 1.0:  # Only calculate meaningful lift when we have speed
            # Calculate the plane's air-relative velocity and right vector in air space
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
                
            # Automatic aircraft orientation alignment with velocity
            if v > 5.0 and not self.is_grounded:  # Only align once we have significant speed and are in the air
                # Calculate how misaligned the plane is with its velocity
                alignment = np.dot(self.forward, self.velocity) / (np.linalg.norm(self.forward) * v)
                
                # Factor controlling how quickly the plane aligns with velocity
                alignment_factor = 0.8 * (1.0 - self.throttle * 0.5)  # More effect at low throttle
                
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
                        
            # Additional pitch-down effect during low throttle
            if self.throttle < 0.2 and self.velocity[1] > -5.0 and not self.is_grounded:
                pitch_down_factor = (0.2 - self.throttle) * 0.5  # Stronger at lower throttle
                pitch_down_axis = self.right  # Pitch around right vector (wings)
                pitch_down_angle = pitch_down_factor * dt
                
                # Apply subtle pitch down rotation
                self.forward = rotate_vector(self.forward, pitch_down_axis, -pitch_down_angle)
                self.up = rotate_vector(self.up, pitch_down_axis, -pitch_down_angle)
        
        # Enhanced gliding physics
        if self.throttle < 0.1 and v > 5.0 and self.velocity[1] < 0 and not self.is_grounded:
            # Calculate current glide ratio (horizontal distance / vertical drop)
            horizontal_velocity = np.array([self.velocity[0], 0, self.velocity[2]])
            horizontal_speed = np.linalg.norm(horizontal_velocity)
            vertical_speed = abs(self.velocity[1])  # Should be negative when descending
            
            # Calculate current and target glide ratio
            current_glide_ratio = horizontal_speed / max(0.1, vertical_speed)
            target_glide_ratio = self.glide_efficiency
            
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
        
        # Check collision with terrain
        self.check_terrain_collisions(terrain, dt)




def rotate_vector(v, axis, angle):
    """Rotate vector v around a given axis by a given angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))
