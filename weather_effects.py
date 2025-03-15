import numpy as np
import random
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

class CloudParticle:
    """Individual cloud particle that makes up larger cloud formations."""
    def __init__(self, position, size, opacity=1.0, drift_speed=None, color=(1.0, 1.0, 1.0)):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.base_size = size  # Store original size for animations
        self.opacity = opacity
        self.base_opacity = opacity  # Store original opacity for animations
        self.color = color  # Allow colored clouds for special effects
        
        # Random rotation for variety
        self.rotation = random.uniform(0, 360)
        
        # Movement properties
        if drift_speed is None:
            # Default subtle drift
            self.drift_speed = np.array([
                random.uniform(-0.5, 0.5),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.5, 0.5)
            ])
        else:
            self.drift_speed = np.array(drift_speed)
        
        # Animation properties
        self.oscillation_phase = random.uniform(0, 2 * math.pi)
        self.oscillation_speed = random.uniform(0.1, 0.3)
        self.oscillation_magnitude = random.uniform(0.02, 0.05)
        
        # Lightning properties (for storm clouds)
        self.lightning_active = False
        self.lightning_duration = 0
        self.lightning_max_duration = random.uniform(0.1, 0.3)  # In seconds

    def update(self, dt, wind_direction, wind_strength, time):
        """Update particle position and appearance."""
        # Apply wind-based drift
        wind_effect = wind_direction * wind_strength * dt
        self.position += self.drift_speed * dt + wind_effect
        
        # Apply subtle size oscillation
        oscillation = math.sin(time * self.oscillation_speed + self.oscillation_phase)
        size_factor = 1.0 + oscillation * self.oscillation_magnitude
        self.size = self.base_size * size_factor
        
        # Update any active lightning
        if self.lightning_active:
            self.lightning_duration -= dt
            if self.lightning_duration <= 0:
                self.lightning_active = False

    def trigger_lightning(self):
        """Trigger a lightning flash in this cloud particle."""
        if random.random() < 0.3:  # Only some particles show lightning
            self.lightning_active = True
            self.lightning_duration = self.lightning_max_duration


class CloudFormation:
    """Base class for cloud formations composed of multiple particles."""
    def __init__(self, center_position, size_factor=1.0, particle_count=20, opacity=0.9):
        self.center_position = np.array(center_position, dtype=float)
        self.size_factor = size_factor
        self.particles = []
        self.display_list = None
        self.compiled = False
        
        # Default cloud color (white)
        self.color = (1.0, 1.0, 1.0)
        
        # Time tracking for animation
        self.formation_time = random.uniform(0, 100)  # Random start time for variation
        
        # Flag for whether this cloud can produce precipitation
        self.can_precipitate = False
        self.precipitation_intensity = 0.0
        self.precipitation_type = None  # "rain", "snow", or None
        
        # Lightning properties
        self.can_lightning = False
        self.lightning_probability = 0.0
        self.lightning_cooldown = 0.0
        
        # Create the cloud particles
        self.generate_particles(particle_count, opacity)
    
    def generate_particles(self, count, opacity):
        """Generate cloud particles arranged in a formation pattern."""
        # Base implementation creates a simple spherical arrangement
        radius = 20.0 * self.size_factor
        for _ in range(count):
            # Random position within a sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            r = radius * random.uniform(0.4, 1.0)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta) * 0.4  # Flatten vertically
            z = r * math.cos(phi)
            
            position = self.center_position + np.array([x, y, z])
            
            # Random size
            particle_size = random.uniform(5, 15) * self.size_factor
            
            # Random opacity
            particle_opacity = opacity * random.uniform(0.6, 1.0)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity))
    
    def update(self, dt, wind_direction, wind_strength):
        """Update cloud formation position and properties."""
        # Update internal time counter
        self.formation_time += dt
        
        # Update center position based on wind
        self.center_position += wind_direction * wind_strength * dt
        
        # Update all particles
        for particle in self.particles:
            particle.update(dt, wind_direction, wind_strength, self.formation_time)
        
        # Handle lightning generation for storm clouds
        if self.can_lightning and self.lightning_cooldown <= 0:
            if random.random() < self.lightning_probability * dt:
                self.generate_lightning()
                # Set cooldown between lightning flashes
                self.lightning_cooldown = random.uniform(2.0, 10.0)
        else:
            self.lightning_cooldown -= dt
    
    def generate_lightning(self):
        """Generate lightning effect in the cloud."""
        # Trigger lightning in random particles
        lightning_particles = min(len(self.particles) // 3, 5)  # Limit number of particles with lightning
        for _ in range(lightning_particles):
            particle_idx = random.randint(0, len(self.particles) - 1)
            self.particles[particle_idx].trigger_lightning()
    
    def compile_display_list(self):
        """Compile cloud billboards into a display list for efficient rendering."""
        if self.compiled:
            return
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Draw a quad aligned to camera
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.0)
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 0.0)
        
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, 0.0)
        glEnd()
        
        glEndList()
        self.compiled = True
    
    def draw(self, camera_position, cloud_texture):
        """Draw cloud formation with billboarded particles."""
        # Compile display list if needed
        if not self.compiled:
            self.compile_display_list()
        
        for particle in self.particles:
            # Skip particles that are too far away
            if np.linalg.norm(particle.position - camera_position) > 2000:
                continue
            
            glPushMatrix()
            
            # Position the particle in 3D space
            glTranslatef(particle.position[0], particle.position[1], particle.position[2])
            
            # Calculate billboarding matrix
            # This makes the cloud particle face the camera
            dx = camera_position[0] - particle.position[0]
            dy = camera_position[1] - particle.position[1]
            dz = camera_position[2] - particle.position[2]
            
            # Distance to camera
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Normalize view vector
            if dist > 0:
                dx /= dist
                dy /= dist
                dz /= dist
            
            # Up vector
            ux, uy, uz = 0, 1, 0  # World up
            
            # Side vector (right) - cross product of up and view direction
            sx = uy * dz - uz * dy
            sy = uz * dx - ux * dz
            sz = ux * dy - uy * dx
            
            # Normalize side vector
            side_len = math.sqrt(sx*sx + sy*sy + sz*sz)
            if side_len > 0:
                sx /= side_len
                sy /= side_len
                sz /= side_len
            
            # New up vector (cross product of view direction and side vector)
            ux = dy * sz - dz * sy
            uy = dz * sx - dx * sz
            uz = dx * sy - dy * sx
            
            # Set billboarding matrix
            billboard_matrix = [
                sx, ux, -dx, 0,
                sy, uy, -dy, 0,
                sz, uz, -dz, 0,
                0, 0, 0, 1
            ]
            
            glMultMatrixf(billboard_matrix)
            
            # Apply rotation around view axis for variety
            glRotatef(particle.rotation, 0, 0, 1)
            
            # Scale particle
            scale = particle.size
            glScalef(scale, scale, 1.0)
            
            # Set color and opacity
            if particle.lightning_active:
                # Flash bright white/blue during lightning
                flash_intensity = min(1.0, particle.lightning_duration / particle.lightning_max_duration * 2.0)
                r = 1.0
                g = 1.0
                b = 1.0 + flash_intensity * 0.5  # Blueish tint
                a = particle.opacity * (1.0 + flash_intensity * 0.5)
                glColor4f(r, g, b, min(1.0, a))
            else:
                # Normal cloud color with particle's opacity
                base_color = particle.color
                glColor4f(base_color[0], base_color[1], base_color[2], particle.opacity)
            
            # Draw the billboard
            glCallList(self.display_list)
            
            glPopMatrix()
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.compiled and self.display_list:
            glDeleteLists(self.display_list, 1)
            self.compiled = False


class CumulusCloud(CloudFormation):
    """Puffy, cotton-like cumulus clouds."""
    def __init__(self, center_position, size_factor=1.0, particle_count=30):
        super().__init__(center_position, size_factor, particle_count, opacity=0.8)
        # Bright white with a slight bluish tint for better visibility
        self.color = (0.95, 0.95, 1.0)  
    
    def generate_particles(self, count, opacity):
        """Generate particles in a puffy cumulus pattern."""
        # Cumulus clouds are wider than they are tall
        width = 50.0 * self.size_factor
        height = 30.0 * self.size_factor
        
        for _ in range(count):
            # Generate positions with denser core and fluffy edges
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            
            # Use different distributions for more dense bottom and fluffy top
            if random.random() < 0.7:  # 70% in main body
                r = random.uniform(0.3, 0.8) * width
                y_offset = random.uniform(-0.2, 0.5) * height
            else:  # 30% in top fluffy part
                r = random.uniform(0.1, 0.6) * width
                y_offset = random.uniform(0.3, 1.0) * height
            
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            y = y_offset
            
            position = self.center_position + np.array([x, y, z])
            
            # Size varies - larger in middle, smaller at edges
            distance_from_center = math.sqrt(x*x + z*z) / width
            particle_size = random.uniform(10, 25) * self.size_factor * (1.0 - 0.5 * distance_from_center)
            
            # Opacity - more opaque in center
            particle_opacity = opacity * (1.0 - 0.3 * distance_from_center)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, color=self.color))


class CirrusCloud(CloudFormation):
    """High, thin, wispy cirrus clouds."""
    def __init__(self, center_position, size_factor=1.0, particle_count=15):
        super().__init__(center_position, size_factor, particle_count, opacity=0.4)
        # Slightly blue-white with enhanced visibility
        self.color = (0.9, 0.9, 1.0)  
        
        # Cirrus clouds move faster with the wind
        self.wind_multiplier = 2.0
    
    def generate_particles(self, count, opacity):
        """Generate particles in a wispy cirrus pattern."""
        # Cirrus clouds are stretched horizontally
        length = 100.0 * self.size_factor
        width = 20.0 * self.size_factor
        height = 10.0 * self.size_factor
        
        # Primary direction of the streaks
        streak_angle = random.uniform(0, 2 * math.pi)
        dx = math.cos(streak_angle)
        dz = math.sin(streak_angle)
        
        for _ in range(count):
            # Position along the streak line
            t = random.uniform(-1.0, 1.0)
            x_offset = t * length * dx
            z_offset = t * length * dz
            
            # Random deviation perpendicular to streak
            perp_x = -dz * random.uniform(-width, width)
            perp_z = dx * random.uniform(-width, width)
            
            # Random height
            y_offset = random.uniform(-height/2, height/2)
            
            position = self.center_position + np.array([
                x_offset + perp_x,
                y_offset,
                z_offset + perp_z
            ])
            
            # Thin, elongated particles
            particle_size = random.uniform(10, 30) * self.size_factor
            
            # Cirrus clouds are semi-transparent but more visible than default
            particle_opacity = opacity * random.uniform(0.5, 0.8)
            
            # Add drift in the streak direction
            drift_speed = np.array([dx, 0, dz]) * random.uniform(0.5, 1.5)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, drift_speed, color=self.color))
    
    def update(self, dt, wind_direction, wind_strength):
        """Update cirrus clouds with enhanced wind effect."""
        # Cirrus clouds move faster with the wind
        effective_wind_strength = wind_strength * self.wind_multiplier
        super().update(dt, wind_direction, effective_wind_strength)


class StratusCloud(CloudFormation):
    """Flat, layered stratus clouds covering large areas."""
    def __init__(self, center_position, size_factor=1.0, particle_count=40):
        super().__init__(center_position, size_factor, particle_count, opacity=0.6)
        # Light gray with better visibility
        self.color = (0.8, 0.8, 0.85)  
        self.can_precipitate = True
        self.precipitation_intensity = 0.2  # Light precipitation
        
        # Determine precipitation type based on altitude (snow at high altitude)
        if center_position[1] > 40:
            self.precipitation_type = "snow"
        else:
            self.precipitation_type = "rain"
    
    def generate_particles(self, count, opacity):
        """Generate particles in a flat stratus pattern."""
        # Stratus clouds spread horizontally in a layer
        spread_x = 120.0 * self.size_factor
        spread_z = 120.0 * self.size_factor
        height_range = 15.0 * self.size_factor
        
        for _ in range(count):
            # Flat distribution with slight undulation
            x = random.uniform(-spread_x, spread_x)
            z = random.uniform(-spread_z, spread_z)
            
            # Very little vertical variation
            distance_factor = math.sqrt(x*x + z*z) / max(spread_x, spread_z)
            y = random.uniform(-height_range/2, height_range/2) * (1.0 + distance_factor)
            
            position = self.center_position + np.array([x, y, z])
            
            # Slightly larger particles for better visibility
            particle_size = random.uniform(20, 40) * self.size_factor
            
            # More uniform opacity
            particle_opacity = opacity * random.uniform(0.8, 1.0)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, color=self.color))


class CumulonimbusCloud(CloudFormation):
    """Towering storm clouds with lightning and heavy precipitation."""
    def __init__(self, center_position, size_factor=1.0, particle_count=50):
        super().__init__(center_position, size_factor, particle_count, opacity=0.8)
        # Darker gray but still visible
        self.color = (0.7, 0.7, 0.8)  
        self.can_precipitate = True
        self.precipitation_intensity = 0.8  # Heavy precipitation
        
        # Determine precipitation type based on altitude
        if center_position[1] > 50:
            self.precipitation_type = "snow"
        else:
            self.precipitation_type = "rain"
        
        # Lightning properties
        self.can_lightning = True
        self.lightning_probability = 0.1  # Probability of lightning per second
        self.lightning_cooldown = random.uniform(1.0, 5.0)  # Initial cooldown
    
    def generate_particles(self, count, opacity):
        """Generate particles in a towering cumulonimbus pattern."""
        # Cumulonimbus clouds are very tall
        width = 80.0 * self.size_factor
        height = 120.0 * self.size_factor
        
        for _ in range(count):
            # Different distributions for different parts of the cloud
            r = random.random()
            
            if r < 0.6:  # Main body (60%)
                x = random.uniform(-0.7, 0.7) * width
                z = random.uniform(-0.7, 0.7) * width
                y = random.uniform(0.0, 0.6) * height
            elif r < 0.9:  # Anvil top (30%)
                anvil_spread = 1.5  # Anvil spreads wider than the base
                x = random.uniform(-0.9, 0.9) * width * anvil_spread
                z = random.uniform(-0.9, 0.9) * width * anvil_spread
                y = random.uniform(0.6, 1.0) * height
            else:  # Base (10%)
                x = random.uniform(-0.6, 0.6) * width
                z = random.uniform(-0.6, 0.6) * width
                y = random.uniform(-0.2, 0.0) * height
            
            position = self.center_position + np.array([x, y, z])
            
            # Size varies by position - larger particles for better visibility
            if y > 0.6 * height:  # Anvil top
                particle_size = random.uniform(20, 35) * self.size_factor
            else:  # Main body and base
                particle_size = random.uniform(25, 45) * self.size_factor
            
            # Opacity - more opaque in center
            distance_from_center = math.sqrt(x*x + z*z) / width
            particle_opacity = opacity * (1.0 - 0.3 * distance_from_center)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, color=self.color))


class TornadoCloud(CloudFormation):
    """Dramatic tornado or severe storm mesocyclone cloud formation."""
    def __init__(self, center_position, size_factor=1.0, particle_count=60):
        super().__init__(center_position, size_factor, particle_count, opacity=0.9)
        # Dark, ominous coloration
        self.color = (0.5, 0.5, 0.6)  
        self.can_precipitate = True
        self.precipitation_intensity = 1.0  # Maximum precipitation
        self.precipitation_type = "rain"
        
        # Enhanced lightning properties
        self.can_lightning = True
        self.lightning_probability = 0.3  # High probability of lightning
        self.lightning_cooldown = random.uniform(0.5, 2.0)  # Short cooldown for frequent lightning
        
        # Rotation properties
        self.rotation_angle = 0  # Current rotation angle
        self.rotation_speed = random.uniform(0.2, 0.5)  # Radians per second
        
        # Create funnel cloud
        self.has_funnel = random.random() < 0.3  # 30% chance of having a visible funnel
        if self.has_funnel:
            self.funnel_particles = []
            self.generate_funnel()
    
    def generate_particles(self, count, opacity):
        """Generate particles in a severe storm cloud pattern."""
        # Base is wide, with a rotating mesocyclone structure
        width = 100.0 * self.size_factor
        height = 80.0 * self.size_factor
        
        for _ in range(count):
            r = random.random()
            
            if r < 0.5:  # Main cloud mass (50%)
                theta = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.3, 1.0) * width
                x = radius * math.cos(theta)
                z = radius * math.sin(theta)
                y = random.uniform(0.2, 0.8) * height
            elif r < 0.8:  # Lower shelf/wall cloud (30%)
                theta = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.6, 1.2) * width
                x = radius * math.cos(theta)
                z = radius * math.sin(theta)
                y = random.uniform(-0.1, 0.3) * height
            else:  # Anvil/top (20%)
                theta = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.5, 1.4) * width
                x = radius * math.cos(theta)
                z = radius * math.sin(theta)
                y = random.uniform(0.7, 1.0) * height
            
            position = self.center_position + np.array([x, y, z])
            
            # Larger, more dramatic cloud particles
            particle_size = random.uniform(25, 50) * self.size_factor
            
            # Dark, more opaque clouds
            particle_opacity = opacity * random.uniform(0.8, 1.0)
            
            # Add swirling drift based on position
            drift_speed = np.array([
                -z * 0.02,  # Rotational component
                random.uniform(-0.1, 0.1),
                x * 0.02
            ])
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, drift_speed, color=self.color))
    
    def generate_funnel(self):
        """Generate a tornado funnel cloud extending from the base."""
        funnel_height = 120.0 * self.size_factor
        max_radius = 15.0 * self.size_factor
        
        # Funnel extends downward from cloud base
        cloud_base_y = self.center_position[1] - 30.0 * self.size_factor
        
        # Generate funnel particles
        count = int(40 * self.size_factor)
        for i in range(count):
            # Position along funnel (t=0 at cloud base, t=1 at ground)
            t = i / (count - 1)
            
            # Narrow at the bottom, wider at top
            radius = max_radius * (1.0 - 0.8 * t)
            
            # Apply spiral pattern
            theta = t * 4 * math.pi  # More twists for longer funnels
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            
            # Height decreases from cloud base
            y = cloud_base_y - t * funnel_height
            
            position = np.array([
                self.center_position[0] + x,
                y,
                self.center_position[2] + z
            ])
            
            # Funnel particles are smaller and more transparent
            particle_size = random.uniform(5, 15) * self.size_factor * (1.0 - 0.5 * t)
            
            # Funnel gets more transparent toward the ground
            particle_opacity = 0.8 * (1.0 - 0.7 * t)
            
            # Special dark gray color for funnel
            funnel_color = (0.4, 0.4, 0.45)
            
            # Strong rotational drift
            drift_speed = np.array([
                -z * 0.05,  # Stronger rotation
                random.uniform(-0.05, 0.05),
                x * 0.05
            ])
            
            self.funnel_particles.append(CloudParticle(
                position, particle_size, particle_opacity, drift_speed, color=funnel_color
            ))
    
    def update(self, dt, wind_direction, wind_strength):
        """Update storm cloud with rotation and special effects."""
        super().update(dt, wind_direction, wind_strength)
        
        # Update rotation angle
        self.rotation_angle += self.rotation_speed * dt
        
        # Update funnel particles if present
        if self.has_funnel:
            for particle in self.funnel_particles:
                # Apply stronger wind and rotation effects
                funnel_wind = wind_direction * wind_strength * 0.5
                
                # Calculate rotational velocity
                rel_x = particle.position[0] - self.center_position[0]
                rel_z = particle.position[2] - self.center_position[2]
                
                # Rotate position around y-axis
                rot_x = rel_x * math.cos(self.rotation_speed * dt) - rel_z * math.sin(self.rotation_speed * dt)
                rot_z = rel_x * math.sin(self.rotation_speed * dt) + rel_z * math.cos(self.rotation_speed * dt)
                
                # Move particle to rotated position
                particle.position[0] = self.center_position[0] + rot_x
                particle.position[2] = self.center_position[2] + rot_z
                
                # Apply other updates
                particle.update(dt, wind_direction, wind_strength * 0.5, self.formation_time)
    
    def draw(self, camera_position, cloud_texture):
        """Draw storm cloud with special rotation and funnel effects."""
        super().draw(camera_position, cloud_texture)
        
        # Draw funnel particles if present
        if self.has_funnel:
            for particle in self.funnel_particles:
                # Skip particles that are too far away
                if np.linalg.norm(particle.position - camera_position) > 2000:
                    continue
                
                glPushMatrix()
                
                # Position the particle in 3D space
                glTranslatef(particle.position[0], particle.position[1], particle.position[2])
                
                # Calculate billboarding matrix
                dx = camera_position[0] - particle.position[0]
                dy = camera_position[1] - particle.position[1]
                dz = camera_position[2] - particle.position[2]
                
                # Distance to camera
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Normalize view vector
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    dz /= dist
                
                # Up vector
                ux, uy, uz = 0, 1, 0  # World up
                
                # Side vector (right) - cross product of up and view direction
                sx = uy * dz - uz * dy
                sy = uz * dx - ux * dz
                sz = ux * dy - uy * dx
                
                # Normalize side vector
                side_len = math.sqrt(sx*sx + sy*sy + sz*sz)
                if side_len > 0:
                    sx /= side_len
                    sy /= side_len
                    sz /= side_len
                
                # New up vector (cross product of view direction and side vector)
                ux = dy * sz - dz * sy
                uy = dz * sx - dx * sz
                uz = dx * sy - dy * sx
                
                # Set billboarding matrix
                billboard_matrix = [
                    sx, ux, -dx, 0,
                    sy, uy, -dy, 0,
                    sz, uz, -dz, 0,
                    0, 0, 0, 1
                ]
                
                glMultMatrixf(billboard_matrix)
                
                # Apply rotation around view axis for variety
                glRotatef(particle.rotation, 0, 0, 1)
                
                # Scale particle
                scale = particle.size
                glScalef(scale, scale, 1.0)
                
                # Set color and opacity
                base_color = particle.color
                glColor4f(base_color[0], base_color[1], base_color[2], particle.opacity)
                
                # Draw the billboard quad
                glCallList(self.display_list)
                
                glPopMatrix()


class CloudSystem:
    """
    Master controller for cloud and weather systems.
    Manages multiple cloud formations and weather conditions.
    """
    def __init__(self, terrain=None):
        self.terrain = terrain
        
        # Cloud and weather parameters
        self.cloud_formations = []
        self.precipitation_particles = []
        self.wind_direction = np.array([1.0, 0.0, 0.0])  # Initial wind direction
        self.wind_strength = 2.0  # Initial wind speed
        
        # Weather state
        self.weather_type = "fair"  # "clear", "fair", "overcast", "stormy", "extreme"
        self.transition_time = 0.0  # Time until next possible weather transition
        self.transition_duration = 0.0  # Duration of ongoing weather transition
        
        # Cloud texture and display lists
        self.cloud_texture = None
        self.precipitation_display_list = None
        self.snow_display_list = None
        self.texture_loaded = False
        
        # Set initial transition time
        self.transition_time = random.uniform(60, 120)
        
        # Initialize with basic clouds
        self.generate_initial_clouds()
    
    def load_textures(self):
        """Load cloud textures for rendering."""
        if self.texture_loaded:
            return

        try:
            # Create cloud texture procedurally
            tex_size = 128

            # Create a pygame surface
            texture_surface = pygame.Surface((tex_size, tex_size), pygame.SRCALPHA)
            texture_surface.fill((0, 0, 0, 0))  # Start with transparent background

            # Create a soft circular gradient for cloud particles
            center_x = center_y = tex_size // 2
            max_radius = tex_size // 2

            # Draw the cloud texture
            for y in range(tex_size):
                for x in range(tex_size):
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < max_radius:
                        # Softer falloff from center - use quadratic function for more defined edges
                        alpha = int(255 * (1.0 - (distance / max_radius) ** 1.5))
                        
                        # Add some noise for texture
                        noise = int(random.uniform(-15, 15))
                        alpha = max(0, min(255, alpha + noise))
                        
                        # Set the pixel color (pure white with variable alpha)
                        texture_surface.set_at((x, y), (255, 255, 255, alpha))

            # Convert pygame surface to string for OpenGL
            texture_data = pygame.image.tostring(texture_surface, "RGBA", True)

            # Generate and bind texture
            self.cloud_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.cloud_texture)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_size, tex_size, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

            # Create display lists for precipitation
            self.create_precipitation_display_lists()

            self.texture_loaded = True
            print("Cloud textures loaded successfully")
        except Exception as e:
            print(f"Error loading cloud textures: {e}")
    
    def create_precipitation_display_lists(self):
        """Create display lists for rain and snow particles."""
        # Rain drop display list
        self.precipitation_display_list = glGenLists(1)
        glNewList(self.precipitation_display_list, GL_COMPILE)
        
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, -1, 0)
        glEnd()
        
        glEndList()
        
        # Snow flake display list
        self.snow_display_list = glGenLists(1)
        glNewList(self.snow_display_list, GL_COMPILE)
        
        glBegin(GL_POINTS)
        glVertex3f(0, 0, 0)
        glEnd()
        
        glEndList()
    
    def generate_initial_clouds(self):
        """Generate initial cloud formations based on weather type."""
        # Clear existing clouds
        for cloud in self.cloud_formations:
            cloud.cleanup()
        self.cloud_formations = []
        
        # Create more background clouds regardless of weather
        self.create_background_clouds()
        
        # Set cloud count based on weather type
        if self.weather_type == "clear":
            cloud_count = 5
        elif self.weather_type == "fair":
            cloud_count = 20
        elif self.weather_type == "overcast":
            cloud_count = 30
        elif self.weather_type == "stormy":
            cloud_count = 25
        elif self.weather_type == "extreme":
            cloud_count = 20  # Fewer but more dramatic clouds
        else:
            cloud_count = 15
        
        # Generate clouds based on weather
        for _ in range(cloud_count):
            # Random position
            x = random.uniform(-1000, 1000)
            z = random.uniform(-1000, 1000)
            
            # Select cloud type based on weather
            if self.weather_type == "clear":
                if random.random() < 0.7:
                    cloud_type = "cirrus"
                    y = random.uniform(150, 250)
                else:
                    cloud_type = "cumulus"
                    y = random.uniform(80, 150)
                size_factor = random.uniform(0.5, 1.0)
            
            elif self.weather_type == "fair":
                r = random.random()
                if r < 0.6:  # Increased probability for cumulus clouds
                    cloud_type = "cumulus"
                    y = random.uniform(60, 120)
                    size_factor = random.uniform(0.8, 1.5)  # Larger clouds
                elif r < 0.9:
                    cloud_type = "cirrus"
                    y = random.uniform(150, 250)
                    size_factor = random.uniform(0.7, 1.2)
                else:
                    cloud_type = "stratus"
                    y = random.uniform(40, 100)
                    size_factor = random.uniform(0.7, 1.0)
            
            elif self.weather_type == "overcast":
                r = random.random()
                if r < 0.7:
                    cloud_type = "stratus"
                    y = random.uniform(30, 80)
                    size_factor = random.uniform(1.0, 1.5)
                else:
                    cloud_type = "cumulus"
                    y = random.uniform(60, 100)
                    size_factor = random.uniform(0.9, 1.4)
            
            elif self.weather_type == "stormy":
                r = random.random()
                if r < 0.6:
                    cloud_type = "cumulonimbus"
                    y = random.uniform(40, 100)
                    size_factor = random.uniform(1.0, 1.8)
                else:
                    cloud_type = "stratus"
                    y = random.uniform(30, 70)
                    size_factor = random.uniform(1.0, 1.5)
            
            elif self.weather_type == "extreme":
                r = random.random()
                if r < 0.6:
                    cloud_type = "tornado"
                    y = random.uniform(40, 80)
                    size_factor = random.uniform(1.5, 2.2)
                else:
                    cloud_type = "cumulonimbus"
                    y = random.uniform(35, 90)
                    size_factor = random.uniform(1.2, 2.0)
            
            # Create cloud formation
            self.add_cloud(cloud_type, [x, y, z], size_factor)
    
    def create_background_clouds(self):
        """Always add some background clouds regardless of weather."""
        # Add scattered background cumulus clouds
        for _ in range(10):
            x = random.uniform(-1500, 1500)
            z = random.uniform(-1500, 1500)
            y = random.uniform(100, 200)
            
            # Small white puffy clouds
            self.add_cloud("cumulus", [x, y, z], size_factor=random.uniform(0.5, 1.0))
        
        # Add high cirrus clouds
        for _ in range(8):
            x = random.uniform(-1800, 1800)
            z = random.uniform(-1800, 1800)
            y = random.uniform(180, 280)
            
            # Wispy high clouds
            self.add_cloud("cirrus", [x, y, z], size_factor=random.uniform(0.8, 1.2))
    
    def add_cloud(self, cloud_type, position, size_factor=1.0):
        """Add a new cloud formation of specified type."""
        if cloud_type == "cumulus":
            cloud = CumulusCloud(position, size_factor)
        elif cloud_type == "cirrus":
            cloud = CirrusCloud(position, size_factor)
        elif cloud_type == "stratus":
            cloud = StratusCloud(position, size_factor)
        elif cloud_type == "cumulonimbus":
            cloud = CumulonimbusCloud(position, size_factor)
        elif cloud_type == "tornado":
            cloud = TornadoCloud(position, size_factor)
        else:
            cloud = CloudFormation(position, size_factor)
        
        self.cloud_formations.append(cloud)
        return cloud
    
    def update(self, dt, time_of_day, player_position):
        """Update all aspects of the cloud and weather system."""
        # Ensure textures are loaded
        if not self.texture_loaded:
            self.load_textures()
            
        # Update weather conditions
        self.update_weather_conditions(dt)
        
        # Update wind
        self.update_wind(dt)
        
        # Update cloud formations
        for cloud in self.cloud_formations:
            cloud.update(dt, self.wind_direction, self.wind_strength)
        
        # Remove clouds that have moved too far away
        self.cloud_formations = [cloud for cloud in self.cloud_formations
                                if np.linalg.norm(cloud.center_position - player_position) < 3000]
        
        # Add new clouds if needed
        if len(self.cloud_formations) < 30:
            # Generate a new cloud at the edge of view distance
            spawn_angle = random.uniform(0, 2 * math.pi)
            spawn_distance = 1500
            x = player_position[0] + math.cos(spawn_angle) * spawn_distance
            z = player_position[2] + math.sin(spawn_angle) * spawn_distance
            
            # Cloud height based on type
            if random.random() < 0.3:
                cloud_type = "cirrus"
                y = random.uniform(150, 250)
            elif random.random() < 0.6 and self.weather_type in ["overcast", "stormy", "extreme"]:
                if self.weather_type == "extreme":
                    cloud_type = "tornado" if random.random() < 0.3 else "cumulonimbus"
                else:
                    cloud_type = "cumulonimbus" if self.weather_type == "stormy" else "stratus"
                y = random.uniform(40, 100)
            else:
                cloud_type = "cumulus"
                y = random.uniform(60, 150)
            
            size_factor = random.uniform(0.8, 1.5)
            self.add_cloud(cloud_type, [x, y, z], size_factor)
        
        # Update precipitation
        self.update_precipitation(dt, player_position)
    
    def update_weather_conditions(self, dt):
        """Update weather conditions based on time and possible transitions."""
        # Adjust transition timer
        self.transition_time -= dt
        
        # Occasionally change weather if the transition timer expires
        if self.transition_time <= 0:
            # Determine possible next weather states based on current weather
            possible_weather = []
            
            if self.weather_type == "clear":
                possible_weather = [("clear", 0.7), ("fair", 0.3)]
            elif self.weather_type == "fair":
                possible_weather = [("clear", 0.3), ("fair", 0.5), ("overcast", 0.2)]
            elif self.weather_type == "overcast":
                possible_weather = [("fair", 0.3), ("overcast", 0.4), ("stormy", 0.3)]
            elif self.weather_type == "stormy":
                possible_weather = [("overcast", 0.5), ("stormy", 0.3), ("extreme", 0.2)]
            elif self.weather_type == "extreme":
                possible_weather = [("stormy", 0.7), ("extreme", 0.3)]
            
            # Pick next weather state
            r = random.random()
            cumulative = 0
            new_weather = self.weather_type  # Default to no change
            
            for weather, probability in possible_weather:
                cumulative += probability
                if r <= cumulative:
                    new_weather = weather
                    break
            
            # If weather is changing, start transition
            if new_weather != self.weather_type:
                self.start_weather_transition(new_weather)
            
            # Set next transition check time (longer if we just transitioned)
            if new_weather != self.weather_type:
                self.transition_time = random.uniform(60, 180)  # 1-3 minutes
            else:
                self.transition_time = random.uniform(30, 90)  # 30-90 seconds
    
    def update_wind(self, dt):
        """Update wind direction and strength based on time of day and weather."""
        # Gradually shift wind direction
        if random.random() < 0.01:  # 1% chance per update
            # Calculate new target direction as slight deviation from current
            angle_change = random.uniform(-0.2, 0.2)  # Radians
            current_angle = math.atan2(self.wind_direction[2], self.wind_direction[0])
            target_angle = current_angle + angle_change
            
            target_x = math.cos(target_angle)
            target_z = math.sin(target_angle)
            
            # Smooth transition to new direction
            self.wind_direction[0] = 0.95 * self.wind_direction[0] + 0.05 * target_x
            self.wind_direction[2] = 0.95 * self.wind_direction[2] + 0.05 * target_z
            
            # Normalize direction vector
            magnitude = math.sqrt(self.wind_direction[0]**2 + self.wind_direction[2]**2)
            if magnitude > 0:
                self.wind_direction[0] /= magnitude
                self.wind_direction[2] /= magnitude
        
        # Update wind strength based on weather
        target_strength = 0.0
        
        if self.weather_type == "clear":
            target_strength = random.uniform(1.0, 3.0)
        elif self.weather_type == "fair":
            target_strength = random.uniform(2.0, 5.0)
        elif self.weather_type == "overcast":
            target_strength = random.uniform(3.0, 8.0)
        elif self.weather_type == "stormy":
            target_strength = random.uniform(8.0, 15.0)
            
            # Add gusts during storms
            if random.random() < 0.05:  # 5% chance of a gust
                gust_strength = random.uniform(15.0, 25.0)
                # For simplicity, just briefly increase wind strength
                target_strength = gust_strength
        elif self.weather_type == "extreme":
            target_strength = random.uniform(15.0, 30.0)
            
            # Frequent strong gusts
            if random.random() < 0.1:  # 10% chance of extreme gust
                gust_strength = random.uniform(25.0, 40.0)
                target_strength = gust_strength
        
        # Smooth transition to target strength
        self.wind_strength = 0.95 * self.wind_strength + 0.05 * target_strength
    
    def start_weather_transition(self, new_weather):
        """Begin transition to new weather state."""
        print(f"Weather transitioning from {self.weather_type} to {new_weather}")
        self.weather_type = new_weather
        self.transition_duration = random.uniform(60, 180)  # 1-3 minutes for complete transition
        
        # Regenerate clouds to match new weather pattern
        self.generate_initial_clouds()
    
    def update_precipitation(self, dt, player_position):
        """Update precipitation particles (rain/snow)."""
        # Remove particles that have reached the ground or moved too far away
        self.precipitation_particles = [p for p in self.precipitation_particles 
                                       if p['lifetime'] > 0 and 
                                       np.linalg.norm(p['position'] - player_position) < 300]
        
        # Update existing particles
        for particle in self.precipitation_particles:
            # Update position based on velocity
            particle['position'] += particle['velocity'] * dt
            
            # Check for ground collision
            if self.terrain:
                ground_height = self.terrain.get_height(particle['position'][0], particle['position'][2])
                if particle['position'][1] <= ground_height:
                    particle['lifetime'] = 0  # Mark for removal
            else:
                # If no terrain, remove particles below a certain height
                if particle['position'][1] < 0:
                    particle['lifetime'] = 0
            
            # Decrease lifetime
            particle['lifetime'] -= dt
        
        # Generate new particles near the player if precipitation is active
        precipitation_clouds = [c for c in self.cloud_formations 
                              if c.can_precipitate and 
                              c.precipitation_intensity > 0 and
                              np.linalg.norm(c.center_position - player_position) < 300]
        
        for cloud in precipitation_clouds:
            # Calculate number of particles to generate
            # Increased from 50 to 100 for more visible precipitation
            particle_count = int(cloud.precipitation_intensity * 100 * dt)
            
            for _ in range(particle_count):
                # Random position under the cloud
                cloud_radius = 80 * cloud.size_factor
                offset_x = random.uniform(-cloud_radius, cloud_radius)
                offset_z = random.uniform(-cloud_radius, cloud_radius)
                
                position = np.array([
                    cloud.center_position[0] + offset_x,
                    cloud.center_position[1] - 10,  # Start below cloud
                    cloud.center_position[2] + offset_z
                ])
                
                # Only add precipitation if position is close enough to player
                if np.linalg.norm(position - player_position) > 300:
                    continue
                
                # Different properties based on precipitation type
                if cloud.precipitation_type == "snow":
                    velocity = np.array([
                        self.wind_direction[0] * self.wind_strength * 0.3,
                        -2.0,  # Slower falling
                        self.wind_direction[2] * self.wind_strength * 0.3
                    ])
                    lifetime = random.uniform(4.0, 8.0)
                    size = random.uniform(0.05, 0.15)
                else:  # Rain
                    velocity = np.array([
                        self.wind_direction[0] * self.wind_strength * 0.2,
                        -10.0,  # Faster falling
                        self.wind_direction[2] * self.wind_strength * 0.2
                    ])
                    lifetime = random.uniform(2.0, 4.0)
                    size = random.uniform(0.5, 1.5)
                
                # Extreme weather has larger precipitation particles
                if self.weather_type == "extreme":
                    size *= 1.5
                
                # Add new precipitation particle
                self.precipitation_particles.append({
                    'position': position,
                    'velocity': velocity,
                    'lifetime': lifetime,
                    'size': size,
                    'type': cloud.precipitation_type
                })
    
    def draw(self, camera_position):
        """Render all cloud formations and precipitation."""
        # Make sure textures are loaded
        if not self.texture_loaded:
            self.load_textures()
        
        # Save OpenGL state
        blend_enabled = glIsEnabled(GL_BLEND)
        depth_mask = glGetBooleanv(GL_DEPTH_WRITEMASK)
        lighting_enabled = glIsEnabled(GL_LIGHTING)
        tex_enabled = glIsEnabled(GL_TEXTURE_2D)
        
        # Set up OpenGL state for cloud rendering
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)  # Don't write to depth buffer for transparent objects
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        
        # Bind cloud texture
        glBindTexture(GL_TEXTURE_2D, self.cloud_texture)
        
        # Sort clouds by distance for better transparency
        sorted_clouds = sorted(
            self.cloud_formations,
            key=lambda c: np.linalg.norm(c.center_position - camera_position),
            reverse=True  # Draw furthest first
        )
        
        # Draw all cloud formations
        for cloud in sorted_clouds:
            dist_to_camera = np.linalg.norm(cloud.center_position - camera_position)
            if dist_to_camera < 3000:  # Only draw clouds within view distance
                cloud.draw(camera_position, self.cloud_texture)
        
        # Draw precipitation if any
        if self.precipitation_particles:
            self.draw_precipitation(camera_position)
        
        # Restore OpenGL state
        if not blend_enabled:
            glDisable(GL_BLEND)
        
        glDepthMask(depth_mask)
        
        if lighting_enabled:
            glEnable(GL_LIGHTING)
        
        if not tex_enabled:
            glDisable(GL_TEXTURE_2D)
    
    def draw_precipitation(self, camera_position):
        """Draw precipitation particles."""
        if not self.precipitation_particles:
            return
            
        # Save states
        blend_enabled = glIsEnabled(GL_BLEND)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        lighting_enabled = glIsEnabled(GL_LIGHTING)
        glDisable(GL_LIGHTING)
        
        point_size = glGetFloatv(GL_POINT_SIZE)
        line_width = glGetFloatv(GL_LINE_WIDTH)
        
        for particle in self.precipitation_particles:
            if particle['type'] == "snow":
                # Draw snowflake
                glPointSize(particle['size'] * 3.0)  # Larger point for snow
                glColor4f(1.0, 1.0, 1.0, 0.8)  # White, slightly transparent
                
                glPushMatrix()
                glTranslatef(particle['position'][0], particle['position'][1], particle['position'][2])
                glCallList(self.snow_display_list)
                glPopMatrix()
            else:
                # Draw raindrop
                glLineWidth(particle['size'])
                
                # Blue-white for raindrops
                glColor4f(0.7, 0.7, 1.0, 0.6)
                
                glPushMatrix()
                glTranslatef(particle['position'][0], particle['position'][1], particle['position'][2])
                glCallList(self.precipitation_display_list)
                glPopMatrix()
        
        # Restore states
        glPointSize(point_size)
        glLineWidth(line_width)
        
        if lighting_enabled:
            glEnable(GL_LIGHTING)
        
        if not blend_enabled:
            glDisable(GL_BLEND)
    
    def get_weather_status(self):
        """Return a string describing the current weather conditions."""
        weather_descriptions = {
            "clear": "Clear Skies",
            "fair": "Fair Weather",
            "overcast": "Overcast",
            "stormy": "Stormy",
            "extreme": "Extreme Weather"
        }
        
        wind_description = "Calm"
        if self.wind_strength > 5:
            wind_description = "Breezy"
        if self.wind_strength > 10:
            wind_description = "Windy"
        if self.wind_strength > 20:
            wind_description = "Strong Winds"
        if self.wind_strength > 30:
            wind_description = "Gale Force"
        
        precipitation = ""
        for cloud in self.cloud_formations:
            if cloud.can_precipitate and cloud.precipitation_intensity > 0:
                if cloud.precipitation_type == "rain":
                    intensity = "Light Rain" if cloud.precipitation_intensity < 0.5 else "Heavy Rain"
                    precipitation = intensity
                    break
                elif cloud.precipitation_type == "snow":
                    intensity = "Light Snow" if cloud.precipitation_intensity < 0.5 else "Heavy Snow"
                    precipitation = intensity
                    break
        
        status = weather_descriptions.get(self.weather_type, "Unknown")
        if precipitation:
            status = f"{status}, {precipitation}"
        status = f"{status}, {wind_description}"
        
        return status
    
    def cleanup(self):
        """Free OpenGL resources."""
        # Release cloud texture
        if self.texture_loaded and self.cloud_texture:
            glDeleteTextures([self.cloud_texture])
        
        # Release display lists for precipitation
        if self.precipitation_display_list:
            glDeleteLists(self.precipitation_display_list, 1)
        if self.snow_display_list:
            glDeleteLists(self.snow_display_list, 1)
        
        # Clean up all cloud formations
        for cloud in self.cloud_formations:
            cloud.cleanup()
        
        self.cloud_formations = []
        self.precipitation_particles = []
        self.texture_loaded = False