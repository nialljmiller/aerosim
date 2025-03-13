"""
Advanced cloud and weather system for flight simulator.
Implements various cloud types, weather conditions, and atmospheric effects
that integrate with the existing celestial system.

Cloud types include:
- Cumulus: Puffy, cotton-like fair weather clouds
- Cirrus: High, thin, wispy clouds
- Stratus: Flat, layered clouds that can produce light precipitation
- Cumulonimbus: Towering storm clouds with lightning and heavy precipitation

Weather conditions:
- Clear: Minimal clouds, mainly cirrus
- Fair: Scattered cumulus clouds
- Overcast: Dense stratus cloud coverage
- Stormy: Cumulonimbus clouds with lightning, rain/snow, and strong winds

The system handles dynamic weather transitions, wind effects, and precipitation (rain/snow).
"""

import numpy as np
import random
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

class CloudParticle:
    """Individual cloud particle/element that makes up larger cloud formations."""
    def __init__(self, position, size, opacity=1.0, drift_speed=None):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.base_size = size  # Store original size for animations
        self.opacity = opacity
        self.base_opacity = opacity  # Store original opacity for animations
        
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
        """Update particle position and appearance based on time and wind."""
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
    def __init__(self, center_position, size_factor=1.0, particle_count=20, opacity=0.7):
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
    
    def draw(self, camera_position):
        """Render the cloud formation."""
        # Distance-based culling
        dist_to_camera = np.linalg.norm(self.center_position - camera_position)
        if dist_to_camera > 2000:
            return
        
        # Ensure display list is compiled
        if not self.compiled:
            self.compile_display_list()
        
        # Save current blend state
        blend_enabled = glIsEnabled(GL_BLEND)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Save current lighting state and disable lighting for clouds
        lighting_enabled = glIsEnabled(GL_LIGHTING)
        glDisable(GL_LIGHTING)
        
        # Save current depth write state and disable for proper transparency
        depth_mask = glGetBooleanv(GL_DEPTH_WRITEMASK)
        glDepthMask(GL_FALSE)
        
        # Enable texture mapping
        tex_enabled = glIsEnabled(GL_TEXTURE_2D)
        glEnable(GL_TEXTURE_2D)
        
        # Draw all particles as billboards
        for particle in self.particles:
            # Skip particles too far from camera
            particle_dist = np.linalg.norm(particle.position - camera_position)
            if particle_dist > 2000:
                continue
            
            # Calculate opacity falloff with distance for better depth effect
            distance_factor = max(0, min(1, 1 - (particle_dist / 2000)))
            effective_opacity = particle.opacity * distance_factor
            
            # Check if this particle has active lightning
            if particle.lightning_active:
                # Bright white/blue color for lightning
                lightning_color = (1.0, 1.0, 1.0)  # White
                glColor4f(lightning_color[0], lightning_color[1], lightning_color[2], 0.9)
            else:
                # Normal cloud color
                glColor4f(self.color[0], self.color[1], self.color[2], effective_opacity)
            
            glPushMatrix()
            
            # Position the particle
            glTranslatef(particle.position[0], particle.position[1], particle.position[2])
            
            # Billboard rotation to face camera
            dx = camera_position[0] - particle.position[0]
            dz = camera_position[2] - particle.position[2]
            rotation_angle = math.degrees(math.atan2(dz, dx))
            glRotatef(rotation_angle, 0, 1, 0)
            
            # Scale to particle size
            glScalef(particle.size, particle.size, particle.size)
            
            # Draw the billboard
            glCallList(self.display_list)
            
            glPopMatrix()
        
        # Restore texture state
        if not tex_enabled:
            glDisable(GL_TEXTURE_2D)
        
        # Restore depth mask state
        glDepthMask(depth_mask)
        
        # Restore lighting state
        if lighting_enabled:
            glEnable(GL_LIGHTING)
        
        # Restore blending state
        if not blend_enabled:
            glDisable(GL_BLEND)
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.compiled and self.display_list:
            glDeleteLists(self.display_list, 1)
            self.compiled = False
    
    def get_weather_status(self):
        """Return current weather information for display."""
        weather_names = {
            "clear": "Clear Skies",
            "fair": "Fair Weather",
            "overcast": "Overcast",
            "stormy": "Stormy"
        }
        
        status = f"{weather_names.get(self.weather_type, self.weather_type)}"
        
        # Add wind information
        wind_speed_kts = self.wind_strength * 1.944  # Convert to knots (approximate)
        
        # Get wind direction as compass heading
        wind_dir = math.atan2(self.wind_direction[2], self.wind_direction[0])
        wind_dir_deg = (math.degrees(wind_dir) + 90) % 360  # Convert to compass heading
        
        # Convert to cardinal direction
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        ix = int((wind_dir_deg + 22.5) % 360 / 45)
        cardinal = dirs[ix]
        
        # Wind speed classification
        wind_desc = "Calm"
        if wind_speed_kts > 33:
            wind_desc = "Gale"
        elif wind_speed_kts > 20:
            wind_desc = "Strong"
        elif wind_speed_kts > 10:
            wind_desc = "Moderate"
        elif wind_speed_kts > 3:
            wind_desc = "Light"
        
        status += f" | Wind: {cardinal} {wind_speed_kts:.0f}kts ({wind_desc})"
        
        # Add precipitation if present
        has_rain = any(p['type'] == 'rain' for p in self.precipitation_particles)
        has_snow = any(p['type'] == 'snow' for p in self.precipitation_particles)
        
        if has_rain:
            rain_intensity = sum(c.precipitation_intensity for c in self.cloud_formations 
                               if c.can_precipitate and c.precipitation_type == 'rain')
            if rain_intensity > 0.6:
                status += " | Heavy Rain"
            elif rain_intensity > 0.3:
                status += " | Moderate Rain"
            else:
                status += " | Light Rain"
                
        if has_snow:
            snow_intensity = sum(c.precipitation_intensity for c in self.cloud_formations 
                               if c.can_precipitate and c.precipitation_type == 'snow')
            if snow_intensity > 0.6:
                status += " | Heavy Snow"
            elif snow_intensity > 0.3:
                status += " | Moderate Snow"
            else:
                status += " | Light Snow"
        
        return status


class CumulusCloud(CloudFormation):
    """Puffy, cotton-like cumulus clouds."""
    def __init__(self, center_position, size_factor=1.0, particle_count=30):
        super().__init__(center_position, size_factor, particle_count, opacity=0.7)
        self.color = (1.0, 1.0, 1.0)  # Bright white
    
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
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity))


class CirrusCloud(CloudFormation):
    """High, thin, wispy cirrus clouds."""
    def __init__(self, center_position, size_factor=1.0, particle_count=15):
        super().__init__(center_position, size_factor, particle_count, opacity=0.3)
        self.color = (0.95, 0.95, 1.0)  # Slightly blue-white
        
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
            
            # Cirrus clouds are semi-transparent
            particle_opacity = opacity * random.uniform(0.3, 0.7)
            
            # Add drift in the streak direction
            drift_speed = np.array([dx, 0, dz]) * random.uniform(0.5, 1.5)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity, drift_speed))
    
    def update(self, dt, wind_direction, wind_strength):
        """Update cirrus clouds with enhanced wind effect."""
        # Cirrus clouds move faster with the wind
        effective_wind_strength = wind_strength * self.wind_multiplier
        super().update(dt, wind_direction, effective_wind_strength)


class StratusCloud(CloudFormation):
    """Flat, layered stratus clouds covering large areas."""
    def __init__(self, center_position, size_factor=1.0, particle_count=40):
        super().__init__(center_position, size_factor, particle_count, opacity=0.5)
        self.color = (0.9, 0.9, 0.9)  # Light gray
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
            
            # Slightly larger particles
            particle_size = random.uniform(15, 35) * self.size_factor
            
            # More uniform opacity
            particle_opacity = opacity * random.uniform(0.7, 0.9)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity))


class CumulonimbusCloud(CloudFormation):
    """Towering storm clouds with lightning and heavy precipitation."""
    def __init__(self, center_position, size_factor=1.0, particle_count=50):
        super().__init__(center_position, size_factor, particle_count, opacity=0.8)
        self.color = (0.6, 0.6, 0.7)  # Darker gray
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
            
            # Size varies by position
            if y > 0.6 * height:  # Anvil top
                particle_size = random.uniform(15, 30) * self.size_factor
            else:  # Main body and base
                particle_size = random.uniform(20, 40) * self.size_factor
            
            # Opacity - more opaque in center
            distance_from_center = math.sqrt(x*x + z*z) / width
            particle_opacity = opacity * (1.0 - 0.3 * distance_from_center)
            
            self.particles.append(CloudParticle(position, particle_size, particle_opacity))


class CloudSystem:
    """
    Master controller for cloud and weather systems.
    Manages multiple cloud formations, weather conditions, and precipitation.
    """
    def __init__(self, terrain=None):
        self.terrain = terrain
        
        # Cloud and weather parameters
        self.cloud_formations = []
        self.precipitation_particles = []
        self.wind_direction = np.array([1.0, 0.0, 0.0])  # Initial wind direction
        self.wind_strength = 2.0  # Initial wind speed
        
        # Weather state
        self.weather_type = "clear"  # "clear", "fair", "overcast", "stormy"
        self.transition_time = 0.0  # Time until next possible weather transition
        self.transition_duration = 0.0  # Duration of ongoing weather transition
        
        # Cloud texture and display lists
        self.cloud_texture = None
        self.precipitation_display_list = None
        self.snow_display_list = None
        self.texture_loaded = False
        
        # Set initial transition time
        self.transition_time = random.uniform(30, 60)
        
        # Initialize with basic clouds
        self.generate_initial_clouds()
    
    def load_textures(self):
        """Load cloud textures for rendering."""
        if self.texture_loaded:
            return
        
        # Create cloud texture procedurally
        tex_size = 128
        texture_data = np.zeros((tex_size, tex_size, 4), dtype=np.uint8)
        
        # Create a soft circular gradient for cloud particles
        center_x = center_y = tex_size // 2
        max_radius = tex_size // 2
        
        for y in range(tex_size):
            for x in range(tex_size):
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < max_radius:
                    # Soft falloff from center
                    alpha = int(255 * (1.0 - distance / max_radius) ** 2)
                    
                    # Add some noise for texture
                    noise = int(random.uniform(-20, 20))
                    alpha = max(0, min(255, alpha + noise))
                    
                    texture_data[y, x] = [255, 255, 255, alpha]
        
        # Create texture with pygame
        texture_surface = pygame.Surface((tex_size, tex_size), SRCALPHA)
        pygame.surfarray.blit_array(texture_surface, texture_data)
        texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
        
        # Generate and bind texture
        self.cloud_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.cloud_texture)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Upload texture data and generate mipmaps
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_size, tex_size, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        # Create display lists for precipitation
        self.create_precipitation_display_lists()
        
        self.texture_loaded = True
    
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
        self.cloud_formations = []
        
        # Set cloud count based on weather type
        if self.weather_type == "clear":
            cloud_count = 5
        elif self.weather_type == "fair":
            cloud_count = 15
        elif self.weather_type == "overcast":
            cloud_count = 25
        elif self.weather_type == "stormy":
            cloud_count = 20
        else:
            cloud_count = 10
        
        # Generate clouds
        for _ in range(cloud_count):
            # Random position
            x = random.uniform(-1000, 1000)
            z = random.uniform(-1000, 1000)
            
            # Height depends on cloud type
            y = random.uniform(50, 200)
            
            # Select cloud type based on weather
            if self.weather_type == "clear":
                if random.random() < 0.7:
                    cloud_type = "cirrus"
                else:
                    cloud_type = "cumulus"
                size_factor = random.uniform(0.5, 1.0)
            
            elif self.weather_type == "fair":
                r = random.random()
                if r < 0.5:
                    cloud_type = "cumulus"
                    size_factor = random.uniform(0.8, 1.3)
                elif r < 0.8:
                    cloud_type = "cirrus"
                    size_factor = random.uniform(0.7, 1.2)
                else:
                    cloud_type = "stratus"
                    size_factor = random.uniform(0.7, 1.0)
                    y = random.uniform(40, 100)  # Lower altitude for stratus
            
            elif self.weather_type == "overcast":
                r = random.random()
                if r < 0.7:
                    cloud_type = "stratus"
                    size_factor = random.uniform(1.0, 1.5)
                    y = random.uniform(30, 80)
                else:
                    cloud_type = "cumulus"
                    size_factor = random.uniform(0.9, 1.4)
            
            elif self.weather_type == "stormy":
                r = random.random()
                if r < 0.6:
                    cloud_type = "cumulonimbus"
                    size_factor = random.uniform(1.0, 1.8)
                    y = random.uniform(40, 100)
                else:
                    cloud_type = "stratus"
                    size_factor = random.uniform(1.0, 1.5)
                    y = random.uniform(30, 70)
            
            # Create cloud formation
            self.add_cloud(cloud_type, [x, y, z], size_factor)
    
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
        else:
            cloud = CloudFormation(position, size_factor)
        
        self.cloud_formations.append(cloud)
        return cloud
        
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
            particle_count = int(cloud.precipitation_intensity * 50 * dt)
            
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
                
                # Add new precipitation particle
                self.precipitation_particles.append({
                    'position': position,
                    'velocity': velocity,
                    'lifetime': lifetime,
                    'size': size,
                    'type': cloud.precipitation_type
                })
    
    def update_wind(self, dt, time_of_day):
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
        
        # Update wind strength based on weather and time
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
                gust_duration = random.uniform(1.0, 3.0)
                # For simplicity, just briefly increase wind strength
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
    
    def update_weather_conditions(self, dt, time_of_day, player_position):
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
                possible_weather = [("overcast", 0.6), ("stormy", 0.4)]
            
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
        
        # Update wind direction and strength
        self.update_wind(dt, time_of_day)
        
        # Update existing clouds
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
            elif random.random() < 0.6 and self.weather_type in ["overcast", "stormy"]:
                cloud_type = "cumulonimbus" if self.weather_type == "stormy" else "stratus"
                y = random.uniform(40, 100)
            else:
                cloud_type = "cumulus"
                y = random.uniform(60, 150)
            
            size_factor = random.uniform(0.8, 1.5)
            self.add_cloud(cloud_type, [x, y, z], size_factor)