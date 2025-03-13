import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import datetime
import pygame
from pygame.locals import *

class CelestialBody:
    """Base class for celestial bodies like sun and moon."""
    def __init__(self, radius, distance, texture_path=None):
        self.radius = radius        # Visual radius of the celestial body
        self.distance = distance    # Distance from world center
        self.position = np.array([0.0, 0.0, 0.0])  # Current position
        self.texture_id = None      # OpenGL texture ID
        self.slices = 32            # Sphere detail
        self.stacks = 32
        self.display_list = None
        
        # Load texture if provided
        if texture_path:
            self.load_texture(texture_path)
        
        # Create display list for efficient rendering
        self.create_display_list()
    
    def load_texture(self, texture_path):
        """Load texture for the celestial body."""
        try:
            # Load image using pygame
            texture_surface = pygame.image.load(texture_path)
            texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
            width, height = texture_surface.get_size()
            
            # Generate and bind texture
            self.texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            
            # Upload texture data and generate mipmaps
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
        except Exception as e:
            print(f"Failed to load texture: {e}")
            self.texture_id = None
    
    def create_display_list(self):
        """Create a display list for efficient rendering."""
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Draw a sphere
        quadric = gluNewQuadric()
        gluQuadricTexture(quadric, GL_TRUE)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        gluSphere(quadric, self.radius, self.slices, self.stacks)
        
        if self.texture_id:
            glDisable(GL_TEXTURE_2D)
        
        gluDeleteQuadric(quadric)
        glEndList()
    
    def draw(self):
        """Draw the celestial body at its current position."""
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Draw using display list
        glCallList(self.display_list)
        
        glPopMatrix()
    
    def update_position(self, azimuth, altitude):
        """
        Update position based on spherical coordinates.
        azimuth: angle in horizontal plane (0 = north, 90 = east)
        altitude: angle above horizon (in degrees)
        """
        # Convert angles to radians
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # Calculate position using spherical coordinates
        x = self.distance * np.cos(altitude_rad) * np.sin(azimuth_rad)
        y = self.distance * np.sin(altitude_rad)
        z = self.distance * np.cos(altitude_rad) * np.cos(azimuth_rad)
        
        self.position = np.array([x, y, z])


class Sun(CelestialBody):
    """Sun implementation with light source."""
    def __init__(self, radius=20.0, distance=1000.0, texture_path=None):
        super().__init__(radius, distance, texture_path)
        self.light_id = GL_LIGHT0
        self.ambient = [0.15, 0.15, 0.15, 1.0]
        self.diffuse = [1.0, 1.0, 0.95, 1.0]
        self.specular = [1.0, 1.0, 0.95, 1.0]
    
    def update_light(self):
        """Update the light source position and parameters."""
        # Set light position (convert to homogeneous coordinates)
        light_position = [self.position[0], self.position[1], self.position[2], 0.0]  # Directional light
        glLightfv(self.light_id, GL_POSITION, light_position)
        
        # Update light color based on sun altitude
        # When sun is low, light becomes more reddish
        altitude = np.degrees(np.arctan2(self.position[1], np.sqrt(self.position[0]**2 + self.position[2]**2)))
        
        # Transition from regular daylight to sunset/sunrise colors
        if altitude < 10:
            # Transition factor: 0 = below horizon, 1 = at 10 degrees
            t = max(0, min(1, altitude / 10.0))
            
            # Sunset/sunrise colors (warmer, redder)
            sunset_diffuse = [1.0, 0.5, 0.2, 1.0]
            sunset_ambient = [0.1, 0.05, 0.05, 1.0]
            
            # Interpolate between sunset and regular colors
            current_diffuse = [
                sunset_diffuse[0] * (1-t) + self.diffuse[0] * t,
                sunset_diffuse[1] * (1-t) + self.diffuse[1] * t,
                sunset_diffuse[2] * (1-t) + self.diffuse[2] * t,
                1.0
            ]
            current_ambient = [
                sunset_ambient[0] * (1-t) + self.ambient[0] * t,
                sunset_ambient[1] * (1-t) + self.ambient[1] * t,
                sunset_ambient[2] * (1-t) + self.ambient[2] * t,
                1.0
            ]
            
            # Night time (if below horizon)
            if altitude < 0:
                # Scale down light intensity based on how far below horizon
                night_factor = max(0, 1 + altitude / 10.0)  # 0 at -10 degrees, 1 at horizon
                
                # Reduce ambient and diffuse for night
                current_ambient = [a * night_factor * 0.5 for a in current_ambient]
                current_diffuse = [d * night_factor for d in current_diffuse]
        else:
            current_diffuse = self.diffuse
            current_ambient = self.ambient
        
        # Apply light properties
        glLightfv(self.light_id, GL_AMBIENT, current_ambient)
        glLightfv(self.light_id, GL_DIFFUSE, current_diffuse)
        glLightfv(self.light_id, GL_SPECULAR, self.specular)
    
    def draw(self):
        """Override draw to add a glow effect around the sun."""
        # First draw the sun itself
        super().draw()
        
        # Then add a glow effect (corona)
        # Save current blending state
        blend_enabled = glIsEnabled(GL_BLEND)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Draw corona as a larger, transparent sphere
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Make corona larger than the sun
        corona_scale = 1.5
        
        # Use a dedicated shader or a simple implementation with alpha gradient
        glColor4f(1.0, 1.0, 0.8, 0.3)  # Yellowish, semi-transparent
        
        quadric = gluNewQuadric()
        gluSphere(quadric, self.radius * corona_scale, 16, 16)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
        
        # Restore blending state
        if not blend_enabled:
            glDisable(GL_BLEND)


class Moon(CelestialBody):
    """Moon implementation with phase calculation."""
    def __init__(self, radius=5.0, distance=800.0, texture_path=None):
        super().__init__(radius, distance, texture_path)
        self.phase = 0.0  # 0.0 to 1.0 (new moon to full to new)
    
    def update_phase(self, sun_position):
        """
        Calculate moon phase based on relative positions of sun and moon.
        Phase is determined by the angle between sun-earth and moon-earth vectors.
        """
        # Vector from origin (earth) to moon
        moon_vec = self.position / np.linalg.norm(self.position)
        
        # Vector from origin (earth) to sun
        sun_vec = sun_position / np.linalg.norm(sun_position)
        
        # Angle between these vectors gives us the phase
        dot_product = np.dot(moon_vec, sun_vec)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert to phase value (0 to 1)
        # When sun and moon are in same direction: New Moon (0.0)
        # When sun and moon are in opposite directions: Full Moon (0.5)
        self.phase = angle / (2 * np.pi)
    
    def draw(self):
        """Override draw to incorporate moon phase in rendering."""
        # For simple implementation, we'll just adjust the moon's brightness
        # based on its phase
        
        # Calculate brightness factor (1.0 at full moon, lower at new moon)
        # This is a simplified model; real moon illumination is more complex
        brightness = abs(self.phase - 0.5) * 2.0  # Max at 0.5 (full moon)
        
        # Save current color
        current_color = glGetFloatv(GL_CURRENT_COLOR)
        
        # Set color based on phase
        glColor3f(brightness, brightness, brightness)
        
        # Draw the moon
        super().draw()
        
        # Restore original color
        glColor4fv(current_color)


class SkyDome:
    """Sky dome with dynamic day/night color transition."""
    def __init__(self, radius=1500.0, resolution=20):
        self.radius = radius
        self.resolution = resolution
        self.display_list = None
        
        # Sky colors for different times of day
        self.day_zenith = [0.5, 0.7, 1.0]
        self.day_horizon = [0.7, 0.8, 1.0]
        self.sunset_zenith = [0.2, 0.2, 0.5]
        self.sunset_horizon = [1.0, 0.5, 0.2]
        self.night_zenith = [0.05, 0.05, 0.1]
        self.night_horizon = [0.1, 0.1, 0.2]
        
        self.stars = []
        self.generate_stars(2000)  # Generate 2000 stars
        
        self.create_display_list()
    
    def generate_stars(self, count):
        """Generate random stars in the sky dome."""
        self.stars = []
        for _ in range(count):
            # Generate random spherical coordinates
            theta = 2 * np.pi * np.random.random()  # Azimuth
            phi = np.arccos(2 * np.random.random() - 1)  # Elevation
            
            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.cos(phi)  # y is up
            z = np.sin(phi) * np.sin(theta)
            
            # Star brightness (some stars are brighter)
            brightness = 0.5 + 0.5 * np.random.random()
            
            # Star color (mostly white with slight variations)
            r = 0.9 + 0.1 * np.random.random()
            g = 0.9 + 0.1 * np.random.random()
            b = 0.9 + 0.1 * np.random.random()
            
            self.stars.append({
                'position': [x, y, z],
                'brightness': brightness,
                'color': [r, g, b]
            })
    
    def create_display_list(self):
        """Create a display list for the sky dome."""
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        # Draw the dome as a hemisphere of triangles
        slices = self.resolution
        stacks = self.resolution
        
        # Use triangle strips for efficiency
        for i in range(stacks):
            # Elevation angles
            phi1 = np.pi * 0.5 * i / stacks
            phi2 = np.pi * 0.5 * (i + 1) / stacks
            
            glBegin(GL_TRIANGLE_STRIP)
            
            for j in range(slices + 1):
                # Azimuth angles
                theta = 2 * np.pi * j / slices
                
                # Calculate vertices for both stacks
                for phi in [phi1, phi2]:
                    # Convert spherical to Cartesian coordinates
                    x = self.radius * np.sin(phi) * np.cos(theta)
                    y = self.radius * np.cos(phi)  # y is up
                    z = self.radius * np.sin(phi) * np.sin(theta)
                    
                    # Normalize position for color interpolation (0 at horizon, 1 at zenith)
                    t = np.cos(phi)  # 0 at horizon, 1 at zenith
                    
                    # Gradient color (will be replaced in update_colors)
                    glColor3f(t, t, t)
                    
                    # Vertex
                    glVertex3f(x, y, z)
            
            glEnd()
        
        glEndList()
        
        # Stars are rendered separately and not part of the display list
        # because they need to stay fixed regardless of time of day
    
    def update_colors(self, sun_altitude):
        """
        Update sky colors based on sun's altitude.
        sun_altitude: angle in degrees (-90 to +90)
        """
        # Determine sky colors based on sun's position
        if sun_altitude > 10:
            # Daytime
            zenith = self.day_zenith
            horizon = self.day_horizon
        elif sun_altitude > -10:
            # Sunset/sunrise transition
            t = (sun_altitude + 10) / 20.0  # 0 at -10 degrees, 1 at +10 degrees
            
            # Interpolate between night/sunset and sunset/day
            if sun_altitude > 0:
                # Sunrise to day transition
                zenith = [
                    self.sunset_zenith[0] * (1-t) + self.day_zenith[0] * t,
                    self.sunset_zenith[1] * (1-t) + self.day_zenith[1] * t,
                    self.sunset_zenith[2] * (1-t) + self.day_zenith[2] * t
                ]
                horizon = [
                    self.sunset_horizon[0] * (1-t) + self.day_horizon[0] * t,
                    self.sunset_horizon[1] * (1-t) + self.day_horizon[1] * t,
                    self.sunset_horizon[2] * (1-t) + self.day_horizon[2] * t
                ]
            else:
                # Night to sunrise transition
                t = t * 2  # Rescale so t is 0 at -10 and 1 at 0 degrees
                zenith = [
                    self.night_zenith[0] * (1-t) + self.sunset_zenith[0] * t,
                    self.night_zenith[1] * (1-t) + self.sunset_zenith[1] * t,
                    self.night_zenith[2] * (1-t) + self.sunset_zenith[2] * t
                ]
                horizon = [
                    self.night_horizon[0] * (1-t) + self.sunset_horizon[0] * t,
                    self.night_horizon[1] * (1-t) + self.sunset_horizon[1] * t,
                    self.night_horizon[2] * (1-t) + self.sunset_horizon[2] * t
                ]
        else:
            # Night
            zenith = self.night_zenith
            horizon = self.night_horizon
    
        return zenith, horizon
    
    def draw(self, sun_altitude, camera_position):
        """
        Draw the sky dome with appropriate colors.
        sun_altitude: sun's angle above/below horizon in degrees
        camera_position: position of the camera to center the sky dome
        """
        # Update colors based on sun position
        zenith, horizon = self.update_colors(sun_altitude)
        
        # Save current matrix and lighting state
        glPushMatrix()
        lighting_enabled = glIsEnabled(GL_LIGHTING)
        glDisable(GL_LIGHTING)  # Sky is self-illuminated
        
        # Move the dome to camera position
        glTranslatef(camera_position[0], camera_position[1], camera_position[2])
        
        # Draw the sky dome with gradient
        stack_count = self.resolution
        slice_count = self.resolution
        
        for i in range(stack_count):
            # Elevation angles
            phi1 = np.pi * 0.5 * i / stack_count
            phi2 = np.pi * 0.5 * (i + 1) / stack_count
            
            glBegin(GL_TRIANGLE_STRIP)
            
            for j in range(slice_count + 1):
                # Azimuth angles
                theta = 2 * np.pi * j / slice_count
                
                # Calculate vertices for both stacks
                for phi in [phi1, phi2]:
                    # Convert spherical to Cartesian coordinates
                    x = self.radius * np.sin(phi) * np.cos(theta)
                    y = self.radius * np.cos(phi)  # y is up
                    z = self.radius * np.sin(phi) * np.sin(theta)
                    
                    # Normalize position for color interpolation (0 at horizon, 1 at zenith)
                    t = np.cos(phi)  # 0 at horizon, 1 at zenith
                    
                    # Interpolate between horizon and zenith colors
                    r = horizon[0] * (1-t) + zenith[0] * t
                    g = horizon[1] * (1-t) + zenith[1] * t
                    b = horizon[2] * (1-t) + zenith[2] * t
                    
                    glColor3f(r, g, b)
                    glVertex3f(x, y, z)
            
            glEnd()
        
        # Draw stars if it's dark enough
        star_visibility = max(0, min(1, -(sun_altitude) / 12.0))
        if star_visibility > 0:
            # Enable point size and blending for stars
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            
            # Set point size
            original_point_size = glGetFloatv(GL_POINT_SIZE)
            glPointSize(2.0)
            
            glBegin(GL_POINTS)
            for star in self.stars:
                # Apply star visibility based on sun altitude
                alpha = star['brightness'] * star_visibility
                glColor4f(
                    star['color'][0],
                    star['color'][1],
                    star['color'][2],
                    alpha
                )
                # Position
                glVertex3f(
                    self.radius * star['position'][0],
                    self.radius * star['position'][1],
                    self.radius * star['position'][2]
                )
            glEnd()
            
            # Restore point size
            glPointSize(original_point_size)
            
            # Restore blending state
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_POINT_SMOOTH)
            glDisable(GL_BLEND)
        
        # Restore original matrix and lighting state
        if lighting_enabled:
            glEnable(GL_LIGHTING)
        glPopMatrix()


class CelestialSystem:
    """Master controller for sun, moon, and day/night cycle."""
    def __init__(self):
        # Create celestial objects
        self.sun = Sun(radius=20.0, distance=1000.0)
        self.moon = Moon(radius=5.0, distance=800.0)
        self.sky_dome = SkyDome(radius=1500.0)
        
        # Time tracking
        self.time_scale = 1.0  # 1.0 = real time, higher for faster day/night cycle
        self.day_length = 24 * 60 * 60  # Length of day in seconds
        self.current_time = 12 * 60 * 60  # Start at noon
        
        # Earth's axial tilt (for season simulation)
        self.axial_tilt = 23.5
        
        # Current in-game date
        self.date = datetime.datetime(2023, 6, 21)  # Summer solstice
        
        # Lunar orbit parameters
        self.lunar_orbit_period = 29.5 * 24 * 60 * 60  # Synodic month in seconds
        self.lunar_orbit_inclination = 5.14  # Moon's orbital inclination in degrees
        
        # Track if initialized
        self.initialized = False
    
    def initialize(self):
        """Initialize the celestial system."""
        # Placeholder for texture loading
        # self.sun.load_texture("textures/sun.png")
        # self.moon.load_texture("textures/moon.png")
        
        # Set initial positions
        self.update(0)
        self.initialized = True
    
    def update(self, dt):
        """
        Update celestial objects based on time progression.
        
        Args:
            dt: Time delta in seconds
        """
        # Update current time
        self.current_time += dt * self.time_scale
        self.current_time %= self.day_length  # Wrap around after a full day
        
        # Convert to hour angle (0 to 360 degrees)
        hour_angle = (self.current_time / self.day_length) * 360.0
        
        # Calculate sun position
        # Solar declination includes axial tilt for seasonal variations
        day_of_year = self.date.timetuple().tm_yday
        solar_declination = self.axial_tilt * np.sin(np.radians((day_of_year - 81) * 360 / 365))
        
        # Sun's altitude and azimuth
        sun_altitude = np.arcsin(
            np.sin(np.radians(solar_declination)) * np.sin(np.radians(90)) +
            np.cos(np.radians(solar_declination)) * np.cos(np.radians(90)) * np.cos(np.radians(hour_angle))
        )
        sun_altitude = np.degrees(sun_altitude)
        
        # Sun azimuth (0 = north, 90 = east)
        sun_azimuth = np.arccos(
            (np.sin(np.radians(solar_declination)) - np.sin(np.radians(sun_altitude)) * np.sin(np.radians(90))) /
            (np.cos(np.radians(sun_altitude)) * np.cos(np.radians(90)))
        )
        sun_azimuth = np.degrees(sun_azimuth)
        
        # Adjust azimuth based on hour angle
        if hour_angle > 180:
            sun_azimuth = 360 - sun_azimuth
        
        # Update sun position
        self.sun.update_position(sun_azimuth, sun_altitude)
        self.sun.update_light()
        
        # Calculate moon position
        # Moon follows a similar path but offset by lunar phase
        lunar_phase_offset = (self.current_time / self.lunar_orbit_period) * 360.0
        moon_hour_angle = (hour_angle + 180 + lunar_phase_offset) % 360
        
        # Moon's declination includes its orbital inclination
        moon_declination = solar_declination + self.lunar_orbit_inclination * np.sin(np.radians(lunar_phase_offset))
        
        # Moon's altitude and azimuth
        moon_altitude = np.arcsin(
            np.sin(np.radians(moon_declination)) * np.sin(np.radians(90)) +
            np.cos(np.radians(moon_declination)) * np.cos(np.radians(90)) * np.cos(np.radians(moon_hour_angle))
        )
        moon_altitude = np.degrees(moon_altitude)
        
        moon_azimuth = np.arccos(
            (np.sin(np.radians(moon_declination)) - np.sin(np.radians(moon_altitude)) * np.sin(np.radians(90))) /
            (np.cos(np.radians(moon_altitude)) * np.cos(np.radians(90)))
        )
        moon_azimuth = np.degrees(moon_azimuth)
        
        # Adjust azimuth based on hour angle
        if moon_hour_angle > 180:
            moon_azimuth = 360 - moon_azimuth
        
        # Update moon position
        self.moon.update_position(moon_azimuth, moon_altitude)
        
        # Update moon phase
        self.moon.update_phase(self.sun.position)
    
    def draw(self, camera_position):
        """Draw all celestial objects."""
        if not self.initialized:
            self.initialize()
        
        # Get sun's altitude for sky coloring
        sun_altitude = np.degrees(np.arctan2(self.sun.position[1], 
                                             np.sqrt(self.sun.position[0]**2 + self.sun.position[2]**2)))
        
        # Draw sky dome first (background)
        self.sky_dome.draw(sun_altitude, camera_position)
        
        # Draw sun and moon
        # Only draw if above horizon (optimization)
        if sun_altitude > -5:
            self.sun.draw()
        
        # Get moon's altitude
        moon_altitude = np.degrees(np.arctan2(self.moon.position[1], 
                                              np.sqrt(self.moon.position[0]**2 + self.moon.position[2]**2)))
        
        # Draw moon if above horizon
        if moon_altitude > -5:
            self.moon.draw()
    
    def get_time_of_day_string(self):
        """Get current time of day as a formatted string."""
        total_seconds = int(self.current_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        am_pm = "AM" if hours < 12 else "PM"
        display_hours = hours if hours <= 12 else hours - 12
        if display_hours == 0:
            display_hours = 12
            
        return f"{display_hours:02d}:{minutes:02d}:{seconds:02d} {am_pm}"
    
    def set_time(self, hours):
        """Set time to a specific hour (0-23)."""
        self.current_time = hours * 60 * 60
    
    def set_time_progression(self, scale):
        """Set time progression speed (1.0 = real time)."""
        self.time_scale = scale
    
    def is_night(self):
        """Check if it's currently night time."""
        # Get sun's altitude
        sun_altitude = np.degrees(np.arctan2(self.sun.position[1], 
                                             np.sqrt(self.sun.position[0]**2 + self.sun.position[2]**2)))
        return sun_altitude < 0