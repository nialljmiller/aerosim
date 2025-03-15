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

class WaterSystem:
    """Renders a large water plane with transparency and wave effects."""
    
    def __init__(self, water_level=-15.0, size=10000.0, wave_speed=0.5, water_color=(0.2, 0.5, 0.8, 0.5)):
        """Initialize water system.
        
        Args:
            water_level: Y-coordinate where water appears (now set much lower to -15.0)
            size: Size of the water plane
            wave_speed: Speed of wave animation
            water_color: RGBA color of water (with alpha for transparency)
        """
        self.water_level = water_level  # This is the base water level
        self.size = size
        self.wave_speed = wave_speed
        self.water_color = water_color
        self.animation_time = 0.0
        
        # Wave parameters
        self.wave_amplitude = 0.05  # Height of waves
        self.wave_frequency = 0.1   # Spatial frequency of waves
        
        # Display list for efficient rendering
        self.display_list = None
        self.display_list_compiled = False
        
        # Minimum height difference to render water when camera is close
        self.min_camera_height_diff = 5.0  # Only render water if camera is at least this far from water level
    
    def compile_display_list(self):
        """Compile water mesh into OpenGL display list."""
        self.display_list = glGenLists(1)
        
        glNewList(self.display_list, GL_COMPILE)
        
        # Draw a large grid for water
        segments = 32
        segment_size = self.size / segments
        
        glBegin(GL_QUADS)
        
        for i in range(-segments//2, segments//2):
            for j in range(-segments//2, segments//2):
                # Calculate vertex positions
                x1 = i * segment_size
                z1 = j * segment_size
                x2 = (i + 1) * segment_size
                z2 = (j + 1) * segment_size
                
                # Base vertices at y=0 (water level will be applied during rendering)
                glVertex3f(x1, 0.0, z1)
                glVertex3f(x2, 0.0, z1)
                glVertex3f(x2, 0.0, z2)
                glVertex3f(x1, 0.0, z2)
        
        glEnd()
        
        glEndList()
        self.display_list_compiled = True
    
    def calculate_wave_height(self, x, z):
        """Calculate wave height at a specific point."""
        # Simple wave calculation using sine functions
        wave1 = math.sin(x * self.wave_frequency + self.animation_time)
        wave2 = math.sin(z * self.wave_frequency * 0.8 + self.animation_time * 0.9)
        wave3 = math.sin((x + z) * self.wave_frequency * 0.3 + self.animation_time * 1.1)
        
        # Combine waves for more natural effect
        return (wave1 + wave2 + wave3) * self.wave_amplitude / 3.0
    
    def update(self, dt, camera_position):
        """Update water animation.
        
        Args:
            dt: Time delta in seconds
            camera_position: Current camera position (to center water on camera)
        """
        # Update animation time for wave effect
        self.animation_time += dt * self.wave_speed
    
    def draw(self, camera_position):
        """Draw water with transparency.
        
        Args:
            camera_position: Current camera position (to center water on camera)
        """
        # Skip rendering the second layer if camera is too close to water level
        camera_height_diff = abs(camera_position[1] - self.water_level)
        if 0 < camera_height_diff < self.min_camera_height_diff and camera_position[1] > self.water_level:
            # Skip rendering when camera is just above water level
            # This prevents the "second layer" visual artifact
            return
        
        if not self.display_list_compiled:
            self.compile_display_list()
        
        # Save current OpenGL state
        blend_enabled = glIsEnabled(GL_BLEND)
        depth_mask = glGetBooleanv(GL_DEPTH_WRITEMASK)
        current_color = glGetFloatv(GL_CURRENT_COLOR)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Make water semi-transparent by not writing to depth buffer when alpha < 1.0
        glDepthMask(GL_FALSE)
        
        # Set up wave effect
        glPushMatrix()
        
        # Center water grid on camera (important: only in X and Z, not Y)
        cam_x = math.floor(camera_position[0] / 100.0) * 100.0
        cam_z = math.floor(camera_position[2] / 100.0) * 100.0
        
        # Apply absolute water level position
        glTranslatef(cam_x, self.water_level, cam_z)
        
        # Set water color with alpha
        glColor4f(*self.water_color)
        
        # Draw water plane
        glCallList(self.display_list)
        
        glPopMatrix()
        
        # Restore OpenGL state
        if not blend_enabled:
            glDisable(GL_BLEND)
        glDepthMask(depth_mask)
        glColor4fv(current_color)
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.display_list_compiled:
            glDeleteLists(self.display_list, 1)
            self.display_list_compiled = False