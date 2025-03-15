"""
Base Biome Class for Flight Simulator Terrain
Defines the interface that all biomes must implement and provides common functionality.
"""

import numpy as np

class BaseBiome:
    """Base class for all terrain biomes."""
    
    def __init__(self):
        """Initialize biome with default parameters."""
        # Default parameters for any biome - can be overridden by subclasses
        self.name = "base"
        self.noise_scale = 0.005  # Controls terrain frequency
        self.height_scale = 40.0  # Controls overall height
        
        # Default colors
        self.base_color = (0.5, 0.5, 0.5)  # Default gray
        
        # Height thresholds for color mapping
        self.color_thresholds = {
            'low': 0.0,
            'medium': 20.0,
            'high': 40.0
        }
    
    def get_height(self, world_x, world_z):
        """
        Generate terrain height for this biome at given world coordinates.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Biomes must implement get_height()")
    
    def get_color(self, world_x, world_z, height):
        """
        Generate terrain color for this biome at given world coordinates.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Biomes must implement get_color()")
    
    def interpolate_color(self, color1, color2, t):
        """Linearly interpolate between two colors."""
        return (
            color1[0] * (1-t) + color2[0] * t,
            color1[1] * (1-t) + color2[1] * t,
            color1[2] * (1-t) + color2[2] * t
        )
    

    def add_color_variation(self, base_color, world_x, world_z, variation=0.05):
        """Add slight variation to terrain colors based on position."""
        try:
            # Use a deterministic hash approach rather than random
            hash_val = abs(int(world_x * 1000 + world_z * 2000)) % 10000
            
            # Generate a consistent variation based on the hash
            offset = ((hash_val / 10000) - 0.5) * 2 * variation
            
            # Apply variation to each color channel
            r = max(0.0, min(1.0, base_color[0] + offset))
            g = max(0.0, min(1.0, base_color[1] + offset))
            b = max(0.0, min(1.0, base_color[2] + offset))
            
            return (r, g, b)
        except Exception as e:
            return base_color  # Return original color if there's an error

            
    def get_feature_noise(self, world_x, world_z, noise_func, scale=1.0, offset=0.0):
        """Calculate noise value for terrain features with configurable parameters."""
        return noise_func(
            (world_x + offset) * self.noise_scale * scale, 
            (world_z + offset) * self.noise_scale * scale
        )
    
    def generate_common_noise(self, world_x, world_z, noise_func, octaves=6, persistence=0.5):
        """Generate common noise patterns for consistent terrain features across biomes."""
        return noise_func.fractal(
            world_x * self.noise_scale,
            world_z * self.noise_scale,
            octaves=octaves,
            persistence=persistence
        )