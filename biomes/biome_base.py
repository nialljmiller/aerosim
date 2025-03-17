"""
Updated Base Biome Class for Flight Simulator Terrain
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
        
        Default implementation provides a simple height field with noise.
        """
        # Apply some error handling to handle type issues
        try:
            world_x = float(world_x)
            world_z = float(world_z)
        except (TypeError, ValueError):
            # If conversion fails, return a safe value
            return self.color_thresholds['medium']
            
        try:
            # Import noise here to avoid circular imports
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Generate simple rolling hills
            base_noise = noise.fractal(
                world_x * self.noise_scale, 
                world_z * self.noise_scale,
                octaves=4,
                persistence=0.5
            )
            
            # Add medium detail
            medium_noise = noise.fractal(
                world_x * self.noise_scale * 3, 
                world_z * self.noise_scale * 3,
                octaves=2,
                persistence=0.4
            ) * 0.3
            
            # Combine and scale
            combined = (base_noise + medium_noise) * self.height_scale
            
            # Apply a base height offset and ensure minimum height
            return max(1.0, combined + 15.0)
        except Exception as e:
            # If noise generation fails, return a safe default
            print(f"Error generating biome height: {e}")
            return self.color_thresholds['medium']
    
    def get_color(self, world_x, world_z, height):
        """
        Generate terrain color for this biome at given world coordinates.
        Must be implemented by subclasses.
        
        Default implementation provides a simple height-based color gradient.
        """
        try:
            # Import noise here to avoid circular imports
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Add some variation to color
            color_noise = noise.noise2d(world_x * self.noise_scale * 10, world_z * self.noise_scale * 10) * 0.1
            
            # Simple height-based coloring
            if height < self.color_thresholds['low']:
                # Water
                base_color = (0.1, 0.3, 0.7)
            elif height < self.color_thresholds['medium']:
                # Lower ground
                t = (height - self.color_thresholds['low']) / (self.color_thresholds['medium'] - self.color_thresholds['low'])
                base_color = self.interpolate_color((0.2, 0.5, 0.2), (0.4, 0.6, 0.3), t)
            else:
                # Higher ground
                t = min(1.0, (height - self.color_thresholds['medium']) / (self.color_thresholds['high'] - self.color_thresholds['medium']))
                base_color = self.interpolate_color((0.4, 0.6, 0.3), (0.6, 0.6, 0.5), t)
            
            # Apply noise variation
            color = (
                max(0.0, min(1.0, base_color[0] + color_noise)),
                max(0.0, min(1.0, base_color[1] + color_noise)),
                max(0.0, min(1.0, base_color[2] + color_noise))
            )
            
            return color
        except Exception as e:
            # Return default color if coloring fails
            print(f"Error generating biome color: {e}")
            return self.base_color
    
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