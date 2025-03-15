"""
Mountain Biome for Flight Simulator
Creates dramatic mountain ranges with peaks, ridges, and valleys.
"""

import numpy as np
import math
import sys
import os

# Ensure biomes directory is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biomes.biome_base import BaseBiome

class MountainBiome(BaseBiome):
    """Mountain biome with rugged peaks and deep valleys."""
    
    def __init__(self):
        super().__init__()
        self.name = "mountain"
        
        # Mountain-specific parameters
        self.height_scale = 400.0  # Mountains are taller than other biomes
        self.noise_scale = 0.007  # Slightly larger scale for mountains
        
        # Ridge noise parameters for jagged mountain ridges
        self.ridge_scale = 0.25
        self.ridge_weight = 1.8
        
        # Mountain colors
        self.colors = {
            'rock': (0.5, 0.48, 0.45),      # Gray-brown
            'scree': (0.6, 0.55, 0.5),      # Light gray
            'forest': (0.2, 0.4, 0.2),      # Dark green
            'snow': (0.95, 0.95, 0.98)      # White with slight blue tint
        }
        
        # Height thresholds for mountain terrain types
        self.thresholds = {
            'forest_line': 20.0,    # Below this is forest
            'tree_line': 35.0,      # Transition from forest to rock
            'snow_line': 55.0       # Above this is snow
        }
    
    def _ridge_noise(self, noise_func, x, y, octaves=4, persistence=0.5):
        """Generate ridge noise for mountain ridges and peaks."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            # Get raw noise
            n = noise_func.noise2d(x * frequency, y * frequency)
            
            # Convert to ridge shape
            n = 1.0 - abs(n)
            n = n * n  # Sharpen ridges
            
            value += n * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            frequency *= 2.0
            
        return value / max_value
    
    def get_height(self, world_x, world_z):
        # Safe input handling
        try:
            world_x = float(world_x)
            world_z = float(world_z)
        except (TypeError, ValueError):
            print(f"Error in {self.name} biome: invalid coordinates")
            return 0.0
            
        try:
            """Generate mountain terrain height."""
            # Import noise here to avoid circular imports
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Base terrain with multiple noise layers
            base_noise = noise.fractal(
                world_x * self.noise_scale, 
                world_z * self.noise_scale,
                octaves=6,
                persistence=0.6
            )
            
            # Medium detail for hills and smaller features
            medium_detail = noise.fractal(
                world_x * self.noise_scale * 4, 
                world_z * self.noise_scale * 4,
                octaves=4,
                persistence=0.5
            ) * 0.25
            
            # Generate ridge noise for mountain peaks
            # Add domain warping for more natural-looking ridges
            warp_x = noise.fractal(
                world_x * self.noise_scale * 0.5 + 1000, 
                world_z * self.noise_scale * 0.5 + 1000,
                octaves=3
            ) * 200  # Strong warping for dramatic mountains
            
            warp_z = noise.fractal(
                world_x * self.noise_scale * 0.5 + 2000, 
                world_z * self.noise_scale * 0.5 + 2000,
                octaves=3
            ) * 200
            
            # Apply domain warping to create ridge features
            ridge_noise = self._ridge_noise(
                noise,
                (world_x + warp_x) * self.noise_scale * self.ridge_scale,
                (world_z + warp_z) * self.noise_scale * self.ridge_scale,
                octaves=5,
                persistence=0.65
            )
            
            # Generate erosion patterns for valleys between mountains
            erosion = noise.fractal(
                world_x * self.noise_scale * 2, 
                world_z * self.noise_scale * 2,
                octaves=4,
                persistence=0.4
            ) * 0.3
            
            # Combine noise components
            combined = (base_noise + medium_detail) * self.height_scale
            
            # Apply ridge noise to create peaks
            # Use non-linear blending to create more dramatic features
            mountain_factor = combined * 0.3 + ridge_noise * self.ridge_weight * 50
            
            # Scale down valleys with erosion
            mountain_factor -= erosion * 20
            
            # Ensure minimum terrain height
            return max(0.5, mountain_factor)



        except Exception as e:
            print(f"Error in {self.name} biome get_height: {e}")
            return 0.0  # Safe default
            
    def get_color(self, world_x, world_z, height):
        """Get mountain terrain colors based on height and other factors."""
        try:
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)  # Use fixed, safe seed
            
            # Generate variation noise
            variation_noise = abs(noise.noise2d(world_x * 0.01, world_z * 0.01))
            
            # Determine base color by height
            if height > self.thresholds['snow_line']:
                # Snow-covered peaks
                color = self.colors['snow']
            elif height > self.thresholds['tree_line']:
                # Rocky slopes above tree line
                if variation_noise > 0.6:
                    # Areas with more exposed rock
                    color = self.colors['rock']
                else:
                    # Areas with some vegetation/lichen
                    blend = (height - self.thresholds['tree_line']) / (self.thresholds['snow_line'] - self.thresholds['tree_line'])
                    color = self.interpolate_color(self.colors['rock_vegetated'], self.colors['rock'], blend)
            else:
                # Forested lower slopes
                if variation_noise > 0.7:
                    # Denser forest
                    color = self.colors['forest_dense']
                else:
                    # Mixed forest
                    color = self.colors['forest']
            
            # Add subtle variation based on position
            return self.add_color_variation(color, world_x, world_z)
        except Exception as e:
            # Return a safe default color if there's an error
            return (0.5, 0.5, 0.5)  # Default gray