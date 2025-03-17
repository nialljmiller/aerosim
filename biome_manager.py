"""
Biome Manager System for Flight Simulator
Handles biome selection, generation, and transitioning between biomes
"""

import os
import sys
import numpy as np
import math
import random

# Ensure biomes directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import biome types - try/except for each to handle import errors gracefully
try:
    from biomes.float_island_biome import FloatingIslandsBiome
    has_floating_islands = True
except ImportError:
    print("Warning: FloatingIslandsBiome not available")
    has_floating_islands = False

try:
    from biomes.mountain_biome import MountainBiome
    has_mountains = True
except ImportError:
    print("Warning: MountainBiome not available")
    has_mountains = False

try:
    from biomes.plains_biome import PlainsBiome
    has_plains = True
except ImportError:
    print("Warning: PlainsBiome not available")
    has_plains = False

try:
    from biomes.volcanic_biome import VolcanicBiome
    has_volcanic = True
except ImportError:
    print("Warning: VolcanicBiome not available")
    has_volcanic = False

try:
    from biomes.caves_biome import CrystalCavesBiome
    has_caves = True
except ImportError:
    print("Warning: CrystalCavesBiome not available")
    has_caves = False


class BiomeManager:
    """Manages biome creation, transitions, and selection for the terrain system."""
    
    def __init__(self, noise_generator, biome_scale=2000.0):
        """Initialize the biome manager.
        
        Args:
            noise_generator: PerlinNoise instance for consistent noise generation
            biome_scale: Controls how large each biome region is (larger = bigger biomes)
        """
        self.noise = noise_generator
        self.biome_scale = biome_scale
        self.cached_biomes = {}  # Cache for biome instances by type
        self.biome_map_cache = {}  # Cache for biome selection at grid coordinates
        
        # Default is floating islands if available, otherwise plains
        if has_floating_islands:
            self.default_biome_type = "floating_islands"
        elif has_plains:
            self.default_biome_type = "plains" 
        else:
            self.default_biome_type = "default"
        
        # Get available biome types
        self.available_biomes = []
        
        if has_floating_islands:
            self.available_biomes.append("floating_islands")
        if has_mountains:
            self.available_biomes.append("mountain")
        if has_plains:
            self.available_biomes.append("plains")
        if has_volcanic:
            self.available_biomes.append("volcanic")
        if has_caves:
            self.available_biomes.append("caves")
            
        if not self.available_biomes:
            # If no biomes are available, use a default fallback
            from biomes.biome_base import BaseBiome
            self.available_biomes.append("default")
            self.default_biome_type = "default"
            
        print(f"Available biomes: {self.available_biomes}")
        print(f"Default biome: {self.default_biome_type}")
            
        # Create and cache all biome instances up front
        self._initialize_biomes()
    
    def _initialize_biomes(self):
        """Create instances of all available biomes."""
        for biome_type in self.available_biomes:
            self._get_biome_instance(biome_type)
            
    def _get_biome_instance(self, biome_type):
        """Get or create a biome instance of the specified type."""
        # Return cached instance if available
        if biome_type in self.cached_biomes:
            return self.cached_biomes[biome_type]
        
        # Otherwise create new instance
        try:
            if biome_type == "floating_islands" and has_floating_islands:
                biome = FloatingIslandsBiome()
            elif biome_type == "mountain" and has_mountains:
                biome = MountainBiome()
            elif biome_type == "plains" and has_plains:
                biome = PlainsBiome()
            elif biome_type == "volcanic" and has_volcanic:
                biome = VolcanicBiome()
            elif biome_type == "caves" and has_caves:
                biome = CrystalCavesBiome()
            else:
                # Fallback to base biome
                from biomes.biome_base import BaseBiome
                biome = BaseBiome()
                biome_type = "default"
                
            # Cache the instance
            self.cached_biomes[biome_type] = biome
            return biome
            
        except Exception as e:
            print(f"Error creating biome {biome_type}: {e}")
            # Fallback to default
            if self.default_biome_type != biome_type and self.default_biome_type in self.cached_biomes:
                return self.cached_biomes[self.default_biome_type]
            
            # Last resort fallback
            from biomes.biome_base import BaseBiome
            fallback = BaseBiome()
            self.cached_biomes["default"] = fallback
            return fallback
    
    def _get_biome_blend_weights(self, world_x, world_z):
        """
        Get biome weights for blending at the given world coordinates.
        Returns a dict of {biome_type: weight} with weights summing to 1.0
        """
        # Get biome grid coordinates
        grid_x = int(world_x / self.biome_scale)
        grid_z = int(world_z / self.biome_scale)
        
        # Get relative position within this grid cell (0.0 - 1.0)
        cell_x = (world_x / self.biome_scale) - grid_x
        cell_z = (world_z / self.biome_scale) - grid_z
        
        # Check if we're near a grid boundary (within 10% of cell size)
        transition_zone = 0.1
        near_x_boundary = cell_x < transition_zone or cell_x > (1.0 - transition_zone)
        near_z_boundary = cell_z < transition_zone or cell_z > (1.0 - transition_zone)
        
        # If not near any boundary, just return the primary biome
        if not (near_x_boundary or near_z_boundary):
            primary_biome = self._get_biome_at_grid(grid_x, grid_z)
            return {primary_biome: 1.0}
            
        # Otherwise, we need to blend between neighboring biomes
        biome_weights = {}
        
        # Calculate weight factors based on distance from boundaries
        x_factor = min(cell_x, 1.0 - cell_x) / transition_zone if near_x_boundary else 1.0
        z_factor = min(cell_z, 1.0 - cell_z) / transition_zone if near_z_boundary else 1.0
        
        # Get neighboring grid cells that we need to blend with
        neighboring_cells = []
        
        # Current cell is always included
        neighboring_cells.append((grid_x, grid_z, 1.0))
        
        # X-axis neighbors if near x boundary
        if near_x_boundary:
            next_x = grid_x + 1 if cell_x > 0.5 else grid_x - 1
            neighboring_cells.append((next_x, grid_z, 1.0 - x_factor))
            
            # Corner neighbors if also near z boundary
            if near_z_boundary:
                next_z = grid_z + 1 if cell_z > 0.5 else grid_z - 1
                neighboring_cells.append((next_x, next_z, (1.0 - x_factor) * (1.0 - z_factor)))
        
        # Z-axis neighbors if near z boundary
        if near_z_boundary:
            next_z = grid_z + 1 if cell_z > 0.5 else grid_z - 1
            neighboring_cells.append((grid_x, next_z, 1.0 - z_factor))
        
        # Get biome types and accumulate weights
        for gx, gz, weight in neighboring_cells:
            biome_type = self._get_biome_at_grid(gx, gz)
            
            if biome_type in biome_weights:
                biome_weights[biome_type] += weight
            else:
                biome_weights[biome_type] = weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(biome_weights.values())
        if total_weight > 0:
            for biome_type in biome_weights:
                biome_weights[biome_type] /= total_weight
        
        return biome_weights
    
    def _get_biome_at_grid(self, grid_x, grid_z):
        """Get the biome type at the specified grid coordinates."""
        # Check cache first
        grid_key = (grid_x, grid_z)
        if grid_key in self.biome_map_cache:
            return self.biome_map_cache[grid_key]
        
        # Determine biome based on noise
        # Use multiple noise layers for better distribution
        primary_noise = self.noise.noise2d(grid_x * 0.5, grid_z * 0.5)
        secondary_noise = self.noise.noise2d(grid_x * 0.5 + 100, grid_z * 0.5 + 100)
        
        # Combine noise values
        combined_noise = (primary_noise + secondary_noise) * 0.5
        # Map to 0.0 - 1.0 range
        normalized_noise = (combined_noise + 1.0) * 0.5
        
        # Default to floating islands biome with higher probability
        if normalized_noise < 0.3 and has_floating_islands:
            biome_type = "floating_islands"
        else:
            # Distribute remaining biomes evenly
            remaining_biomes = [b for b in self.available_biomes if b != "floating_islands"]
            if not remaining_biomes:
                biome_type = self.default_biome_type
            else:
                # Map the remaining noise range to available biomes
                biome_index = int((normalized_noise - 0.3) / 0.7 * len(remaining_biomes))
                biome_index = max(0, min(len(remaining_biomes) - 1, biome_index))
                biome_type = remaining_biomes[biome_index]
        
        # Cache the result
        self.biome_map_cache[grid_key] = biome_type
        return biome_type
    
    def get_height(self, world_x, world_z):
        """Get terrain height at the specified world coordinates."""
        # Get biome weights for this location
        biome_weights = self._get_biome_blend_weights(world_x, world_z)
        
        # If only one biome, return its height directly
        if len(biome_weights) == 1:
            biome_type = list(biome_weights.keys())[0]
            biome = self._get_biome_instance(biome_type)
            return biome.get_height(world_x, world_z)
        
        # Otherwise blend heights from multiple biomes
        blended_height = 0.0
        for biome_type, weight in biome_weights.items():
            biome = self._get_biome_instance(biome_type)
            biome_height = biome.get_height(world_x, world_z)
            blended_height += biome_height * weight
            
        return blended_height
    
    def get_color(self, world_x, world_z, height):
        """Get terrain color at the specified world coordinates."""
        # Get biome weights for this location
        biome_weights = self._get_biome_blend_weights(world_x, world_z)
        
        # If only one biome, return its color directly
        if len(biome_weights) == 1:
            biome_type = list(biome_weights.keys())[0]
            biome = self._get_biome_instance(biome_type)
            return biome.get_color(world_x, world_z, height)
        
        # Otherwise blend colors from multiple biomes
        r, g, b = 0.0, 0.0, 0.0
        for biome_type, weight in biome_weights.items():
            biome = self._get_biome_instance(biome_type)
            biome_color = biome.get_color(world_x, world_z, height)
            
            # Add weighted color components
            r += biome_color[0] * weight
            g += biome_color[1] * weight
            b += biome_color[2] * weight
            
        return (r, g, b)
    
    def get_biome_at_position(self, world_x, world_z):
        """Get the primary biome type at the given world position."""
        grid_x = int(world_x / self.biome_scale)
        grid_z = int(world_z / self.biome_scale)
        return self._get_biome_at_grid(grid_x, grid_z)