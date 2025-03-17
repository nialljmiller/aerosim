"""
Floating Islands Biome for Flight Simulator - Enhanced for More Islands
Creates an otherworldly landscape with islands floating in the sky.
"""

import numpy as np
import math
import sys
import os

# Ensure biomes directory is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biomes.biome_base import BaseBiome

class FloatingIslandsBiome(BaseBiome):
    """Surreal biome with floating islands of various sizes."""
    
    def __init__(self):
        super().__init__()
        self.name = "floating_islands"
        
        # Floating islands specific parameters - DRAMATICALLY ENHANCED
        self.height_scale = 120.0           # Increased for more dramatic height variation
        self.noise_scale = 0.006
        self.island_density = 0.9           # Very high density for many more islands
        self.base_altitude = 80.0           # Higher base altitude for truly floating islands
        
        # Parameters for island shape generation
        self.island_scale = 0.015
        self.detail_scale = 0.05
        
        # Island size variations - broader range
        self.min_island_size = 15.0        
        self.max_island_size = 100.0        # Larger maximum size for more dramatic islands
        
        # Waterfalls and features
        self.waterfall_probability = 0.4    # More waterfalls
        
        # Floating crystal parameters
        self.has_crystals = True
        self.crystal_density = 0.3
        
        # Hanging chains and structures
        self.has_hanging_features = True
        self.hanging_feature_probability = 0.4
        
        # Colors - more vibrant and fantastical
        self.colors = {
            'grass': (0.3, 0.8, 0.3),       # Brighter green
            'edge': (0.7, 0.6, 0.4),        # Brown cliff edges
            'rock': (0.6, 0.5, 0.5),        # Stone formations
            'water': (0.3, 0.7, 1.0),       # Vibrant blue water
            'waterfall': (0.7, 0.9, 1.0),   # Bright waterfall
            'moss': (0.2, 0.7, 0.3),        # Bright moss
            'exotic': (0.7, 0.3, 0.9),      # Vibrant purple vegetation
            'crystal': (0.8, 0.5, 1.0),     # Purple crystals
            'gold': (1.0, 0.9, 0.4)         # Gold-colored elements
        }
    
    def _island_shape(self, x, y, noise_func, island_id, size_factor):
        """Generate the shape of a single floating island."""
        # Core shape (roughly circular but with noise-based distortion)
        distance = math.sqrt(x**2 + y**2)
        
        # Base profile is a radial gradient with falloff
        base_radius = size_factor * self.min_island_size + (1.0 - size_factor) * self.max_island_size
        
        # Use a sharper falloff for more defined island edges
        radial_value = max(0.0, 1.0 - (distance / base_radius)**2)  # Squared for sharper edges
        
        # Apply edge noise for more organic shapes
        edge_noise = noise_func.fractal(
            x * self.detail_scale + island_id * 100, 
            y * self.detail_scale + island_id * 100,
            octaves=4,
            persistence=0.5
        ) * 0.4 + 0.6  # Scale to 0.6-1.0 range for more dramatic edges
        
        # Combine radial gradient with noise for the island shape
        shape = radial_value * edge_noise
        
        # Add steep cliffs on the edges with more dramatic falloff
        if 0.05 < shape < 0.5:  # Wider range for cliff zone
            cliff_steepness = 8.0  # Increased for more dramatic cliffs
            shape = shape * cliff_steepness
        
        # Flatten the top slightly for better landing areas
        if shape > 0.7:
            shape = 0.7 + (shape - 0.7) * 0.6  # More flat area on top
        
        return shape
    
    def get_height(self, world_x, world_z):
        """Generate floating islands terrain height."""
        # Make sure inputs are float
        try:
            world_x = float(world_x)
            world_z = float(world_z)
        except (TypeError, ValueError):
            print(f"Error in {self.name} biome: invalid coordinates")
            return 0.0
        
        try:
            # Import noise here to avoid circular imports
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Generate noise to determine island positions
            island_field = noise.fractal(
                world_x * self.noise_scale * 0.2, 
                world_z * self.noise_scale * 0.2,
                octaves=2,
                persistence=0.5
            )
            
            # Noise for vertical placement of islands
            altitude_noise = noise.fractal(
                world_x * self.noise_scale * 0.1, 
                world_z * self.noise_scale * 0.1,
                octaves=1
            )
            
            # Determine if current point could be an island
            island_value = 0.0
            max_island_contribution = 0.0
            island_id = 0
            
            # Use a smaller cell size to create more islands
            cell_size = self.max_island_size * 1.0  # Reduced for more islands
            
            # Search a larger area for island influences
            for ix in range(-3, 4):  # Increased range from (-2,3) to (-3,4)
                for iz in range(-3, 4):  # Increased range for more islands
                    # Create a grid of potential island centers
                    cell_x = math.floor(world_x / cell_size)
                    cell_z = math.floor(world_z / cell_size)
                    
                    center_x = (cell_x + ix) * cell_size + noise.noise2d(cell_x + ix, cell_z + iz) * cell_size * 0.5
                    center_z = (cell_z + iz) * cell_size + noise.noise2d(cell_x + ix + 50, cell_z + iz + 50) * cell_size * 0.5
                    
                    # Distance to island center
                    dx = world_x - center_x
                    dz = world_z - center_z
                    
                    # Generate a unique id for this potential island
                    current_island_id = int(abs(center_x * 1000 + center_z))
                    
                    # Determine island size based on its id - more variation
                    size_noise = abs(noise.noise2d(current_island_id * 0.01, 0))
                    size_factor = size_noise ** 0.6  # Adjust distribution for more larger islands
                    
                    # The threshold is now much lower due to increased island_density
                    island_threshold = 1.0 - self.island_density
                    
                    # Modified condition for more islands - add island_id influence
                    island_specific_noise = (noise.noise2d(current_island_id * 0.05, current_island_id * 0.05) + 1) * 0.5
                    if island_field > island_threshold or island_specific_noise > 0.4:  # Easier to spawn islands
                        # Calculate shape and height of this island
                        island_shape = self._island_shape(dx, dz, noise, current_island_id, size_factor)
                        
                        # Only contribute if this point is on the island
                        if island_shape > 0:
                            # More dramatic height variation based on island ID and size
                            variation = noise.noise2d(current_island_id * 0.02, 0) * 40.0  # More variation
                            base_height = self.base_altitude + altitude_noise * 60.0 + variation
                            island_height = base_height + size_factor * 80.0  # More height variation
                            
                            # Calculate the contribution of this island to the current point
                            contribution = island_shape * island_height
                            
                            # Keep track of the strongest island influence
                            if contribution > max_island_contribution:
                                max_island_contribution = contribution
                                island_value = contribution
                                island_id = current_island_id
            
            # Add details to island surfaces
            if island_value > 0:
                # Surface detail noise - more pronounced
                detail = noise.fractal(
                    world_x * self.noise_scale * 4, 
                    world_z * self.noise_scale * 4,
                    octaves=4,
                    persistence=0.5
                ) * 7.0  # Increased for more dramatic surface detail
                
                # Only add detail to the top of islands
                if island_value > 0.7 * max_island_contribution:
                    island_value += detail
                
                # Create occasional crater lakes on larger islands
                crater_noise = noise.fractal(
                    world_x * self.noise_scale * 2 + 500, 
                    world_z * self.noise_scale * 2 + 500,
                    octaves=2
                )
                
                if crater_noise > 0.7 and island_value > 0.9 * max_island_contribution:
                    # Create a depression for a lake - deeper
                    depression = (crater_noise - 0.7) * 35.0  # Increased from 25.0
                    island_value -= depression
                
                # Create occasional spikes/towers on islands
                spike_noise = noise.fractal(
                    world_x * self.noise_scale * 10 + 800, 
                    world_z * self.noise_scale * 10 + 800,
                    octaves=2
                )
                
                if spike_noise > 0.92 and island_value > 0.8 * max_island_contribution:
                    # Create dramatic spires
                    spike_height = (spike_noise - 0.92) * 20.0 * 70.0  # Very tall spires
                    island_value += spike_height
            
            # Add occasional smaller floating rocks
            if island_value <= 0:
                small_rock_noise = noise.fractal(
                    world_x * self.noise_scale * 4 + 300, 
                    world_z * self.noise_scale * 4 + 300,
                    octaves=2
                )
                
                if small_rock_noise > 0.96:
                    # Small floating rock
                    rock_size = (small_rock_noise - 0.96) * 25.0 * 6.0
                    rock_height_base = self.base_altitude + altitude_noise * 30.0
                    rock_height_offset = noise.noise2d(world_x * 0.1, world_z * 0.1) * 60.0
                    island_value = rock_size + rock_height_base + rock_height_offset
            
            # Add hanging crystals below some islands
            if island_value > 0 and self.has_crystals:
                crystal_noise = noise.fractal(
                    world_x * self.noise_scale * 5 + 700, 
                    world_z * self.noise_scale * 5 + 700,
                    octaves=2
                )
                
                if crystal_noise > 0.85 and island_value > 0.6 * max_island_contribution:
                    # Detect bottom of island
                    bottom_y = island_value - 10.0  # Offset from island bottom
                    
                    # Crystal stalactite parameters
                    crystal_size = (crystal_noise - 0.85) * 20.0 * 15.0
                    crystal_length = crystal_size * 2.0
                    
                    # Hanging crystal (negative height extending down)
                    if bottom_y - crystal_length < 0:
                        # No negative height
                        pass
            
            # If not on an island, return a very low value for empty sky
            if island_value <= 0:
                return -100.0  # Below water level, will be replaced by water plane
            
            # Add a small global elevation to ensure islands are above water level
            return island_value + 5.0
            
        except Exception as e:
            print(f"Error in floating islands get_height: {e}")
            return 0.0  # Safe default
    
    def get_color(self, world_x, world_z, height):
        """Get floating island terrain colors."""
        try:
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Return water color if below water level
            if height < 0.0:
                return self.colors['water']
            
            # Generate noise for terrain features
            detail_noise = noise.fractal(
                world_x * self.noise_scale * 6, 
                world_z * self.noise_scale * 6,
                octaves=3,
                persistence=0.5
            )
            
            # Edge detection - use gradient of height
            epsilon = 0.5
            h_dx = self.get_height(world_x + epsilon, world_z)
            h_dz = self.get_height(world_x, world_z + epsilon)
            gradient_mag = math.sqrt((h_dx - height)**2 + (h_dz - height)**2)
            
            # Check for steep edges (cliffs)
            is_cliff = gradient_mag > 1.0
            
            # Check for flat areas that could be water pools
            is_flat = gradient_mag < 0.1
            
            # Generate exotic vegetation patterns
            exotic_pattern = noise.fractal(
                world_x * self.noise_scale * 3 + 1000, 
                world_z * self.noise_scale * 3 + 1000,
                octaves=2
            )
            
            # Generate crystal patterns
            crystal_pattern = noise.fractal(
                world_x * self.noise_scale * 8 + 2000, 
                world_z * self.noise_scale * 8 + 2000,
                octaves=2
            )
            
            # Generate golden patterns for special features
            gold_pattern = noise.fractal(
                world_x * self.noise_scale * 10 + 3000, 
                world_z * self.noise_scale * 10 + 3000,
                octaves=2
            )
            
            # Determine color based on features
            if is_cliff:
                # Cliff faces and edges
                if crystal_pattern > 0.92:
                    # Crystal veins in rock
                    return self.colors['crystal']
                elif gold_pattern > 0.96:
                    # Gold veins in rock
                    return self.colors['gold']
                else:
                    return self.colors['edge']
            elif is_flat and detail_noise < 0.3 and height > 0.0:
                # Small pools of water on top of islands
                return self.colors['water']
            elif crystal_pattern > 0.9:
                # Crystal formations
                return self.colors['crystal']
            elif gold_pattern > 0.95:
                # Gold deposits
                return self.colors['gold']
            elif exotic_pattern > 0.7:
                # Areas of exotic vegetation
                return self.colors['exotic']
            elif detail_noise < 0.4:
                # Mossy areas
                return self.colors['moss']
            else:
                # Default island grass
                return self.colors['grass']
                
        except Exception as e:
            print(f"Error in floating islands get_color: {e}")
            return (0.3, 0.7, 0.3)  # Default green