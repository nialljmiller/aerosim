"""
Crystal Caves Biome for Flight Simulator
Creates an alien landscape with massive crystal formations and glass-like terrain.
"""

import numpy as np
import math
import sys
import os

# Ensure biomes directory is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biomes.biome_base import BaseBiome

class CrystalCavesBiome(BaseBiome):
    """Surreal crystal landscape with massive gem formations."""
    
    def __init__(self):
        super().__init__()
        self.name = "crystal_caves"
        
        # Crystal biome specific parameters
        self.height_scale = 60.0
        self.noise_scale = 0.005
        
        # Crystal formation parameters
        self.crystal_density = 0.5
        self.min_crystal_size = 5.0
        self.max_crystal_size = 40.0
        self.crystal_sharpness = 3.0  # Controls how pointed crystals are
        
        # Underground cave parameters
        self.cave_ground_level = 15.0  # Base level for the cave floor
        self.ceiling_height = 100.0    # Height of the cave ceiling
        
        # Colors
        self.colors = {
            'amethyst': (0.6, 0.3, 0.8),     # Purple crystal
            'emerald': (0.1, 0.8, 0.4),      # Green crystal
            'ruby': (0.8, 0.1, 0.2),         # Red crystal
            'sapphire': (0.1, 0.4, 0.8),     # Blue crystal
            'quartz': (0.8, 0.8, 0.9),       # White crystal
            'amber': (0.9, 0.6, 0.1),        # Orange crystal
            'cave_wall': (0.2, 0.2, 0.25),   # Dark cave background
            'water': (0.1, 0.6, 0.8, 0.7)    # Glowing water (with alpha)
        }
        
        # Create own noise instance to avoid the dependency issue
        try:
            from terrain import PerlinNoise
            self.noise = PerlinNoise(seed=42)
        except ImportError:
            # If terrain module isn't available, we'll create it on demand in get_height
            self.noise = None
    
    def _crystal_shape(self, world_x, world_z, center_x, center_z, size, noise_func, crystal_id):
        """Generate a crystal formation shape."""
        # Distance from crystal center
        dx = world_x - center_x
        dz = world_z - center_z
        dist = math.sqrt(dx**2 + dz**2)
        
        # Crystal seed controls its unique shape
        seed_offset = crystal_id * 100
        
        # Generate parameters for this crystal
        crystal_noise = noise_func.fractal(
            world_x * 0.05 + seed_offset, 
            world_z * 0.05 + seed_offset,
            octaves=3,
            persistence=0.5
        )
        
        # Crystal rotation angle
        rotation = crystal_noise * math.pi * 2.0
        
        # Rotate the point around the crystal center
        rotated_x = dx * math.cos(rotation) - dz * math.sin(rotation)
        rotated_z = dx * math.sin(rotation) + dz * math.cos(rotation)
        
        # Convert to polar coordinates for easier shape manipulation
        angle = math.atan2(rotated_z, rotated_x)
        
        # Make crystals pointier at the top
        if dist < size:
            # Use a power function to create a pointed shape
            # Higher sharpness = more pointed
            pointed_factor = 1.0 - (dist / size) ** self.crystal_sharpness
            
            # Add angle dependence for crystal faces
            faces = 3 + int(crystal_noise * 5)  # 3 to 7 faces
            face_factor = abs(math.sin(angle * faces / 2.0)) * 0.3 + 0.7
            
            # Combine factors
            height_factor = pointed_factor * face_factor
            
            # Add some noise to the crystal surface
            surface_noise = noise_func.fractal(
                world_x * 0.1 + seed_offset + 1000, 
                world_z * 0.1 + seed_offset + 1000,
                octaves=2
            ) * 0.1
            
            return height_factor * size + surface_noise * size
        
        return 0.0
    
    def _get_crystal_color(self, crystal_id, detail_noise):
        """Determine crystal color based on its ID."""
        # Use crystal ID to determine its type
        crystal_type = crystal_id % 6
        
        if crystal_type == 0:
            return self.colors['amethyst']
        elif crystal_type == 1:
            return self.colors['emerald']
        elif crystal_type == 2:
            return self.colors['ruby']
        elif crystal_type == 3:
            return self.colors['sapphire']
        elif crystal_type == 4:
            return self.colors['quartz']
        else:
            return self.colors['amber']
    
    def get_height(self, world_x, world_z):
        """Generate crystal caves terrain height."""
        # Get a noise generator if we don't already have one
        if self.noise is None:
            try:
                from terrain import PerlinNoise
                self.noise = PerlinNoise(seed=42)
            except ImportError:
                # If terrain module isn't available, use a default height
                print("Warning: PerlinNoise not available for CrystalCavesBiome")
                return self.cave_ground_level + 10.0
        
        # Base cave floor with undulations
        base_height = self.cave_ground_level + self.noise.fractal(
            world_x * self.noise_scale, 
            world_z * self.noise_scale,
            octaves=4,
            persistence=0.5
        ) * 10.0
        
        # Add medium-scale details for rock formations
        medium_detail = self.noise.fractal(
            world_x * self.noise_scale * 3, 
            world_z * self.noise_scale * 3,
            octaves=3,
            persistence=0.4
        ) * 5.0
        
        # Ceiling height variation
        ceiling_noise = self.noise.fractal(
            world_x * self.noise_scale * 0.2, 
            world_z * self.noise_scale * 0.2,
            octaves=2
        )
        ceiling_height = self.ceiling_height * (0.8 + ceiling_noise * 0.2)
        
        # Generate crystal field distribution
        crystal_field = self.noise.fractal(
            world_x * self.noise_scale * 0.3, 
            world_z * self.noise_scale * 0.3,
            octaves=2
        )
        
        # Crystal height contribution
        crystal_height = 0.0
        
        # Check if we should generate a crystal here
        if crystal_field > (1.0 - self.crystal_density):
            # Find the nearest crystal center point on a grid
            cell_size = self.max_crystal_size * 1.5
            cell_x = math.floor(world_x / cell_size)
            cell_z = math.floor(world_z / cell_size)
            
            # Check nearby cells for potential crystals
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Jitter the crystal center position for natural placement
                    center_x = (cell_x + i) * cell_size + self.noise.noise2d(cell_x + i, cell_z + j) * cell_size * 0.3
                    center_z = (cell_z + j) * cell_size + self.noise.noise2d(cell_x + i + 50, cell_z + j + 50) * cell_size * 0.3
                    
                    # Distance to crystal center
                    dist = math.sqrt((world_x - center_x)**2 + (world_z - center_z)**2)
                    
                    # Generate unique ID for this crystal
                    current_crystal_id = int(abs(center_x * 1000 + center_z))
                    
                    # Use ID to determine if this point has a crystal
                    crystal_chance = self.noise.noise2d(current_crystal_id * 0.01, 0)
                    
                    if crystal_chance > 0.2:  # 80% of potential points have crystals
                        # Determine crystal size based on its id
                        size_factor = abs(self.noise.noise2d(current_crystal_id * 0.01, 100))
                        crystal_size = self.min_crystal_size + size_factor * (self.max_crystal_size - self.min_crystal_size)
                        
                        # Calculate crystal shape contribution to height
                        if dist < crystal_size * 1.2:  # Check if within crystal influence
                            # Calculate the shape
                            shape_height = self._crystal_shape(
                                world_x, world_z, 
                                center_x, center_z, 
                                crystal_size, 
                                self.noise,
                                current_crystal_id
                            )
                            
                            # Add crystal height contribution
                            crystal_height = max(crystal_height, shape_height)
        
        # Combine all height elements
        final_height = base_height + medium_detail + crystal_height
        
        # Make sure we respect the ceiling height
        final_height = min(final_height, ceiling_height)
        
        return final_height
    
    def get_color(self, world_x, world_z, height):
        """Get color for the crystal caves terrain."""
        # Get a noise generator if we don't already have one
        if self.noise is None:
            try:
                from terrain import PerlinNoise
                self.noise = PerlinNoise(seed=42)
            except ImportError:
                # If terrain module isn't available, return default color
                return self.colors['cave_wall']
        
        # Calculate base cave floor height (without crystals)
        base_height = self.cave_ground_level + self.noise.fractal(
            world_x * self.noise_scale, 
            world_z * self.noise_scale,
            octaves=4,
            persistence=0.5
        ) * 10.0
        
        # Detail noise for color variation
        detail_noise = self.noise.fractal(
            world_x * self.noise_scale * 5, 
            world_z * self.noise_scale * 5,
            octaves=3
        ) * 0.3
        
        # Generate crystal field to determine if this is a crystal
        crystal_field = self.noise.fractal(
            world_x * self.noise_scale * 0.3, 
            world_z * self.noise_scale * 0.3,
            octaves=2
        )
        
        # Is this part of a crystal?
        if height > base_height + 5.0 and crystal_field > (1.0 - self.crystal_density):
            # Find the nearest crystal center for color determination
            cell_size = self.max_crystal_size * 1.5
            cell_x = math.floor(world_x / cell_size)
            cell_z = math.floor(world_z / cell_size)
            
            # Find the closest crystal center point
            closest_dist = float('inf')
            closest_crystal_id = 0
            
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Crystal center position
                    center_x = (cell_x + i) * cell_size + self.noise.noise2d(cell_x + i, cell_z + j) * cell_size * 0.3
                    center_z = (cell_z + j) * cell_size + self.noise.noise2d(cell_x + i + 50, cell_z + j + 50) * cell_size * 0.3
                    
                    # Distance to crystal center
                    dist = math.sqrt((world_x - center_x)**2 + (world_z - center_z)**2)
                    
                    # Generate unique ID for this crystal
                    current_crystal_id = int(abs(center_x * 1000 + center_z))
                    
                    # Use ID to determine if this point has a crystal
                    crystal_chance = self.noise.noise2d(current_crystal_id * 0.01, 0)
                    
                    if crystal_chance > 0.2 and dist < closest_dist:  # It's a crystal and closer
                        closest_dist = dist
                        closest_crystal_id = current_crystal_id
            
            # Get crystal color
            if closest_crystal_id > 0:  # Found a valid crystal
                base_color = self._get_crystal_color(closest_crystal_id, detail_noise)
                
                # Apply detail noise for crystal variation
                r = min(1.0, max(0.0, base_color[0] + detail_noise * 0.2))
                g = min(1.0, max(0.0, base_color[1] + detail_noise * 0.2))
                b = min(1.0, max(0.0, base_color[2] + detail_noise * 0.2))
                
                return (r, g, b)
        
        # For cave floor, add water pools in the lower areas
        if height < base_height + 1.0:
            # Water pool probability increases in deeper areas
            water_chance = 1.0 - (height - (self.cave_ground_level - 5.0)) / 10.0
            water_noise = self.noise.fractal(
                world_x * self.noise_scale * 2, 
                world_z * self.noise_scale * 2,
                octaves=2
            )
            
            if water_noise > (1.0 - water_chance * 0.3):
                # Water color
                water_color = self.colors['water']
                
                # Add glow effect based on noise
                glow = self.noise.fractal(
                    world_x * self.noise_scale * 10, 
                    world_z * self.noise_scale * 10,
                    octaves=2
                ) * 0.3
                
                r = min(1.0, water_color[0] + glow)
                g = min(1.0, water_color[1] + glow)
                b = min(1.0, water_color[2] + glow)
                
                return (r, g, b)
        
        # Cave wall color with variation
        wall_color = self.colors['cave_wall']
        
        # Add some variation
        r = min(1.0, max(0.0, wall_color[0] + detail_noise))
        g = min(1.0, max(0.0, wall_color[1] + detail_noise))
        b = min(1.0, max(0.0, wall_color[2] + detail_noise))
        
        return (r, g, b)