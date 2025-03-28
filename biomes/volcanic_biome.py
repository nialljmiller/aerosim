"""
Volcanic Biome for Flight Simulator
Creates a dramatic landscape with active volcanoes, lava flows, and smoke plumes.
"""

import numpy as np
import math
import sys
import os

# Ensure biomes directory is in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biomes.biome_base import BaseBiome

class VolcanicBiome(BaseBiome):
    """Active volcanic landscape with lava flows and ash fields."""
    
    def __init__(self):
        super().__init__()
        self.name = "volcanic"
        
        # Volcanic specific parameters
        self.height_scale = 70.0  # For dramatic volcanoes
        self.noise_scale = 0.008
        
        # Volcano parameters
        self.volcano_density = 0.4  # Controls volcano distribution
        self.min_volcano_radius = 50.0
        self.max_volcano_radius = 150.0
        self.max_volcano_height = 80.0
        self.crater_depth_factor = 0.3  # How deep craters are relative to volcano height
        
        # Lava and terrain features
        self.lava_coverage = 0.6  # How much lava appears
        
        # Colors
        self.colors = {
            'basalt': (0.2, 0.2, 0.2),       # Black volcanic rock
            'lava': (0.9, 0.3, 0.05),        # Bright orange lava
            'obsidian': (0.1, 0.1, 0.12),    # Very dark glassy rock
            'ash': (0.3, 0.3, 0.3),          # Gray ash
            'scorched': (0.4, 0.2, 0.1),     # Burnt ground
            'hot_rock': (0.5, 0.3, 0.2),     # Heat-affected rock
            'sulfur': (0.9, 0.8, 0.2)        # Yellow sulfur deposits
        }
    
    def _volcano_shape(self, dist_from_center, radius, height, crater_radius, crater_depth):
        """Generate the shape of a volcano with crater."""
        # Basic volcano cone shape
        if dist_from_center > radius:
            return 0.0  # Outside volcano radius
        
        # Calculate height based on distance from center (cone function)
        height_factor = 1.0 - (dist_from_center / radius)
        cone_height = height * height_factor
        
        # Apply crater depression if near the center
        if dist_from_center < crater_radius:
            # Parabolic crater shape
            crater_factor = 1.0 - (dist_from_center / crater_radius)**2
            crater_height = -crater_depth * crater_factor
            return cone_height + crater_height
        
        return cone_height
    
    def _generate_lava_flow(self, world_x, world_z, volcano_x, volcano_z, noise_func, flow_seed):
        """Generate lava flow pattern from a volcano."""
        # Direction of flow (from crater outward)
        dx = world_x - volcano_x
        dz = world_z - volcano_z
        
        dist = math.sqrt(dx**2 + dz**2)
        if dist < 1.0:
            dist = 1.0
            
        # Angle from volcano center
        angle = math.atan2(dz, dx)
        
        # Generate noise to determine flow directions
        flow_noise = noise_func.fractal(
            volcano_x * 0.01 + flow_seed, 
            volcano_z * 0.01 + flow_seed,
            octaves=2
        )
        
        # Create preferred flow direction
        preferred_angle = flow_noise * math.pi * 2.0
        
        # Calculate how aligned this point is with the preferred flow direction
        angle_diff = abs(((angle - preferred_angle + math.pi) % (2 * math.pi)) - math.pi)
        flow_alignment = 1.0 - min(1.0, angle_diff / (math.pi * 0.5))
        
        # Flow strength decreases with distance
        flow_strength = max(0.0, 1.0 - (dist / (self.max_volcano_radius * 0.7)))
        
        # Combine alignment and distance
        return flow_alignment * flow_strength
    
    def get_height(self, world_x, world_z):
        # Safe input handling
        try:
            world_x = float(world_x)
            world_z = float(world_z)
        except (TypeError, ValueError):
            print(f"Error in {self.name} biome: invalid coordinates")
            return 0.0
            
        try:
            """Generate volcanic terrain height."""
            from terrain import PerlinNoise
            noise = PerlinNoise(seed=42)
            
            # Base terrain with ash fields and rough terrain
            base_height = noise.fractal(
                world_x * self.noise_scale, 
                world_z * self.noise_scale,
                octaves=6,
                persistence=0.5
            ) * self.height_scale * 0.5
            
            # Medium detail for lava fields and smaller features
            medium_detail = noise.fractal(
                world_x * self.noise_scale * 4, 
                world_z * self.noise_scale * 4,
                octaves=3,
                persistence=0.4
            ) * 5.0
            
            # Generate volcano field distribution
            volcano_field = noise.fractal(
                world_x * self.noise_scale * 0.1, 
                world_z * self.noise_scale * 0.1,
                octaves=2,
                persistence=0.5
            )
            
            # Volcanoes appear where volcano_field exceeds threshold
            volcano_elevation = 0.0
            lava_flow_factor = 0.0
            
            # Check for volcano influence
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Create a grid of potential volcano centers
                    cell_size = self.max_volcano_radius * 2.0
                    cell_x = math.floor(world_x / cell_size)
                    cell_z = math.floor(world_z / cell_size)
                    
                    center_x = (cell_x + i) * cell_size + noise.noise2d(cell_x + i, cell_z + j) * cell_size * 0.3
                    center_z = (cell_z + j) * cell_size + noise.noise2d(cell_x + i + 50, cell_z + j + 50) * cell_size * 0.3
                    
                    # Distance to volcano center
                    dx = world_x - center_x
                    dz = world_z - center_z
                    dist = math.sqrt(dx**2 + dz**2)
                    
                    # Generate a unique id for this potential volcano
                    volcano_id = int(abs(center_x * 1000 + center_z))
                    
                    # Determine volcano parameters based on its id
                    size_noise = abs(noise.noise2d(volcano_id * 0.01, 0))
                    activity_noise = abs(noise.noise2d(volcano_id * 0.01, 100))
                    
                    # Size parameters
                    volcano_radius = self.min_volcano_radius + size_noise * (self.max_volcano_radius - self.min_volcano_radius)
                    volcano_height = self.max_volcano_height * (0.5 + size_noise * 0.5)
                    crater_radius = volcano_radius * 0.2
                    crater_depth = volcano_height * self.crater_depth_factor
                    
                    # Only create a volcano if the field value exceeds threshold
                    if volcano_field > (1.0 - self.volcano_density):
                        # Calculate volcano shape at this point
                        v_elevation = self._volcano_shape(dist, volcano_radius, volcano_height, crater_radius, crater_depth)
                        volcano_elevation = max(volcano_elevation, v_elevation)
                        
                        # Calculate lava flow from this volcano
                        if activity_noise > 0.4:  # Only active volcanoes have lava flows
                            flow = self._generate_lava_flow(world_x, world_z, center_x, center_z, noise, volcano_id)
                            lava_flow_factor = max(lava_flow_factor, flow)
            
            # Final terrain height with base, volcanos, and lava flows
            final_height = base_height + volcano_elevation
            
            # Lava fields create slightly raised areas
            lava_field_noise = noise.fractal(
                world_x * self.noise_scale * 2 + 1000, 
                world_z * self.noise_scale * 2 + 1000,
                octaves=2
            )
            
            if lava_field_noise < self.lava_coverage:
                # Add small bumps for cooling lava
                lava_bump = lava_field_noise * 3.0
                final_height += lava_bump
            
            # Apply lava flow influence
            if lava_flow_factor > 0.0:
                lava_crust = medium_detail * lava_flow_factor
                final_height += lava_crust
                
            # Ensure minimum terrain height
            return max(0.5, final_height)
        

        except Exception as e:
            print(f"Error in {self.name} biome get_height: {e}")
            return 0.0  # Safe default
            
    def get_color(self, world_x, world_z, height):
        """Get volcanic terrain colors."""
        from terrain import PerlinNoise
        noise = PerlinNoise(seed=42)
        
        # Generate noise for lava and terrain features
        lava_noise = noise.fractal(
            world_x * self.noise_scale * 2 + 1000, 
            world_z * self.noise_scale * 2 + 1000,
            octaves=2
        )
        
        detail_noise = noise.fractal(
            world_x * self.noise_scale * 8, 
            world_z * self.noise_scale * 8,
            octaves=4,
            persistence=0.5
        )
        
        # Generate active volcano field
        volcano_activity = noise.fractal(
            world_x * self.noise_scale * 0.1 + 500, 
            world_z * self.noise_scale * 0.1 + 500,
            octaves=1
        )
        
        # Calculate gradient for crater and steep slope detection
        epsilon = 0.5
        h_dx = self.get_height(world_x + epsilon, world_z)
        h_dz = self.get_height(world_x, world_z + epsilon)
        gradient_mag = math.sqrt((h_dx - height)**2 + (h_dz - height)**2)
        
        # Determine if this point is likely a crater or steep volcano side
        is_steep_slope = gradient_mag > 0.7
        is_very_steep = gradient_mag > 1.5
        
        # Determine color based on features
        if lava_noise < 0.2 and height > 5.0:
            # Lava flows and pools
            if detail_noise < 0.3:
                # Bright glowing lava
                return self.colors['lava']
            else:
                # Cooling lava with darker crust
                return self.interpolate_color(self.colors['lava'], self.colors['hot_rock'], detail_noise)
        elif is_very_steep and volcano_activity > 0.6:
            # Steep volcano sides, possibly with exposed rock
            if detail_noise < 0.4:
                # Dark volcanic rock
                return self.colors['obsidian']
            else:
                return self.colors['basalt']
        elif is_steep_slope:
            # Less steep slopes with ash
            return self.colors['ash']
        elif height > 40.0 and detail_noise < 0.3:
            # Sulfur deposits on higher elevations
            return self.colors['sulfur']
        else:
            # Default scorched ground
            return self.interpolate_color(self.colors['basalt'], self.colors['scorched'], detail_noise)