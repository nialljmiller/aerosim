
import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class PerlinNoise:
    """Deterministic Perlin noise implementation using NumPy."""
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Create permutation table for consistent noise
        self.perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.perm)
        self.perm = np.concatenate([self.perm, self.perm]).astype(np.int32)
        
    def _fade(self, t):
        """Quintic interpolation curve for smooth transitions."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a, b, t):
        """Linear interpolation between values."""
        return a * (1 - t) + b * t
    
    def _grad(self, hash_val, x, y, z=0.0):
        """Gradient vector selection based on hash value."""
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        
    def noise2d(self, x, y):
        """Generate 2D noise at coordinates (x, y)."""
        # Integer coordinates
        X = np.floor(x).astype(np.int32) & 255
        Y = np.floor(y).astype(np.int32) & 255
        
        # Fractional coordinates
        x -= np.floor(x)
        y -= np.floor(y)
        
        # Compute fade curves
        u = self._fade(x)
        v = self._fade(y)
        
        # Hash coordinates
        A = self.perm[X] + Y
        AA = self.perm[A]
        AB = self.perm[A + 1]
        B = self.perm[X + 1] + Y
        BA = self.perm[B]
        BB = self.perm[B + 1]
        
        # Gradient values
        g1 = self._grad(self.perm[AA], x, y)
        g2 = self._grad(self.perm[BA], x - 1, y)
        g3 = self._grad(self.perm[AB], x, y - 1)
        g4 = self._grad(self.perm[BB], x - 1, y - 1)
        
        # Bilinear interpolation
        x1 = self._lerp(g1, g2, u)
        x2 = self._lerp(g3, g4, u)
        return self._lerp(x1, x2, v)
    
    def fractal(self, x, y, octaves=6, persistence=0.5, lacunarity=2.0):
        """Generate fractal noise by summing multiple octaves."""
        total = 0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        # Sum multiple noise octaves
        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
            
        # Normalize the result
        return total / max_value

class TerrainChunk:
    """Individual chunk of procedurally generated terrain."""
    def __init__(self, chunk_x, chunk_z, chunk_size, resolution, noise_generator):
        self.chunk_x = chunk_x  # Chunk position in world (grid coordinates)
        self.chunk_z = chunk_z
        self.chunk_size = chunk_size  # Size of each chunk in world units
        self.resolution = resolution  # Resolution within chunk (vertex spacing)
        
        # World space coordinates of this chunk
        self.world_x = chunk_x * chunk_size
        self.world_z = chunk_z * chunk_size
        
        # Reference to the shared noise generator
        self.noise = noise_generator
        
        # Display list IDs
        self.display_list_solid = None
        self.display_list_wireframe = None
        
        # Generate heightmap and colors
        self.generate_data()
        self.compile_display_lists()
        



    def generate_data(self):
        """Generate heightmap and color data for this chunk with enhanced variety."""
        # Calculate grid size within this chunk
        grid_size = int(self.chunk_size / self.resolution) + 1
        
        # Storage for height and color data
        self.heightmap = np.zeros((grid_size, grid_size))
        self.color_data = np.zeros((grid_size, grid_size, 3))
        
        # Enhanced settings for more dramatic terrain features
        noise_scale = 0.005  # Controls terrain scale/frequency
        height_scale = 80.0  # Increased from 40.0 for more dramatic heights
        
        # Generate height values for each vertex
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate world coordinates
                world_x = self.world_x + i * self.resolution
                world_z = self.world_z + j * self.resolution
                
                # Base terrain using fractal noise
                base_height = self.noise.fractal(
                    world_x * noise_scale, 
                    world_z * noise_scale,
                    octaves=6,
                    persistence=0.6,  # Increased from 0.5 for more variation
                    lacunarity=2.2    # Increased from 2.0 for more detailed features
                )
                
                # Add medium-scale variation with stronger effect
                medium_detail = self.noise.fractal(
                    world_x * noise_scale * 4, 
                    world_z * noise_scale * 4,
                    octaves=4,       # Increased from 3
                    persistence=0.4  # Increased from 0.3
                ) * 0.35            # Increased from 0.25
                
                # Add small-scale variation with stronger effect
                small_detail = self.noise.fractal(
                    world_x * noise_scale * 16, 
                    world_z * noise_scale * 16,
                    octaves=3,       # Increased from 2
                    persistence=0.2  # Increased from 0.15
                ) * 0.1             # Increased from 0.05
                
                # Enhanced river and canyon carving system
                # Use multiple noise layers for more complex river networks
                river_noise_primary = self.noise.fractal(
                    world_x * noise_scale * 2 + 500, 
                    world_z * noise_scale * 2 + 500,
                    octaves=1
                )
                
                river_noise_secondary = self.noise.fractal(
                    world_x * noise_scale * 1.5 + 1000, 
                    world_z * noise_scale * 1.5 + 1000,
                    octaves=1
                )
                
                # River and canyon carving with variable depths
                river_factor = 0
                canyon_factor = 0
                
                # Primary river system
                river_range = 0.05
                if 0.48 < river_noise_primary < 0.48 + river_range:
                    # Smooth transition from regular terrain to river
                    river_depth = 15.0  # Increased from 10.0
                    t = (river_noise_primary - 0.48) / river_range
                    river_factor = (1.0 - 4 * (t - 0.5) * (t - 0.5)) * river_depth
                
                # Secondary river/canyon system
                canyon_range = 0.04
                if 0.46 < river_noise_secondary < 0.46 + canyon_range:
                    # Create deeper canyons with steeper walls
                    canyon_depth = 25.0
                    t = (river_noise_secondary - 0.46) / canyon_range
                    canyon_shape = 1.0 - 8 * (t - 0.5) * (t - 0.5) * (t - 0.5) * (t - 0.5)
                    canyon_factor = canyon_shape * canyon_depth
                
                # Enhanced mountain generation with multiple mountain ranges
                # Mountain range 1 - Higher jagged mountains
                mountain_mask1 = self.noise.fractal(
                    world_x * noise_scale * 0.125, 
                    world_z * noise_scale * 0.125,
                    octaves=2
                )
                
                # Mountain range 2 - Smoother, rolling mountains
                mountain_mask2 = self.noise.fractal(
                    world_x * noise_scale * 0.1 + 3000, 
                    world_z * noise_scale * 0.1 + 3000,
                    octaves=3,
                    persistence=0.4
                )
                
                # Apply domain warping for more natural-looking mountain shapes
                mountain_warp_x = self.noise.fractal(
                    world_x * noise_scale * 0.5 + 1000, 
                    world_z * noise_scale * 0.5 + 1000,
                    octaves=3        # Increased from 2
                ) * 150             # Increased from 100
                
                mountain_warp_z = self.noise.fractal(
                    world_x * noise_scale * 0.5 + 2000, 
                    world_z * noise_scale * 0.5 + 2000,
                    octaves=3        # Increased from 2
                ) * 150             # Increased from 100
                
                # Compute mountain height using warped coordinates for more natural formations
                mountain_height1 = self.noise.fractal(
                    (world_x + mountain_warp_x) * noise_scale * 0.25, 
                    (world_z + mountain_warp_z) * noise_scale * 0.25,
                    octaves=5,       # Increased from 4
                    persistence=0.65 # Increased from 0.6
                )
                
                mountain_height2 = self.noise.fractal(
                    (world_x - mountain_warp_z * 0.5) * noise_scale * 0.2, 
                    (world_z + mountain_warp_x * 0.5) * noise_scale * 0.2,
                    octaves=4,
                    persistence=0.5
                )
                
                # Apply mountain heights with more dramatic peaks
                mountain_factor1 = 0
                mountain_factor2 = 0
                
                # Jagged mountain range
                if mountain_mask1 > 0.55:
                    mountain_factor1 = (mountain_mask1 - 0.55) * 2.5 * 90 * mountain_height1  # Increased height multiplier
                
                # Smoother mountain range
                if mountain_mask2 > 0.6:
                    mountain_factor2 = (mountain_mask2 - 0.6) * 2.0 * 60 * mountain_height2
                
                # Add plateaus in some regions for interesting landing spots
                plateau_noise = self.noise.fractal(
                    world_x * noise_scale * 0.05 + 5000, 
                    world_z * noise_scale * 0.05 + 5000,
                    octaves=1
                )
                
                plateau_factor = 0
                if plateau_noise > 0.7:
                    plateau_height = 30 + 20 * self.noise.fractal(
                        world_x * noise_scale * 0.5, 
                        world_z * noise_scale * 0.5,
                        octaves=1
                    )
                    plateau_blend = min(1.0, (plateau_noise - 0.7) * 10)
                    plateau_factor = plateau_height * plateau_blend
                
                # Combine all height components
                height = (base_height + medium_detail + small_detail) * height_scale
                
                # Apply all terrain features
                height = height - river_factor - canyon_factor + mountain_factor1 + mountain_factor2
                
                # Apply plateau as a blended max operation to create flat tops
                if plateau_factor > 0:
                    blend = (plateau_noise - 0.7) * 10
                    height = height * (1 - blend) + max(height, plateau_factor) * blend
                
                # Ensure minimum terrain height to prevent gaps
                height = max(0.5, height)
                
                # Store height
                self.heightmap[i, j] = height
                
                # Compute color based on height and features
                self.assign_color(i, j, height, river_factor > 0 or canyon_factor > 0)

                
    def assign_color(self, i, j, height, is_river):
        """Assign colors based on terrain features."""
        # Color parameters
        water_level = 2.0
        sand_level = 4.0
        grass_level = 15.0
        rock_level = 30.0
        snow_level = 50.0
        
        # Define colors (RGB)
        water_color = (0.0, 0.3, 0.8)
        sand_color = (0.8, 0.7, 0.4)
        grass_color = (0.3, 0.6, 0.2)
        rock_color = (0.5, 0.5, 0.5)
        snow_color = (0.9, 0.9, 0.95)
        river_color = (0.0, 0.4, 0.7)
        
        # Determine base color by height
        if is_river:
            color = river_color
        elif height <= water_level:
            color = water_color
        elif height <= sand_level:
            t = (height - water_level) / (sand_level - water_level)
            color = self.interpolate_color(water_color, sand_color, t)
        elif height <= grass_level:
            t = (height - sand_level) / (grass_level - sand_level)
            color = self.interpolate_color(sand_color, grass_color, t)
        elif height <= rock_level:
            t = (height - grass_level) / (rock_level - grass_level)
            color = self.interpolate_color(grass_color, rock_color, t)
        else:
            t = min(1.0, (height - rock_level) / (snow_level - rock_level))
            color = self.interpolate_color(rock_color, snow_color, t)
        
        # Add some noise to the colors for variation
        world_x = self.world_x + i * self.resolution
        world_z = self.world_z + j * self.resolution
        noise_value = (np.sin(world_x * 0.05) * np.sin(world_z * 0.05)) * 0.1
        color = (
            max(0, min(1, color[0] + noise_value)),
            max(0, min(1, color[1] + noise_value)),
            max(0, min(1, color[2] + noise_value))
        )
        
        # Store the color
        self.color_data[i, j] = color
    
    def interpolate_color(self, color1, color2, t):
        """Linearly interpolate between two colors."""
        return (
            color1[0] * (1-t) + color2[0] * t,
            color1[1] * (1-t) + color2[1] * t,
            color1[2] * (1-t) + color2[2] * t
        )
    
    def get_height(self, world_x, world_z):
        """Get interpolated height at any point within this chunk."""
        # Convert world coordinates to local chunk coordinates
        local_x = world_x - self.world_x
        local_z = world_z - self.world_z
        
        # Convert to grid coordinates
        grid_x = local_x / self.resolution
        grid_z = local_z / self.resolution
        
        # Grid indices
        i = int(grid_x)
        j = int(grid_z)
        
        # Check bounds (should be within chunk)
        grid_size = int(self.chunk_size / self.resolution)
        if i < 0 or i >= grid_size or j < 0 or j >= grid_size:
            return 0  # Default height for out-of-bounds
        
        # Get fractional parts for interpolation
        fx = grid_x - i
        fz = grid_z - j
        
        # Bilinear interpolation of height values
        h1 = self.heightmap[i, j]
        h2 = self.heightmap[min(i+1, grid_size), j]
        h3 = self.heightmap[i, min(j+1, grid_size)]
        h4 = self.heightmap[min(i+1, grid_size), min(j+1, grid_size)]
        
        # Interpolate along x
        h12 = h1 * (1-fx) + h2 * fx
        h34 = h3 * (1-fx) + h4 * fx
        
        # Interpolate result along z
        height = h12 * (1-fz) + h34 * fz
        
        return height
    
    def get_normal(self, world_x, world_z):
        """Calculate terrain normal at world coordinate point using central differences."""
        epsilon = 0.1  # Small offset for normal calculation
        
        # Get heights at nearby points
        h_center = self.get_height(world_x, world_z)
        h_dx = self.get_height(world_x + epsilon, world_z)
        h_dz = self.get_height(world_x, world_z + epsilon)
        
        # Calculate partial derivatives
        dx = (h_dx - h_center) / epsilon
        dz = (h_dz - h_center) / epsilon
        
        # Cross product to get normal vector
        normal = np.array([-dx, 1.0, -dz])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        return normal
    
    def compile_display_lists(self):
        """Compile terrain mesh into OpenGL display lists for efficient rendering."""
        # Create display lists
        self.display_list_solid = glGenLists(1)
        self.display_list_wireframe = glGenLists(1)
        
        grid_size = self.heightmap.shape[0]
        
        # Compile solid terrain rendering
        glNewList(self.display_list_solid, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Calculate actual vertex positions
                x1 = self.world_x + i * self.resolution
                z1 = self.world_z + j * self.resolution
                y1 = self.heightmap[i, j]
                
                x2 = self.world_x + (i+1) * self.resolution
                z2 = self.world_z + j * self.resolution
                y2 = self.heightmap[i+1, j]
                
                x3 = self.world_x + i * self.resolution
                z3 = self.world_z + (j+1) * self.resolution
                y3 = self.heightmap[i, j+1]
                
                x4 = self.world_x + (i+1) * self.resolution
                z4 = self.world_z + (j+1) * self.resolution
                y4 = self.heightmap[i+1, j+1]
                
                # Get vertex colors
                c1 = self.color_data[i, j]
                c2 = self.color_data[i+1, j]
                c3 = self.color_data[i, j+1]
                c4 = self.color_data[i+1, j+1]
                
                # Calculate face normals for lighting
                v1 = np.array([x1, y1, z1])
                v2 = np.array([x2, y2, z2])
                v3 = np.array([x3, y3, z3])
                v4 = np.array([x4, y4, z4])
                
                # Vectors for cross product
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal1 = np.cross(edge1, edge2)
                normal1 = normal1 / np.linalg.norm(normal1)
                
                edge1 = v2 - v4
                edge2 = v3 - v4
                normal2 = np.cross(edge2, edge1)
                normal2 = normal2 / np.linalg.norm(normal2)
                
                # Draw first triangle
                glNormal3f(normal1[0], normal1[1], normal1[2])
                glColor3f(*c1)
                glVertex3f(x1, y1, z1)
                glColor3f(*c2)
                glVertex3f(x2, y2, z2)
                glColor3f(*c3)
                glVertex3f(x3, y3, z3)
                
                # Draw second triangle
                glNormal3f(normal2[0], normal2[1], normal2[2])
                glColor3f(*c2)
                glVertex3f(x2, y2, z2)
                glColor3f(*c4)
                glVertex3f(x4, y4, z4)
                glColor3f(*c3)
                glVertex3f(x3, y3, z3)
        
        glEnd()
        glEndList()
        
        # Compile wireframe terrain rendering
        glNewList(self.display_list_wireframe, GL_COMPILE)
        glBegin(GL_LINES)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Calculate vertex positions
                x1 = self.world_x + i * self.resolution
                z1 = self.world_z + j * self.resolution
                y1 = self.heightmap[i, j]
                
                x2 = self.world_x + (i+1) * self.resolution
                z2 = self.world_z + j * self.resolution
                y2 = self.heightmap[i+1, j]
                
                x3 = self.world_x + i * self.resolution
                z3 = self.world_z + (j+1) * self.resolution
                y3 = self.heightmap[i, j+1]
                
                # Draw grid lines
                glColor3f(0.0, 0.0, 0.0)  # Black lines
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
                
                glVertex3f(x1, y1, z1)
                glVertex3f(x3, y3, z3)
        
        glEnd()
        glEndList()
    
    def draw(self, wireframe=False):
        """Draw the terrain chunk."""
        if wireframe:
            # Draw wireframe
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glCallList(self.display_list_wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            # Draw solid terrain
            glCallList(self.display_list_solid)
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.display_list_solid:
            glDeleteLists(self.display_list_solid, 1)
        if self.display_list_wireframe:
            glDeleteLists(self.display_list_wireframe, 1)

class InfiniteTerrain:
    """Manager for dynamically generated infinite terrain."""
    def __init__(self, chunk_size=100, resolution=5, view_distance=800):
        self.chunk_size = chunk_size  # Size of each chunk in world units
        self.resolution = resolution  # Distance between vertices
        self.view_distance = view_distance  # How far to render chunks
        
        # Seed for consistent terrain generation
        self.seed = 42
        self.noise = PerlinNoise(seed=self.seed)
        
        # Dictionary to store loaded chunks (key: (chunk_x, chunk_z))
        self.chunks = {}
        
        # Track the center position for chunk loading/unloading
        self.last_center_chunk = None
    
    def get_chunk_position(self, world_x, world_z):
        """Convert world coordinates to chunk coordinates."""
        chunk_x = math.floor(world_x / self.chunk_size)
        chunk_z = math.floor(world_z / self.chunk_size)
        return chunk_x, chunk_z
    
    def get_or_create_chunk(self, chunk_x, chunk_z):
        """Get an existing chunk or create a new one if it doesn't exist."""
        chunk_key = (chunk_x, chunk_z)
        if chunk_key not in self.chunks:
            # Create new chunk
            self.chunks[chunk_key] = TerrainChunk(
                chunk_x, chunk_z, 
                self.chunk_size, 
                self.resolution,
                self.noise
            )
        return self.chunks[chunk_key]
    
    def update_chunks(self, camera_position):
        """Update which chunks are loaded based on camera position."""
        # Get chunk coordinates for camera position
        center_chunk_x, center_chunk_z = self.get_chunk_position(
            camera_position[0], camera_position[2]
        )
        center_chunk = (center_chunk_x, center_chunk_z)
        
        # If center hasn't changed, no need to update chunks
        if center_chunk == self.last_center_chunk:
            return
        
        self.last_center_chunk = center_chunk
        
        # Calculate view distance in chunks
        chunk_view_distance = math.ceil(self.view_distance / self.chunk_size)
        
        # Determine which chunks should be loaded
        chunks_to_load = set()
        for x in range(center_chunk_x - chunk_view_distance, center_chunk_x + chunk_view_distance + 1):
            for z in range(center_chunk_z - chunk_view_distance, center_chunk_z + chunk_view_distance + 1):
                # Calculate distance from center chunk
                dx = x - center_chunk_x
                dz = z - center_chunk_z
                distance = math.sqrt(dx*dx + dz*dz)
                
                # Only load chunks within view distance
                if distance <= chunk_view_distance:
                    chunks_to_load.add((x, z))
        
        # Unload chunks that are too far away
        chunks_to_unload = set(self.chunks.keys()) - chunks_to_load
        for chunk_key in chunks_to_unload:
            # Clean up OpenGL resources
            self.chunks[chunk_key].cleanup()
            # Remove from dictionary
            del self.chunks[chunk_key]
        
        # Load new chunks
        for chunk_key in chunks_to_load:
            if chunk_key not in self.chunks:
                self.get_or_create_chunk(*chunk_key)
    
    def get_height(self, world_x, world_z):
        """Get terrain height at any world position."""
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)
        
        # If the chunk is loaded, query it
        if chunk_key in self.chunks:
            return self.chunks[chunk_key].get_height(world_x, world_z)
        
        # For unloaded chunks, generate height on-the-fly without creating the full chunk
        # This is for collision detection at large distances
        noise_scale = 0.005
        height_scale = 40.0
        
        # Generate height using the same algorithm as in TerrainChunk.generate_data
        # But simplified to only what's needed for height
        base_height = self.noise.fractal(
            world_x * noise_scale, 
            world_z * noise_scale,
            octaves=6,
            persistence=0.5,
            lacunarity=2.0
        )
        
        # Add medium details
        medium_detail = self.noise.fractal(
            world_x * noise_scale * 4, 
            world_z * noise_scale * 4,
            octaves=3,
            persistence=0.3
        ) * 0.25
        
        # Apply river carving
        river_noise = self.noise.fractal(
            world_x * noise_scale * 2 + 500, 
            world_z * noise_scale * 2 + 500,
            octaves=1
        )
        
        river_factor = 0
        river_range = 0.05
        if 0.48 < river_noise < 0.48 + river_range:
            river_depth = 10.0
            t = (river_noise - 0.48) / river_range
            river_factor = (1.0 - 4 * (t - 0.5) * (t - 0.5)) * river_depth
        
        # Calculate mountain influence
        mountain_mask = self.noise.fractal(
            world_x * noise_scale * 0.125, 
            world_z * noise_scale * 0.125,
            octaves=2
        )
        
        mountain_factor = 0
        if mountain_mask > 0.55:
            mountain_warp_x = self.noise.fractal(
                world_x * noise_scale * 0.5 + 1000, 
                world_z * noise_scale * 0.5 + 1000,
                octaves=2
            ) * 100
            
            mountain_warp_z = self.noise.fractal(
                world_x * noise_scale * 0.5 + 2000, 
                world_z * noise_scale * 0.5 + 2000,
                octaves=2
            ) * 100
            
            mountain_height = self.noise.fractal(
                (world_x + mountain_warp_x) * noise_scale * 0.25, 
                (world_z + mountain_warp_z) * noise_scale * 0.25,
                octaves=4,
                persistence=0.6
            )
            
            mountain_factor = (mountain_mask - 0.55) * 2.0 * 60 * mountain_height
        
        # Combine components
        height = (base_height + medium_detail) * height_scale
        height = height - river_factor + mountain_factor
        
        # Ensure minimum height
        return max(0.5, height)
    
    def get_terrain_normal(self, world_x, world_z):
        """Calculate terrain normal at any world position."""
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)
        
        # If the chunk is loaded, query it for the normal
        if chunk_key in self.chunks:
            return self.chunks[chunk_key].get_normal(world_x, world_z)
        
        # For unloaded chunks, calculate normal on-the-fly
        epsilon = 0.1  # Small offset for normal calculation
        
        # Get heights at nearby points
        h_center = self.get_height(world_x, world_z)
        h_dx = self.get_height(world_x + epsilon, world_z)
        h_dz = self.get_height(world_x, world_z + epsilon)
        
        # Calculate partial derivatives
        dx = (h_dx - h_center) / epsilon
        dz = (h_dz - h_center) / epsilon
        
        # Cross product to get normal vector
        normal = np.array([-dx, 1.0, -dz])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        return normal
    
    def draw(self, wireframe=False):
        """Draw all loaded terrain chunks."""
        for chunk in self.chunks.values():
            chunk.draw(wireframe)
    
    def cleanup(self):
        """Free all OpenGL resources."""
        for chunk in self.chunks.values():
            chunk.cleanup()
        self.chunks.clear()






class Terrain_class:
    """Class to manage procedural terrain generation and rendering."""
    def __init__(self, size=400, resolution=10, scale=1.0, octaves=6, persistence=0.5, lacunarity=2.0):
        self.size = size            # Total size of terrain
        self.resolution = resolution # Resolution of the grid
        self.scale = scale          # Vertical scale of terrain
        
        # Noise parameters
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = random.randint(0, 1000)
        
        # Create noise generator
        self.noise = PerlinNoise(seed=self.seed)
        
        # Generate terrain data
        self.generate_heightmap()
        self.generate_textures()
        
        # Compile display lists for efficient rendering
        self.compile_display_lists()
    
    def get_height(self, x, z):
        """Get the interpolated height at any point (x,z) on the terrain."""
        # Convert world coordinates to grid coordinates
        grid_x = (x + self.size/2) / self.resolution
        grid_z = (z + self.size/2) / self.resolution
        
        # Get grid cell indices
        i = int(grid_x)
        j = int(grid_z)
        
        # Handle out-of-bounds coordinates
        if i < 0 or i >= len(self.heightmap)-1 or j < 0 or j >= len(self.heightmap[0])-1:
            # Return ground level for out-of-bounds points
            return 0.0
        
        # Get fractional parts for interpolation
        fx = grid_x - i
        fz = grid_z - j
        
        # Bilinear interpolation of height values
        h1 = self.heightmap[i][j]
        h2 = self.heightmap[i+1][j]
        h3 = self.heightmap[i][j+1]
        h4 = self.heightmap[i+1][j+1]
        
        # First interpolate along x
        h12 = h1 * (1-fx) + h2 * fx
        h34 = h3 * (1-fx) + h4 * fx
        
        # Then interpolate result along z
        height = h12 * (1-fz) + h34 * fz
        
        return height
    
    def get_terrain_normal(self, x, z):
        """Calculate terrain normal at point (x,z) using central differences."""
        epsilon = 0.1  # Small offset for normal calculation
        
        # Get heights at nearby points
        h_center = self.get_height(x, z)
        h_dx = self.get_height(x + epsilon, z)
        h_dz = self.get_height(x, z + epsilon)
        
        # Calculate partial derivatives
        dx = (h_dx - h_center) / epsilon
        dz = (h_dz - h_center) / epsilon
        
        # Cross product to get normal vector
        normal = np.array([-dx, 1.0, -dz])
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        return normal
    
    def generate_heightmap(self):
        """Generate a heightmap using our NumPy-based noise implementation."""
        # Calculate grid dimensions
        grid_size = self.size // self.resolution + 1
        self.heightmap = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Set frequency (smaller value = smoother terrain)
        frequency = 0.01
        
        print("Generating terrain heightmap...")
        
        # Generate height values using our Perlin noise implementation
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * self.resolution - self.size/2
                z = j * self.resolution - self.size/2
                
                # Compute noise value using our fractal noise function
                height = self.noise.fractal(
                    x * frequency, 
                    z * frequency, 
                    octaves=self.octaves,
                    persistence=self.persistence, 
                    lacunarity=self.lacunarity
                )
                
                # Apply scaling and adjustments
                height = height * self.scale * 20  # Scale to reasonable heights
                
                # Apply additional terrain features
                # Create flat areas and valleys
                dist_from_center = np.sqrt(x**2 + z**2)
                if dist_from_center < 50:
                    # Create a relatively flat central area for takeoff/landing
                    height = max(0, height * 0.2)
                elif 50 <= dist_from_center < 120:
                    # Create a gentle valley around the center
                    valley_factor = (dist_from_center - 50) / 70  # 0 to 1
                    height = max(0, height * (0.2 + 0.8 * valley_factor))
                    
                    # Add a river in part of the valley
                    if -20 <= x <= 20:
                        river_depth = 5 * (1 - abs(x) / 20)
                        height = max(-river_depth, height - river_depth)
                        
                # Create some mountains in specific regions
                if x > 100 and z > 100:
                    mountain_factor = min(1.0, (x-100)/100) * min(1.0, (z-100)/100)
                    height += 15 * mountain_factor
                
                # Store height value
                self.heightmap[i][j] = height
    
    def generate_textures(self):
        """Generate texture coloring information based on terrain properties."""
        grid_size = len(self.heightmap)
        self.color_data = [[[0, 0, 0] for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Color parameters
        water_level = -1.0
        sand_level = 1.0
        grass_level = 10.0
        rock_level = 20.0
        snow_level = 30.0
        
        # Define colors (RGB)
        water_color = (0.0, 0.3, 0.8)
        sand_color = (0.8, 0.7, 0.4)
        grass_color = (0.3, 0.6, 0.2)
        rock_color = (0.5, 0.5, 0.5)
        snow_color = (0.9, 0.9, 0.95)
        
        # Generate color data based on height and other factors
        for i in range(grid_size):
            for j in range(grid_size):
                height = self.heightmap[i][j]
                x = i * self.resolution - self.size/2
                z = j * self.resolution - self.size/2
                
                # Base color based on height
                if height <= water_level:
                    color = water_color
                elif height <= sand_level:
                    t = (height - water_level) / (sand_level - water_level)
                    color = self.interpolate_color(water_color, sand_color, t)
                elif height <= grass_level:
                    t = (height - sand_level) / (grass_level - sand_level)
                    color = self.interpolate_color(sand_color, grass_color, t)
                elif height <= rock_level:
                    t = (height - grass_level) / (rock_level - grass_level)
                    color = self.interpolate_color(grass_color, rock_color, t)
                else:
                    t = min(1.0, (height - rock_level) / (snow_level - rock_level))
                    color = self.interpolate_color(rock_color, snow_color, t)
                
                # Add some noise to the colors for variation
                # Use a simplified noise function for color variation
                noise_value = (np.sin(x * 0.05) * np.sin(z * 0.05)) * 0.1
                color = (
                    max(0, min(1, color[0] + noise_value)),
                    max(0, min(1, color[1] + noise_value)),
                    max(0, min(1, color[2] + noise_value))
                )
                
                # Store the color
                self.color_data[i][j] = color
    
    def interpolate_color(self, color1, color2, t):
        """Linearly interpolate between two colors."""
        return (
            color1[0] * (1-t) + color2[0] * t,
            color1[1] * (1-t) + color2[1] * t,
            color1[2] * (1-t) + color2[2] * t
        )
    
    def compile_display_lists(self):
        """Compile terrain into OpenGL display lists for efficient rendering."""
        # Create display lists
        self.terrain_list = glGenLists(1)
        self.wireframe_list = glGenLists(1)
        
        grid_size = len(self.heightmap)
        
        # Compile solid terrain rendering
        glNewList(self.terrain_list, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Calculate actual positions
                x1 = i * self.resolution - self.size/2
                z1 = j * self.resolution - self.size/2
                y1 = self.heightmap[i][j]
                
                x2 = (i+1) * self.resolution - self.size/2
                z2 = j * self.resolution - self.size/2
                y2 = self.heightmap[i+1][j]
                
                x3 = i * self.resolution - self.size/2
                z3 = (j+1) * self.resolution - self.size/2
                y3 = self.heightmap[i][j+1]
                
                x4 = (i+1) * self.resolution - self.size/2
                z4 = (j+1) * self.resolution - self.size/2
                y4 = self.heightmap[i+1][j+1]
                
                # Get colors
                c1 = self.color_data[i][j]
                c2 = self.color_data[i+1][j]
                c3 = self.color_data[i][j+1]
                c4 = self.color_data[i+1][j+1]
                
                # Draw first triangle
                glColor3f(*c1)
                glVertex3f(x1, y1, z1)
                glColor3f(*c2)
                glVertex3f(x2, y2, z2)
                glColor3f(*c3)
                glVertex3f(x3, y3, z3)
                
                # Draw second triangle
                glColor3f(*c2)
                glVertex3f(x2, y2, z2)
                glColor3f(*c4)
                glVertex3f(x4, y4, z4)
                glColor3f(*c3)
                glVertex3f(x3, y3, z3)
        
        glEnd()
        glEndList()
        
        # Compile wireframe terrain rendering
        glNewList(self.wireframe_list, GL_COMPILE)
        glBegin(GL_LINES)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Calculate actual positions
                x1 = i * self.resolution - self.size/2
                z1 = j * self.resolution - self.size/2
                y1 = self.heightmap[i][j]
                
                x2 = (i+1) * self.resolution - self.size/2
                z2 = j * self.resolution - self.size/2
                y2 = self.heightmap[i+1][j]
                
                x3 = i * self.resolution - self.size/2
                z3 = (j+1) * self.resolution - self.size/2
                y3 = self.heightmap[i][j+1]
                
                # Draw grid lines
                glColor3f(0.0, 0.0, 0.0)  # Black lines
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
                
                glVertex3f(x1, y1, z1)
                glVertex3f(x3, y3, z3)
        
        glEnd()
        glEndList()
    
    def draw(self, wireframe=False):
        """Draw the terrain."""
        if wireframe:
            # Draw wireframe for debugging
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glCallList(self.wireframe_list)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            # Draw solid terrain
            glCallList(self.terrain_list)
