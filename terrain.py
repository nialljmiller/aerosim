"""
Simplified Terrain System for Flight Simulator
This module provides a working terrain generator that's compatible 
with the existing flight simulator code.
"""

import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import random
# In the imports section, add:
import os
import sys
try:
    from biome_manager import BiomeManager
    has_biome_manager = True
except ImportError:
    print("Warning: BiomeManager not available, using default terrain generation")
    has_biome_manager = False


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


    def __init__(self, chunk_x, chunk_z, chunk_size, resolution, noise_generator, terrain=None):
        self.chunk_x = chunk_x  # Chunk position in world (grid coordinates)
        self.chunk_z = chunk_z
        self.chunk_size = chunk_size  # Size of each chunk in world units
        self.resolution = resolution  # Resolution within chunk (vertex spacing)
        
        # Store reference to parent terrain system
        self.terrain = terrain
        
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
        """Generate heightmap and color data for this chunk."""
        # Calculate grid size within this chunk
        grid_size = int(self.chunk_size / self.resolution) + 1
        
        # Storage for height and color data
        self.heightmap = np.zeros((grid_size, grid_size))
        self.color_data = np.zeros((grid_size, grid_size, 3))
        
        # Check if biome manager is available through terrain instance
        has_biome_manager = hasattr(self.terrain, 'biome_manager') and self.terrain.biome_manager is not None
        
        # Generate height values for each vertex
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate world coordinates
                world_x = self.world_x + i * self.resolution
                world_z = self.world_z + j * self.resolution
                
                if has_biome_manager:
                    # Use biome manager for height
                    height = self.terrain.biome_manager.get_height(world_x, world_z)
                    
                    # Store height
                    self.heightmap[i, j] = height
                    
                    # Assign color using biome manager
                    color = self.terrain.biome_manager.get_color(world_x, world_z, height)
                    self.color_data[i, j] = color
                else:
                    # Original terrain generation code - keep as fallback
                    noise_scale = 0.005
                    height_scale = 40.0
                    
                    # Base terrain using fractal noise
                    base_height = self.noise.fractal(
                        world_x * noise_scale, 
                        world_z * noise_scale,
                        octaves=6,
                        persistence=0.5
                    )
                    
                    # Medium-scale variation
                    medium_detail = self.noise.fractal(
                        world_x * noise_scale * 4, 
                        world_z * noise_scale * 4,
                        octaves=3
                    ) * 0.25
                    
                    # Simple river system
                    river_noise = self.noise.fractal(
                        world_x * noise_scale * 2 + 500, 
                        world_z * noise_scale * 2 + 500,
                        octaves=1
                    )
                    
                    river_factor = 0
                    if 0.48 < river_noise < 0.53:
                        river_depth = 10.0
                        river_dist = min(abs(river_noise - 0.48), abs(river_noise - 0.53))
                        river_factor = (1.0 - river_dist / 0.05) * river_depth
                    
                    # Create mountains in some regions
                    mountain_mask = self.noise.fractal(
                        world_x * noise_scale * 0.25, 
                        world_z * noise_scale * 0.25,
                        octaves=2
                    )
                    
                    mountain_factor = 0
                    if mountain_mask > 0.55:
                        mountain_height = self.noise.fractal(
                            world_x * noise_scale * 0.5, 
                            world_z * noise_scale * 0.5,
                            octaves=4
                        )
                        mountain_factor = (mountain_mask - 0.55) * 2.0 * 60 * mountain_height
                    
                    # Calculate final height
                    height = (base_height + medium_detail) * height_scale
                    height = height - river_factor + mountain_factor
                    
                    # Ensure minimum terrain height
                    height = max(0.5, height)
                    
                    # Store height
                    self.heightmap[i, j] = height
                    
                    # Assign color based on height and features
                    self.assign_color(i, j, height, river_factor > 0)

    
    def assign_color(self, i, j, height, is_river):
        """Assign colors based on terrain features."""
        # Color parameters
        water_level = 2.0
        sand_level = 5.0
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
                
                # Calculate normals
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal1 = np.cross(edge1, edge2)
                if np.linalg.norm(normal1) > 0:
                    normal1 = normal1 / np.linalg.norm(normal1)
                else:
                    normal1 = np.array([0, 1, 0])  # Default up normal
                
                edge1 = v2 - v4
                edge2 = v3 - v4
                normal2 = np.cross(edge2, edge1)
                if np.linalg.norm(normal2) > 0:
                    normal2 = normal2 / np.linalg.norm(normal2)
                else:
                    normal2 = np.array([0, 1, 0])  # Default up normal
                
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

    def __init__(self, chunk_size=100, resolution=5, view_distance=800):
        self.chunk_size = chunk_size
        self.resolution = resolution
        self.view_distance = view_distance
        
        # Seed for consistent terrain generation
        self.seed = 42
        self.noise = PerlinNoise(seed=self.seed)
        
        # Initialize biome manager if available
        self.biome_manager = None
        if has_biome_manager:
            try:
                self.biome_manager = BiomeManager(self.noise, biome_scale=2000.0)
                print("Biome manager initialized successfully")
            except Exception as e:
                print(f"Error initializing biome manager: {e}")
                self.biome_manager = None
        
        # Dictionary to store loaded chunks
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
            # Create new chunk with reference to self (terrain)
            self.chunks[chunk_key] = TerrainChunk(
                chunk_x, chunk_z, 
                self.chunk_size, 
                self.resolution,
                self.noise,
                terrain=self  # Pass reference to self
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



                
        # NEW: Limit the number of new chunks per frame to reduce stuttering
        max_chunks_per_frame = 2  # Only load 2 new chunks per frame to spread the load
        chunk_count = 0
        # Load new chunks
        for chunk_key in chunks_to_load:
            chunk_count =+1
            if chunk_key not in self.chunks:
                self.get_or_create_chunk(*chunk_key)
            if chunk_count == max_chunks_per_frame:
                break


    def get_height(self, world_x, world_z):
        """Get terrain height at any world position."""
        # Use biome manager if available
        if self.biome_manager is not None:
            try:
                return self.biome_manager.get_height(world_x, world_z)
            except Exception as e:
                print(f"Error in biome height generation: {e}")
                # Fall back to chunk-based height if biome manager fails
        
        # Original chunk-based method as fallback
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)
        
        # If the chunk is loaded, query it
        if chunk_key in self.chunks:
            return self.chunks[chunk_key].get_height(world_x, world_z)
        
        # For unloaded chunks, generate height on-the-fly
        noise_scale = 0.005
        height_scale = 40.0
        
        # A simplified version of the chunk terrain generation
        base_height = self.noise.fractal(
            world_x * noise_scale, 
            world_z * noise_scale,
            octaves=4
        ) * height_scale
        
        # Ensure minimum height
        return max(0.5, base_height)

    # In the InfiniteTerrain class, update the get_terrain_normal method:
    def get_terrain_normal(self, world_x, world_z):
        """Calculate terrain normal at any world position."""
        # Use the height-based gradient method regardless of whether biome manager
        # is used, since we need to be consistent with get_height method
        
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
        
        # Normalize the normal vector
        length = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if length > 0:
            normal = normal / length
        else:
            normal = np.array([0, 1, 0])  # Default up normal
            
        return normal

    # Add this method to InfiniteTerrain class to get the current biome:
    def get_current_biome(self, world_x, world_z):
        """Get the biome type at the specified world position."""
        if self.biome_manager is not None:
            try:
                return self.biome_manager.get_biome_at_position(world_x, world_z)
            except Exception as e:
                print(f"Error getting biome at position: {e}")
                return "unknown"
        return "default"  # Default if biome manager is not available
    
    def draw(self, wireframe=False):
        """Draw all loaded terrain chunks."""
        for chunk in self.chunks.values():
            chunk.draw(wireframe)
    
    def cleanup(self):
        """Free all OpenGL resources."""
        for chunk in self.chunks.values():
            chunk.cleanup()
        self.chunks.clear()