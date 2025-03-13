import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLU import *

class TreeSystem:
    """System for generating and rendering pine/fir trees in the game world."""
    def __init__(self, terrain, density=0.005, max_trees_per_chunk=150):
        self.terrain = terrain
        self.density = density  # Tree density (0-1)
        self.max_trees_per_chunk = max_trees_per_chunk
        
        # Dictionary to store tree positions for each chunk
        self.tree_chunks = {}
        
        # Compilation flag
        self.display_list_compiled = False
        self.display_list_id = None
        
        # Tree variations
        self.tree_variations = 3
        self.variation_lists = []
        
        # Tree colors
        self.trunk_color = (0.45, 0.30, 0.15)  # Brown
        self.pine_colors = [
            (0.0, 0.5, 0.2),    # Dark green
            (0.1, 0.6, 0.2),    # Medium green
            (0.15, 0.55, 0.15), # Olive green
        ]
    
    def compile_tree_models(self):
        """Compile display lists for different tree variations."""
        if self.display_list_compiled:
            return
        
        # Create display lists for tree variations
        base_list_id = glGenLists(self.tree_variations)
        
        for i in range(self.tree_variations):
            list_id = base_list_id + i
            self.variation_lists.append(list_id)
            
            # Compile display list for this tree variation
            glNewList(list_id, GL_COMPILE)
            self._draw_tree_model(i)
            glEndList()
        
        self.display_list_compiled = True
    
    def _draw_tree_model(self, variation):
        """Draw a single pine/fir tree model."""
        # Tree parameters vary by type
        if variation == 0:
            # Tall, thin pine
            trunk_height = 5.0
            trunk_radius = 0.3
            layers = 8
            max_radius = 3.0
            min_radius = 0.8
            layer_height = 1.2
            color_idx = 0
        elif variation == 1:
            # Medium, fuller pine
            trunk_height = 4.0
            trunk_radius = 0.4
            layers = 6
            max_radius = 3.5
            min_radius = 1.0
            layer_height = 1.4
            color_idx = 1
        else:
            # Shorter, wider pine
            trunk_height = 3.5
            trunk_radius = 0.45
            layers = 5
            max_radius = 4.0
            min_radius = 1.2
            layer_height = 1.5
            color_idx = 2
        
        # Draw trunk (brown cylinder)
        glColor3f(*self.trunk_color)
        
        # Draw trunk using polygons
        glBegin(GL_QUAD_STRIP)
        segments = 8
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = trunk_radius * np.cos(angle)
            z = trunk_radius * np.sin(angle)
            
            glVertex3f(x, 0, z)
            glVertex3f(x, trunk_height, z)
        glEnd()
        
        # Draw top and bottom trunk caps
        glBegin(GL_POLYGON)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = trunk_radius * np.cos(angle)
            z = trunk_radius * np.sin(angle)
            glVertex3f(x, 0, z)
        glEnd()
        
        glBegin(GL_POLYGON)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = trunk_radius * np.cos(angle)
            z = trunk_radius * np.sin(angle)
            glVertex3f(x, trunk_height, z)
        glEnd()
        
        # Draw pine layers (green cones)
        glColor3f(*self.pine_colors[color_idx])
        
        height_offset = trunk_height * 0.3  # Start foliage before trunk ends
        
        for layer in range(layers):
            # Calculate layer properties
            rel_height = layer / max(1, layers - 1)
            layer_y = height_offset + (trunk_height - height_offset) * rel_height
            
            # Invert rel_height to make base wider
            inv_rel_height = 1.0 - rel_height
            layer_radius = min_radius + (max_radius - min_radius) * inv_rel_height
            
            # Draw cone for this layer
            glBegin(GL_TRIANGLE_FAN)
            
            # Cone apex
            apex_y = layer_y + layer_height
            glVertex3f(0, apex_y, 0)
            
            # Cone base
            segments = 12
            for i in range(segments + 1):
                angle = 2.0 * np.pi * i / segments
                x = layer_radius * np.cos(angle)
                z = layer_radius * np.sin(angle)
                glVertex3f(x, layer_y, z)
            
            glEnd()
        
        # Draw a small cone on top
        top_y = trunk_height# + layer_height * (layers - 1)
        top_radius = min_radius * 0.7
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, top_y + layer_height, 0)  # Apex
        
        segments = 8
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = top_radius * np.cos(angle)
            z = top_radius * np.sin(angle)
            glVertex3f(x, top_y, z)
        glEnd()
    
    def generate_trees_for_chunk(self, chunk_key):
        """Generate tree positions for a terrain chunk."""
        if chunk_key in self.tree_chunks:
            # Trees already generated for this chunk
            return self.tree_chunks[chunk_key]
        
        chunk_x, chunk_z = chunk_key
        chunk_size = self.terrain.chunk_size
        
        # World space coordinates of this chunk
        world_x = chunk_x * chunk_size
        world_z = chunk_z * chunk_size
        
        # Determine suitable number of trees based on terrain type
        # We'll use noise to create natural patterns of tree density
        chunk_mid_x = world_x + chunk_size / 2
        chunk_mid_z = world_z + chunk_size / 2
        
        # Use terrain's noise generator for consistent patterns
        forest_density_noise = self.terrain.noise.fractal(
            chunk_mid_x * 0.001, 
            chunk_mid_z * 0.001,
            octaves=2
        )
        
        # Scale density based on noise
        local_density = self.density * (0.5 + forest_density_noise)
        
        # Calculate number of trees
        area = chunk_size * chunk_size
        target_tree_count = int(area * local_density)
        tree_count = min(target_tree_count, self.max_trees_per_chunk)
        
        # Generate tree positions
        tree_data = []
        
        # Margin from chunk edge to prevent trees from appearing abruptly
        margin = 5.0
        
        for _ in range(tree_count):
            # Random position within chunk (with margin)
            local_x = margin + random.random() * (chunk_size - 2 * margin)
            local_z = margin + random.random() * (chunk_size - 2 * margin)
            
            # Convert to world coordinates
            tree_x = world_x + local_x
            tree_z = world_z + local_z
            
            # Get terrain height at this position
            ground_y = self.terrain.get_height(tree_x, tree_z)
            
            # Check if this is a good place for a tree
            # No trees on steep slopes or in water/rivers
            terrain_normal = self.terrain.get_terrain_normal(tree_x, tree_z)
            slope = np.arccos(np.clip(terrain_normal[1], 0, 1))  # Angle from vertical
            
            # Skip if the slope is too steep (> ~20 degrees)
            if slope > 0.35:
                continue
                
            # Skip if height is too low (water/river)
            if ground_y < 3.0:
                continue
                
            # Skip if too high (above tree line)
            if ground_y > 50.0:
                continue
            
            # Choose tree variation based on height and noise
            variation_noise = self.terrain.noise.noise2d(tree_x * 0.05, tree_z * 0.05)
            
            # Bias toward different tree types based on elevation
            if ground_y > 30.0:
                # High elevation - more tall, thin pines
                type_bias = 0.3
            elif ground_y > 15.0:
                # Medium elevation - more medium pines
                type_bias = 0.0
            else:
                # Low elevation - more short, wide pines
                type_bias = -0.3
            
            # Determine tree variation
            tree_type = int((variation_noise + type_bias + 1.0) / 2.0 * self.tree_variations)
            tree_type = max(0, min(self.tree_variations - 1, tree_type))
            
            # Random rotation
            rotation = random.random() * 360.0
            
            # Random scale variation (80% to 120% of normal size)
            scale = 0.8 + random.random() * 0.4
            
            # Add tree data
            tree_data.append({
                'position': (tree_x, ground_y, tree_z),
                'type': tree_type,
                'rotation': rotation,
                'scale': scale
            })
        
        # Store and return tree data
        self.tree_chunks[chunk_key] = tree_data
        return tree_data
    
    def update(self, camera_position):
        """Update visible trees based on camera position."""
        # Ensure tree models are compiled
        if not self.display_list_compiled:
            self.compile_tree_models()
        
        # Get chunks that should have trees, same as terrain's visible chunks
        chunk_x, chunk_z = self.terrain.get_chunk_position(
            camera_position[0], camera_position[2]
        )
        
        # Generate a view distance box around the camera
        chunk_view_distance = self.terrain.view_distance // self.terrain.chunk_size
        
        # Pre-generate trees for visible chunks
        for x in range(chunk_x - chunk_view_distance, chunk_x + chunk_view_distance + 1):
            for z in range(chunk_z - chunk_view_distance, chunk_z + chunk_view_distance + 1):
                chunk_key = (x, z)
                # Only generate if this chunk is loaded in terrain
                if chunk_key in self.terrain.chunks:
                    self.generate_trees_for_chunk(chunk_key)
        
        # Clean up tree data for unloaded chunks
        chunks_to_remove = []
        for chunk_key in self.tree_chunks:
            if chunk_key not in self.terrain.chunks:
                chunks_to_remove.append(chunk_key)
        
        for chunk_key in chunks_to_remove:
            del self.tree_chunks[chunk_key]
    
    def draw(self, camera_position):
        """Render visible trees using display lists."""
        if not self.display_list_compiled:
            self.compile_tree_models()
        
        # Calculate squared view distance for culling
        max_view_dist_sq = (self.terrain.view_distance * 0.6) ** 2
        
        # Render trees with frustum culling
        for chunk_key, trees in self.tree_chunks.items():
            # Cull entire chunks that are too far
            chunk_x, chunk_z = chunk_key
            chunk_mid_x = (chunk_x + 0.5) * self.terrain.chunk_size
            chunk_mid_z = (chunk_z + 0.5) * self.terrain.chunk_size
            
            chunk_dist_sq = ((chunk_mid_x - camera_position[0]) ** 2 + 
                             (chunk_mid_z - camera_position[2]) ** 2)
            
            if chunk_dist_sq > max_view_dist_sq * 1.2:
                continue
            
            # Render individual trees with instance culling
            for tree in trees:
                pos = tree['position']
                
                # Distance-based culling
                dist_sq = ((pos[0] - camera_position[0]) ** 2 + 
                           (pos[2] - camera_position[2]) ** 2)
                
                if dist_sq > max_view_dist_sq:
                    continue
                
                # LOD based on distance
                lod_threshold = max_view_dist_sq * 0.25
                detailed = dist_sq < lod_threshold
                
                # Draw the tree
                glPushMatrix()
                
                # Position
                glTranslatef(pos[0], pos[1], pos[2])
                
                # Rotation
                glRotatef(tree['rotation'], 0, 1, 0)
                
                # Scale
                scale = tree['scale']
                glScalef(scale, scale, scale)
                
                # Draw tree model
                glCallList(self.variation_lists[tree['type']])
                
                glPopMatrix()
    
    def cleanup(self):
        """Free OpenGL resources."""
        if self.display_list_compiled:
            for list_id in self.variation_lists:
                glDeleteLists(list_id, 1)
            self.display_list_compiled = False