"""
Plane Model Renderer for Flight Simulator
This module contains all functions related to rendering the 3D aircraft model.
Separating these functions allows for easier modification of the aircraft appearance
without changing the core simulation code.
"""

import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame

def draw_plane(plane, wireframe_mode=False):
    """Draw the plane using the enhanced detailed model."""
    glPushMatrix()
    # First translate to the plane's position
    glTranslatef(plane.position[0], plane.position[1], plane.position[2])
    
    # Build a rotation matrix from the plane's basis vectors (COLUMN-MAJOR ORDER for OpenGL)
    # Each column represents one of the plane's basis vectors
    matrix = [
        plane.forward[0], plane.forward[1], plane.forward[2], 0,
        plane.up[0], plane.up[1], plane.up[2], 0,
        plane.right[0], plane.right[1], plane.right[2], 0,
        0, 0, 0, 1
    ]
    glMultMatrixf(matrix)
    
    # Draw the enhanced plane model with all details
    draw_detailed_plane(plane, wireframe_mode)
    
    # Optional: Draw velocity vector for debugging
    glColor3f(1, 0, 0)  # Red for velocity
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    # Scale velocity vector for better visualization
    vel_scale = 0.1
    glVertex3f(
        plane.velocity[0] * vel_scale,
        plane.velocity[1] * vel_scale,
        plane.velocity[2] * vel_scale
    )
    glEnd()
    
    # Draw collision points for debugging (useful to see where the plane is touching the ground)
    for collision in plane.collision_points:
        if 'wheel' in collision:
            # Wheel collision point - draw in green
            world_point = collision['world_point']
            normal = collision['normal']
            
            glPushMatrix()
            glTranslatef(world_point[0] - plane.position[0], 
                        world_point[1] - plane.position[1], 
                        world_point[2] - plane.position[2])
            glColor3f(0, 1, 0)  # Green for wheel contact
            try:
                glutSolidSphere(0.2, 8, 8)  # Small sphere
            except:
                # Fallback if GLUT is not available
                pass
            
            # Draw the terrain normal
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(normal[0], normal[1], normal[2])
            glEnd()
            glPopMatrix()
        elif 'point_idx' in collision:
            # Regular collision point - draw in red
            world_point = collision['world_point']
            normal = collision['normal']
            
            glPushMatrix()
            glTranslatef(world_point[0] - plane.position[0], 
                        world_point[1] - plane.position[1], 
                        world_point[2] - plane.position[2])
            glColor3f(1, 0, 0)  # Red for collision
            try:
                glutSolidSphere(0.2, 8, 8)  # Small sphere
            except:
                # Fallback if GLUT is not available
                pass
            
            # Draw the terrain normal
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(normal[0], normal[1], normal[2])
            glEnd()
            glPopMatrix()
    
    glPopMatrix()

def draw_detailed_plane(plane, wire_mode=True):
    """Draw the plane model with enhanced detail, including landing gear.
    This function should be called from your rendering code after importing OpenGL.
    """
    try:
        # Set color based on damage
        damage_factor = plane.damage
        undamaged_color = (1.0, 1.0, 0.0)  # Yellow
        damaged_color = (1.0, 0.0, 0.0)    # Red
        color = (
            undamaged_color[0] * (1-damage_factor) + damaged_color[0] * damage_factor,
            undamaged_color[1] * (1-damage_factor) + damaged_color[1] * damage_factor,
            undamaged_color[2] * (1-damage_factor) + damaged_color[2] * damage_factor
        )
        
        # Store original line width to restore later
        original_line_width = glGetFloatv(GL_LINE_WIDTH)
        
        # Use thicker lines for better visibility
        glLineWidth(2.0)
        
        # ---------- Draw fuselage ----------
        glColor3f(*color)
        
        # Main fuselage (3D shape)
        fuselage_length = plane.fuselage_length
        fuselage_height = plane.fuselage_height
        fuselage_width = plane.fuselage_width
        
        # Top profile
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, fuselage_height/4, 0)  # Nose
        glVertex3f(fuselage_length/4, fuselage_height/2, 0)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/2, 0)  # Top middle
        glVertex3f(-fuselage_length/2, fuselage_height/4, 0)  # Top rear
        glEnd()
        
        # Bottom profile
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, -fuselage_height/4, 0)  # Nose bottom
        glVertex3f(fuselage_length/4, -fuselage_height/2, 0)  # Bottom front
        glVertex3f(-fuselage_length/3, -fuselage_height/2, 0)  # Bottom middle
        glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)  # Bottom rear
        glEnd()
        
        # Side struts to connect top and bottom
        glBegin(GL_LINES)
        # Front
        glVertex3f(fuselage_length/2, fuselage_height/4, 0)
        glVertex3f(fuselage_length/2, -fuselage_height/4, 0)
        
        # Mid front
        glVertex3f(fuselage_length/4, fuselage_height/2, 0)
        glVertex3f(fuselage_length/4, -fuselage_height/2, 0)
        
        # Mid rear
        glVertex3f(-fuselage_length/3, fuselage_height/2, 0)
        glVertex3f(-fuselage_length/3, -fuselage_height/2, 0)
        
        # Rear
        glVertex3f(-fuselage_length/2, fuselage_height/4, 0)
        glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)
        glEnd()
        
        # Side profiles (port side)
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, 0, -fuselage_width/2)  # Nose side
        glVertex3f(fuselage_length/4, fuselage_height/3, -fuselage_width/2)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/3, -fuselage_width/2)  # Top rear
        glVertex3f(-fuselage_length/2, 0, -fuselage_width/2)  # Tail
        glVertex3f(-fuselage_length/3, -fuselage_height/3, -fuselage_width/2)  # Bottom rear
        glVertex3f(fuselage_length/4, -fuselage_height/3, -fuselage_width/2)  # Bottom front
        glEnd()
        
        # Side profile (starboard side)
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/2, 0, fuselage_width/2)  # Nose side
        glVertex3f(fuselage_length/4, fuselage_height/3, fuselage_width/2)  # Top front
        glVertex3f(-fuselage_length/3, fuselage_height/3, fuselage_width/2)  # Top rear
        glVertex3f(-fuselage_length/2, 0, fuselage_width/2)  # Tail
        glVertex3f(-fuselage_length/3, -fuselage_height/3, fuselage_width/2)  # Bottom rear
        glVertex3f(fuselage_length/4, -fuselage_height/3, fuselage_width/2)  # Bottom front
        glEnd()
        
        # Cockpit lines
        glBegin(GL_LINE_LOOP)
        glVertex3f(fuselage_length/5, fuselage_height/2.2, -fuselage_width/4)
        glVertex3f(fuselage_length/5, fuselage_height/2.2, fuselage_width/4)
        glVertex3f(-fuselage_length/8, fuselage_height/2.2, fuselage_width/4)
        glVertex3f(-fuselage_length/8, fuselage_height/2.2, -fuselage_width/4)
        glEnd()
        
        # ---------- Draw wings ----------
        wing_span = plane.wing_span
        wing_chord = plane.wing_chord
        
        # Wing color - blue with damage tint
        wing_color = (0.1, 0.3 * (1-damage_factor), 0.8 * (1-damage_factor))
        glColor3f(*wing_color)
        
        # Main wings - a quad with sweep
        glBegin(GL_LINE_LOOP)
        glVertex3f(wing_chord * 0.1, 0, -wing_span/2)  # Left wing tip leading edge
        glVertex3f(wing_chord * 0.1, 0, wing_span/2)   # Right wing tip leading edge
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2)  # Right wing tip trailing edge
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2)  # Left wing tip trailing edge
        glEnd()
        
        # Wing details (ribs and spars) for visual effect
        glBegin(GL_LINES)
        # Main spar
        glVertex3f(-wing_chord * 0.3, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.3, 0, wing_span/2)
        
        # Secondary spar
        glVertex3f(-wing_chord * 0.7, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.7, 0, wing_span/2)
        
        # Wing ribs
        sections = 10
        for i in range(sections + 1):
            z_pos = -wing_span/2 + i * wing_span/sections
            glVertex3f(wing_chord * 0.1, 0, z_pos)
            glVertex3f(-wing_chord * 0.9, 0, z_pos)
        glEnd()
        
        # Draw control surfaces (ailerons) with animation
        aileron_width = wing_span * 0.3
        aileron_depth = wing_chord * 0.2
        
        # Left aileron
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, -wing_span/2)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, -wing_span/2 + aileron_width)
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2 + aileron_width)
        glVertex3f(-wing_chord * 0.9, 0, -wing_span/2)
        glEnd()
        
        # Right aileron
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, wing_span/2 - aileron_width)
        glVertex3f(-wing_chord * 0.9 + aileron_depth, 0, wing_span/2)
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2)
        glVertex3f(-wing_chord * 0.9, 0, wing_span/2 - aileron_width)
        glEnd()
        
        # Flaps (animated based on flap position)
        flap_extend = plane.flaps * wing_chord * 0.15  # How far flaps extend down
        flap_width = wing_span * 0.4
        flap_start = wing_span * 0.05  # Distance from fuselage
        
        # Flap connection lines
        glBegin(GL_LINES)
        # Left flap
        glVertex3f(-wing_chord * 0.9, 0, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start - flap_width)
        
        glVertex3f(-wing_chord * 0.9, 0, -flap_start)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start)
        
        # Right flap
        glVertex3f(-wing_chord * 0.9, 0, flap_start)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start)
        
        glVertex3f(-wing_chord * 0.9, 0, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start + flap_width)
        glEnd()
        
        # Left flap surface
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9, 0, -flap_start)
        glVertex3f(-wing_chord * 0.9, 0, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start - flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, -flap_start)
        glEnd()
        
        # Right flap surface
        glBegin(GL_LINE_LOOP)
        glVertex3f(-wing_chord * 0.9, 0, flap_start)
        glVertex3f(-wing_chord * 0.9, 0, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start + flap_width)
        glVertex3f(-wing_chord * 0.9 - flap_extend, -flap_extend, flap_start)
        glEnd()
        
        # ---------- Draw tail surfaces ----------
        tail_color = (0.8, 0.4, 0.9)  # Purple
        glColor3f(*tail_color)
        
        # Horizontal stabilizer
        h_stab_span = wing_span * 0.4
        h_stab_chord = wing_chord * 0.5
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.8, 0, -h_stab_span/2)  # Left tip
        glVertex3f(-fuselage_length * 0.8, 0, h_stab_span/2)   # Right tip
        glVertex3f(-fuselage_length * 0.95, 0, h_stab_span/2)  # Right rear
        glVertex3f(-fuselage_length * 0.95, 0, -h_stab_span/2)  # Left rear
        glEnd()
        
        # Vertical stabilizer
        v_stab_height = fuselage_height * 1.5
        v_stab_chord = wing_chord * 0.6
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.7, 0, 0)  # Bottom front
        glVertex3f(-fuselage_length * 0.7, v_stab_height, 0)  # Top front
        glVertex3f(-fuselage_length * 0.95, v_stab_height * 0.8, 0)  # Top rear
        glVertex3f(-fuselage_length * 0.95, 0, 0)  # Bottom rear
        glEnd()
        
        # Elevator (animated)
        elevator_deflection = 0  # Add control for this if needed
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.9, elevator_deflection, -h_stab_span/2)
        glVertex3f(-fuselage_length * 0.9, elevator_deflection, h_stab_span/2)
        glVertex3f(-fuselage_length * 0.95, elevator_deflection, h_stab_span/2)
        glVertex3f(-fuselage_length * 0.95, elevator_deflection, -h_stab_span/2)
        glEnd()
        
        # Rudder (animated)
        rudder_deflection = 0  # Add control for this if needed
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-fuselage_length * 0.85, v_stab_height * 0.2, rudder_deflection)
        glVertex3f(-fuselage_length * 0.85, v_stab_height * 0.9, rudder_deflection)
        glVertex3f(-fuselage_length * 0.95, v_stab_height * 0.7, rudder_deflection)
        glVertex3f(-fuselage_length * 0.95, 0, rudder_deflection)
        glEnd()
        
        # ---------- Draw landing gear ----------
        # Only draw if partially or fully extended
        if plane.gear_state > 0.01:
            gear_color = (0.5, 0.5, 0.5)  # Gray for landing gear
            glColor3f(*gear_color)
            
            wheel_radius = plane.wheel_radius
            gear_height = plane.gear_height
            
            # Left main gear
            left_wheel_pos = plane.wheel_points['left'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                retract_angle = (1.0 - plane.gear_state) * 90  # Degrees
                retract_angle_rad = retract_angle * math.pi / 180.0
                
                # Calculate retracted position
                retracted_y = -plane.fuselage_height * 0.2
                retracted_z = left_wheel_pos[2] * 0.5
                
                # Interpolate
                left_wheel_pos[1] = left_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                left_wheel_pos[2] = left_wheel_pos[2] * plane.gear_state + retracted_z * (1.0 - plane.gear_state)
            
            # Adjust for compression
            left_wheel_pos[1] += plane.wheel_compression['left'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(0, 0, left_wheel_pos[2])  # Gear attachment point
            glVertex3f(left_wheel_pos[0], left_wheel_pos[1], left_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(left_wheel_pos[0], left_wheel_pos[1], left_wheel_pos[2])
            draw_wheel(wheel_radius)
            glPopMatrix()
            
            # Right main gear
            right_wheel_pos = plane.wheel_points['right'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                retract_angle = (1.0 - plane.gear_state) * 90  # Degrees
                retract_angle_rad = retract_angle * math.pi / 180.0
                
                # Calculate retracted position
                retracted_y = -plane.fuselage_height * 0.2
                retracted_z = right_wheel_pos[2] * 0.5
                
                # Interpolate
                right_wheel_pos[1] = right_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                right_wheel_pos[2] = right_wheel_pos[2] * plane.gear_state + retracted_z * (1.0 - plane.gear_state)
            
            # Adjust for compression
            right_wheel_pos[1] += plane.wheel_compression['right'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(0, 0, right_wheel_pos[2])  # Gear attachment point
            glVertex3f(right_wheel_pos[0], right_wheel_pos[1], right_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(right_wheel_pos[0], right_wheel_pos[1], right_wheel_pos[2])
            draw_wheel(wheel_radius)
            glPopMatrix()
            
            # Nose gear
            nose_wheel_pos = plane.wheel_points['nose'].copy()
            # Adjust for gear state (retraction animation)
            if plane.gear_state < 1.0:
                # Calculate retracted position (different for nose gear - retracts forward)
                retracted_y = -plane.fuselage_height * 0.2
                retracted_x = nose_wheel_pos[0] * 0.5
                
                # Interpolate
                nose_wheel_pos[1] = nose_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                nose_wheel_pos[0] = nose_wheel_pos[0] * plane.gear_state + retracted_x * (1.0 - plane.gear_state)
            
            # Adjust for compression
            nose_wheel_pos[1] += plane.wheel_compression['nose'] * wheel_radius
            
            # Draw strut
            glBegin(GL_LINES)
            glVertex3f(fuselage_length/4, -fuselage_height/3, 0)  # Gear attachment point
            glVertex3f(nose_wheel_pos[0], nose_wheel_pos[1], nose_wheel_pos[2])  # Wheel center
            glEnd()
            
            # Draw wheel using a circle of lines
            glPushMatrix()
            glTranslatef(nose_wheel_pos[0], nose_wheel_pos[1], nose_wheel_pos[2])
            draw_wheel(wheel_radius * 0.8)  # Smaller nose wheel
            glPopMatrix()
            
            # Tail wheel (if plane has one)
            if 'tail' in plane.wheel_points:
                tail_wheel_pos = plane.wheel_points['tail'].copy()
                # Adjust for gear state (minimal retraction for tail wheel)
                if plane.gear_state < 1.0:
                    # Calculate retracted position (tail wheel retracts less)
                    retracted_y = -plane.fuselage_height * 0.1
                    
                    # Interpolate
                    tail_wheel_pos[1] = tail_wheel_pos[1] * plane.gear_state + retracted_y * (1.0 - plane.gear_state)
                
                # Adjust for compression
                tail_wheel_pos[1] += plane.wheel_compression['tail'] * wheel_radius
                
                # Draw strut
                glBegin(GL_LINES)
                glVertex3f(-fuselage_length/2, -fuselage_height/4, 0)  # Gear attachment point
                glVertex3f(tail_wheel_pos[0], tail_wheel_pos[1], tail_wheel_pos[2])  # Wheel center
                glEnd()
                
                # Draw wheel using a circle of lines
                glPushMatrix()
                glTranslatef(tail_wheel_pos[0], tail_wheel_pos[1], tail_wheel_pos[2])
                draw_wheel(wheel_radius * 0.5)  # Smaller tail wheel
                glPopMatrix()
        
        # Draw propeller
        prop_color = (0.8, 0.8, 0.8)  # Light gray
        glColor3f(*prop_color)
        
        # Propeller spinner
        glPushMatrix()
        glTranslatef(fuselage_length/2 + 0.2, 0, 0)
        if wire_mode:
            glutWireSphere(plane.fuselage_width * 0.15, 8, 8)
        else:
            glutSolidSphere(plane.fuselage_width * 0.15, 8, 8)
        glPopMatrix()
        
        # Propeller blades (rotating)
        # Could animate rotation based on throttle
        prop_radius = wing_span * 0.15
        
        glBegin(GL_LINES)
        # Vertical blade
        glVertex3f(fuselage_length/2 + 0.2, prop_radius, 0)
        glVertex3f(fuselage_length/2 + 0.2, -prop_radius, 0)
        
        # Horizontal blade
        glVertex3f(fuselage_length/2 + 0.2, 0, prop_radius)
        glVertex3f(fuselage_length/2 + 0.2, 0, -prop_radius)
        glEnd()
        
        # Restore original line width
        glLineWidth(original_line_width)
        
    except Exception as e:
        # Fall back to basic drawing if there's an error
        print(f"Error rendering plane: {e}")
        basic_draw_plane(plane)

def draw_wheel(radius):
    """Draw a wheel as a circle of lines."""
    try:
        segments = 12
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(0, math.sin(angle) * radius, math.cos(angle) * radius)
        glEnd()
        
        # Add wheel hub details
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(0, math.sin(angle) * radius * 0.3, math.cos(angle) * radius * 0.3)
        glEnd()
        
        # Add spokes
        glBegin(GL_LINES)
        for i in range(6):  # 6 spokes
            angle = 2 * math.pi * i / 6
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            
            # Inner point
            glVertex3f(0, sin_a * radius * 0.3, cos_a * radius * 0.3)
            # Outer point
            glVertex3f(0, sin_a * radius, cos_a * radius)
        glEnd()
    except Exception as e:
        print(f"Error drawing wheel: {e}")

def basic_draw_plane(plane):
    """Simple fallback drawing function if the detailed one fails."""
    try:
        # Set color based on damage
        damage_factor = plane.damage
        color = (1.0, 1.0 * (1-damage_factor), 0.0)
        glColor3f(*color)
        
        # Simple fuselage
        glBegin(GL_LINE_LOOP)
        glVertex3f(1, 0, 0)  # Nose
        glVertex3f(-1, 0.5, 0)  # Top back
        glVertex3f(-1, -0.5, 0)  # Bottom back
        glEnd()
        
        # Wings
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, -plane.wing_span/2)  # Left wing tip
        glVertex3f(0, 0, plane.wing_span/2)   # Right wing tip
        glVertex3f(-0.5, 0, plane.wing_span/2)  # Right wing back
        glVertex3f(-0.5, 0, -plane.wing_span/2)  # Left wing back
        glEnd()
        
        # Tail
        glBegin(GL_LINE_LOOP)
        glVertex3f(-0.8, 0, 0)    # Bottom
        glVertex3f(-0.8, 0.5, 0)  # Top
        glVertex3f(-1, 0, 0)      # Back
        glEnd()
        
    except Exception as e:
        print(f"Error in basic plane rendering: {e}")