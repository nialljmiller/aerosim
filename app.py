# api.py
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import numpy as np
import json
import sys
import os

# Import your simulation modules
import celestial_system
import terrain as plane_terrain
import model as plane_model
import tree_system
import bird_system
from plane import update_with_terrain

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = 'flight_simulator_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
sim_running = False
sim_thread = None
simulation_time = 0
last_update_time = 0

# Control inputs from the client
control_inputs = {
    'throttle_up': False,
    'throttle_down': False,
    'roll_left': False,
    'roll_right': False,
    'pitch_up': False,
    'pitch_down': False,
    'yaw_left': False,
    'yaw_right': False,
    'throttle_preset': None,
    'reset': False
}

# Simulation objects
plane = None
terrain = None
celestial = None
trees = None
birds = None
dt = 1/60  # Fixed timestep for simulation

def initialize_simulation():
    """Initialize the simulation objects and state"""
    global plane, terrain, celestial, trees, birds
    
    # Initialize terrain
    terrain = plane_terrain.InfiniteTerrain(chunk_size=100, resolution=50, view_distance=600)
    
    # Initialize celestial system
    celestial = celestial_system.CelestialSystem()
    celestial.initialize()
    celestial.set_time_progression(100.0)
    celestial.set_time(10)
    
    # Initialize tree system
    trees = tree_system.TreeSystem(terrain, density=0.008, max_trees_per_chunk=200)
    
    # Initialize bird system
    birds = bird_system.BirdSystem(terrain, max_flocks=15)
    
    # Simulation parameters
    params = {
        'gravity': 9.81,
        'air_density': 1.225,
        'mass': 500.0,
        'wing_area': 50.0,
        'drag_coefficient': 0.2,
        'lift_coefficient': 1.5,
        'propeller_thrust': 5000.0,
        'glide_ratio': 15.0,
    }
    
    # Create plane
    initial_position = [0.0, 30.0, 0.0]
    initial_velocity = [10.0, 0.0, 0.0]
    forward = [1.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    plane = plane_model.Plane3D(initial_position, initial_velocity, forward, up, params)
    plane.throttle = 0.3
    
    # Update terrain chunks for initial position
    terrain.update_chunks(initial_position)

def simulation_thread_function():
    """Main simulation loop that runs in a separate thread"""
    global sim_running, simulation_time, plane, terrain, celestial, dt, last_update_time
    
    game_time = 0.0
    
    while sim_running:
        # Handle control inputs
        handle_controls()
        
        # Update physics
        celestial.update(dt)
        terrain.update_chunks(plane.position)
        trees.update(plane.position)
        birds.update(dt, plane.position, game_time)
        
        # Update plane physics
        update_with_terrain(plane, dt, terrain)
        
        # Increase simulation time
        simulation_time += dt
        game_time += dt
        
        # If it's time to send an update to clients
        if simulation_time - last_update_time >= 0.05:  # 20 updates per second
            # Get simulation state and broadcast to clients
            sim_state = get_simulation_state()
            socketio.emit('simulation_state', sim_state)
            last_update_time = simulation_time
        
        # Sleep to control simulation speed
        time.sleep(max(0, 0.01))  # Up to 100 physics updates per second

def handle_controls():
    """Process the current control inputs and apply them to the plane"""
    global plane, control_inputs
    
    # Apply flight controls
    pitch = roll = yaw = 0.0
    control_speed = np.radians(45)  # 45 deg/sec
    
    if control_inputs['pitch_up']:
        pitch = control_speed
    if control_inputs['pitch_down']:
        pitch = -control_speed
    if control_inputs['roll_left']:
        roll = control_speed
    if control_inputs['roll_right']:
        roll = -control_speed
    if control_inputs['yaw_left']:
        yaw = control_speed
    if control_inputs['yaw_right']:
        yaw = -control_speed
    if control_inputs['throttle_up']:
        plane.throttle = min(1.0, plane.throttle + 0.01)
    if control_inputs['throttle_down']:
        plane.throttle = max(0.0, plane.throttle - 0.01)
    
    # Handle throttle presets
    if control_inputs['throttle_preset'] is not None:
        preset_value = control_inputs['throttle_preset'] / 100.0
        plane.throttle = preset_value
        control_inputs['throttle_preset'] = None
    
    # Reset plane if requested
    if control_inputs['reset']:
        plane.position = np.array([0.0, 30.0, 0.0])
        plane.velocity = np.array([10.0, 0.0, 0.0])
        plane.forward = np.array([1.0, 0.0, 0.0])
        plane.up = np.array([0.0, 1.0, 0.0])
        plane.right = np.array([0.0, 0.0, 1.0])
        plane.throttle = 0.3
        plane.damage = 0.0
        control_inputs['reset'] = False
    
    # Apply controls to the plane
    plane.apply_controls(pitch, -roll, yaw, dt)

def get_simulation_state():
    """Get the current state of the simulation for sending to the client"""
    global plane, terrain, celestial
    
    # Get current flight data
    speed = np.linalg.norm(plane.velocity)
    altitude = plane.position[1]
    ground_height = terrain.get_height(plane.position[0], plane.position[2])
    height_above_ground = altitude - ground_height
    
    # Get sun position for lighting
    sun_altitude = np.degrees(np.arctan2(celestial.sun.position[1], 
                                           np.sqrt(celestial.sun.position[0]**2 + celestial.sun.position[2]**2)))
    
    # Base state to send to client
    state = {
        'plane': {
            'position': plane.position.tolist(),
            'forward': plane.forward.tolist(),
            'up': plane.up.tolist(),
            'right': plane.right.tolist(),
            'velocity': plane.velocity.tolist(),
            'throttle': float(plane.throttle),
            'damage': float(plane.damage),
            'is_grounded': plane.is_grounded
        },
        'telemetry': {
            'speed': float(speed),
            'altitude': float(height_above_ground),
            'time_of_day': celestial.get_time_of_day_string()
        },
        'environment': {
            'sun_position': celestial.sun.position.tolist(),
            'sun_altitude': float(sun_altitude),
            'time_of_day': celestial.current_time / (24 * 60 * 60)  # Normalized 0-1
        },
        'terrain': get_terrain_data()
    }
    
    return state

def get_terrain_data():
    """Get terrain data for the area around the plane"""
    global plane, terrain
    
    # Get a sample of terrain heights around the plane
    terrain_data = {
        'center': [float(plane.position[0]), float(plane.position[2])],
        'heights': []
    }
    
    # Get current chunk
    chunk_x, chunk_z = terrain.get_chunk_position(plane.position[0], plane.position[2])
    
    # Return list of chunk coordinates that are loaded
    loaded_chunks = list(terrain.chunks.keys())
    
    return {
        'current_chunk': [chunk_x, chunk_z],
        'loaded_chunks': loaded_chunks
    }

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/state')
def get_state():
    """REST endpoint for getting current simulation state"""
    if not sim_running:
        return jsonify({"error": "Simulation not running"}), 400
    
    return jsonify(get_simulation_state())

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connection_status', {'status': 'connected'})

@socketio.on('start_simulation')
def handle_start_simulation():
    """Start the simulation"""
    global sim_running, sim_thread
    
    if not sim_running:
        # Initialize simulation
        initialize_simulation()
        
        # Start simulation thread
        sim_running = True
        sim_thread = threading.Thread(target=simulation_thread_function)
        sim_thread.daemon = True
        sim_thread.start()
        
        emit('simulation_status', {'status': 'started'})
    else:
        emit('simulation_status', {'status': 'already_running'})

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop the simulation"""
    global sim_running
    
    if sim_running:
        sim_running = False
        emit('simulation_status', {'status': 'stopped'})
    else:
        emit('simulation_status', {'status': 'not_running'})

@socketio.on('control_input')
def handle_control_input(data):
    """Handle control input from client"""
    global control_inputs
    
    if 'control' in data and 'state' in data:
        control = data['control']
        state = data['state']
        
        if control in control_inputs:
            control_inputs[control] = state
            
        # Special handling for throttle presets
        if control == 'throttle_preset' and state is not None:
            try:
                control_inputs['throttle_preset'] = float(state)
            except (ValueError, TypeError):
                pass

@socketio.on('request_terrain_mesh')
def handle_request_terrain_mesh(data):
    """Handle request for terrain mesh data"""
    if not sim_running:
        return
    
    # Extract requested region from data
    x_center = data.get('x', 0)
    z_center = data.get('z', 0)
    size = data.get('size', 200)
    resolution = data.get('resolution', 20)
    
    # Generate terrain mesh data
    mesh_data = generate_terrain_mesh(x_center, z_center, size, resolution)
    
    # Send mesh data
    emit('terrain_mesh', mesh_data)

def generate_terrain_mesh(x_center, z_center, size, resolution):
    """Generate terrain mesh data for WebGL rendering"""
    global terrain
    
    # Calculate grid dimensions
    grid_size = int(size / resolution)
    half_size = size / 2
    
    # Prepare data structures
    vertices = []
    indices = []
    normals = []
    
    # Generate vertices and heights
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            # Calculate world coordinates
            x = x_center - half_size + i * resolution
            z = z_center - half_size + j * resolution
            y = terrain.get_height(x, z)
            
            # Add vertex
            vertices.extend([x, y, z])
            
            # Calculate normal
            normal = terrain.get_terrain_normal(x, z)
            normals.extend([normal[0], normal[1], normal[2]])
    
    # Generate triangle indices
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate indices for the quad's two triangles
            a = i * (grid_size + 1) + j
            b = i * (grid_size + 1) + (j + 1)
            c = (i + 1) * (grid_size + 1) + j
            d = (i + 1) * (grid_size + 1) + (j + 1)
            
            # Triangle 1
            indices.extend([a, b, c])
            # Triangle 2
            indices.extend([b, d, c])
    
    return {
        'vertices': vertices,
        'indices': indices,
        'normals': normals,
        'center': [x_center, 0, z_center],
        'size': size
    }

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
