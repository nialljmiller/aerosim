"""
Integration code for adding the cloud and weather system to the flight simulator.
This module provides the necessary code modifications to incorporate the cloud
system into the existing flight simulator with the day/night cycle.
"""

def integrate_cloud_weather_system(main_file_code):
    """
    Function to integrate the cloud and weather system into the main flight simulator code.
    
    Args:
        main_file_code: The current contents of the main.py file
        
    Returns:
        Modified code with cloud and weather system integrated
    """
    # Import statements to add
    import_statement = """
import cloud_system  # Import the cloud and weather system
"""
    
    # Find an appropriate position to add the import statement
    # Let's add it after bird_system import
    import_pos = main_file_code.find("import bird_system")
    if import_pos != -1:
        main_file_code = main_file_code[:import_pos + len("import bird_system")] + import_statement + main_file_code[import_pos + len("import bird_system"):]
    
    # Initialize cloud system after bird system initialization
    init_code = """
    # Initialize cloud and weather system
    print("Initializing cloud and weather system...")
    clouds = cloud_system.CloudSystem(terrain)
    # Start with clear weather
    clouds.weather_type = "clear"
"""
    
    init_pos = main_file_code.find("# Initialize bird flocking system")
    if init_pos != -1:
        end_of_bird_init = main_file_code.find("\n", init_pos + len("# Initialize bird flocking system"))
        if end_of_bird_init != -1:
            main_file_code = main_file_code[:end_of_bird_init + 1] + init_code + main_file_code[end_of_bird_init + 1:]
    
    # Update clouds system in the main game loop after updating birds
    update_code = """
        # Update cloud and weather system
        clouds.update(dt, celestial.current_time / (24 * 60 * 60), plane.position)
"""
    
    update_pos = main_file_code.find("birds.update(dt, plane.position, game_time)")
    if update_pos != -1:
        end_of_update = main_file_code.find("\n", update_pos)
        if end_of_update != -1:
            main_file_code = main_file_code[:end_of_update + 1] + update_code + main_file_code[end_of_update + 1:]
    
    # Draw clouds after drawing the celestial system but before terrain
    draw_code = """
        # Draw cloud and weather systems
        clouds.draw(cam_position)
"""
    
    draw_pos = main_file_code.find("# Draw celestial system (sky, sun, moon, stars)")
    if draw_pos != -1:
        end_of_celestial_draw = main_file_code.find("\n", main_file_code.find("celestial.draw(cam_position)"))
        if end_of_celestial_draw != -1:
            main_file_code = main_file_code[:end_of_celestial_draw + 1] + draw_code + main_file_code[end_of_celestial_draw + 1:]
    
    # Add weather control keys
    controls_code = """
                elif event.key == K_w and pygame.key.get_mods() & KMOD_CTRL:
                    # Cycle through weather types
                    weather_types = ["clear", "fair", "overcast", "stormy"]
                    current_idx = weather_types.index(clouds.weather_type)
                    next_idx = (current_idx + 1) % len(weather_types)
                    clouds.start_weather_transition(weather_types[next_idx])
"""
    
    controls_pos = main_file_code.find("elif event.key == K_PAGEDOWN:")
    if controls_pos != -1:
        end_of_controls = main_file_code.find("\n", main_file_code.find("celestial.current_time -= 3600", controls_pos))
        if end_of_controls != -1:
            main_file_code = main_file_code[:end_of_controls + 1] + controls_code + main_file_code[end_of_controls + 1:]
    
    # Add weather info to sky text
    hud_code = """
            f"WEATHER: {clouds.get_weather_status()}",
"""
    
    hud_pos = main_file_code.find('f"TIME: {celestial.get_time_of_day_string()}",')
    if hud_pos != -1:
        end_of_hud_line = main_file_code.find("\n", hud_pos)
        if end_of_hud_line != -1:
            main_file_code = main_file_code[:end_of_hud_line + 1] + hud_code + main_file_code[end_of_hud_line + 1:]
    
    # Add weather controls to HUD instructions
    instructions_code = """
            "CTRL+W: CYCLE WEATHER",
"""
    
    instructions_pos = main_file_code.find('"PGUP/PGDN: ADV/REW TIME",')
    if instructions_pos != -1:
        end_of_instruction = main_file_code.find("\n", instructions_pos)
        if end_of_instruction != -1:
            main_file_code = main_file_code[:end_of_instruction + 1] + instructions_code + main_file_code[end_of_instruction + 1:]
    
    # Add cleanup for cloud system
    cleanup_code = """
    # Clean up cloud system resources
    clouds.cleanup()
"""
    
    cleanup_pos = main_file_code.find("# Clean up resources")
    if cleanup_pos != -1:
        birds_cleanup_pos = main_file_code.find("birds.cleanup()", cleanup_pos)
        if birds_cleanup_pos != -1:
            end_of_birds_cleanup = main_file_code.find("\n", birds_cleanup_pos)
            if end_of_birds_cleanup != -1:
                main_file_code = main_file_code[:end_of_birds_cleanup + 1] + cleanup_code + main_file_code[end_of_birds_cleanup + 1:]
    
    return main_file_code

# Example usage:
with open('plane.py', 'r') as f:
    main_code = f.read()
    modified_code = integrate_cloud_weather_system(main_code)

with open('plane_with_weather.py', 'w') as f:
    f.write(modified_code)