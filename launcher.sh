#!/bin/bash

# Start Xvfb with a virtual screen
Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!

# Wait a moment for Xvfb to initialize
sleep 2

# Set the DISPLAY environment variable to use our virtual display
export DISPLAY=:99

# Run the flight simulator
python app.py

# Clean up when the app exits
kill $XVFB_PID
