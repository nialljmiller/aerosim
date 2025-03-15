import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {
    "packages": ["OpenGL", "pygame", "numpy", "math", "random"],
    "excludes": [],
    "include_files": []  # Add any texture files or assets here
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use this to hide the console window

setup(
    name="FlightSimulator",
    version="1.0",
    description="3D Flight Simulator",
    options={"build_exe": build_exe_options},
    executables=[Executable("plane.py", base=base)]
)
