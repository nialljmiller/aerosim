from PyInstaller.utils.hooks import collect_all

# Collect all OpenGL related packages
datas, binaries, hiddenimports = collect_all('OpenGL')
datas += collect_all('OpenGL.platform')[0]
datas += collect_all('OpenGL.arrays')[0]
