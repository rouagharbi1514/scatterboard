# Hotel Dashboard - Build Instructions

This guide will help you create executable files for the Hotel Dashboard that can be double-clicked to run on Windows and macOS.

## Prerequisites

### For All Platforms
- Python 3.8 or higher
- Virtual environment (recommended)
- All project dependencies installed

### Platform-Specific Requirements

#### macOS
- Xcode Command Line Tools: `xcode-select --install`
- Optional: `create-dmg` for DMG installer: `brew install create-dmg`

#### Windows
- Microsoft Visual C++ Build Tools (usually included with Python)
- Windows SDK (optional, for advanced features)

## Quick Start

### macOS/Linux
```bash
# Make the build script executable
chmod +x build.sh

# Run the build script
./build.sh
```

### Windows
```cmd
# Run the build script
build.bat
```

## Manual Build Process

If you prefer to build manually or the scripts don't work:

### Step 1: Set Up Environment
```bash
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pyinstaller pillow
```

### Step 2: Build the Executable
```bash
# Clean previous builds
rm -rf build/ dist/  # macOS/Linux
# or
rmdir /s build dist  # Windows

# Build with PyInstaller
pyinstaller hotel_dashboard.spec
```

## Output Files

### macOS
- **Application Bundle**: `dist/Hotel Dashboard.app`
  - Double-click to run
  - Can be moved to Applications folder
  - Self-contained (includes all dependencies)

- **DMG Installer** (if create-dmg is installed): `dist/Hotel Dashboard.dmg`
  - Professional installer package
  - Easy distribution to other Mac users

### Windows
- **Executable**: `dist/Hotel Dashboard.exe`
  - Double-click to run
  - Self-contained (includes all dependencies)
  - Can be distributed as a single file

### Linux
- **Executable**: `dist/Hotel Dashboard`
  - Run with: `./dist/Hotel\ Dashboard`
  - Self-contained (includes all dependencies)

## Distribution

### macOS
1. **Simple Distribution**: Share the `.app` file
2. **Professional Distribution**: Share the `.dmg` file
3. **App Store**: Requires additional code signing and notarization

### Windows
1. **Simple Distribution**: Share the `.exe` file
2. **Installer**: Use tools like NSIS or Inno Setup to create an installer
3. **Microsoft Store**: Requires packaging as MSIX

## Troubleshooting

### Common Issues

#### "Module not found" errors
- Ensure all dependencies are installed in the virtual environment
- Check `hotel_dashboard.spec` for missing hidden imports
- Add missing modules to the `hiddenimports` list

#### Large file size
- PyInstaller includes all dependencies, making files large (50-200MB typical)
- This is normal and ensures the app works on systems without Python

#### App won't start
- Check console output for error messages
- Ensure all data files are included in the spec file
- Test in the same environment where you built it first

#### macOS Security Warnings
- Right-click the app and select "Open" to bypass Gatekeeper
- For distribution, consider code signing: `codesign -s "Developer ID" Hotel\ Dashboard.app`

#### Windows Antivirus False Positives
- Some antivirus software flags PyInstaller executables
- This is a known issue with packaged Python applications
- Consider code signing for distribution

### Performance Tips

1. **Faster Startup**: Use `--onefile` flag in spec for single executable (slower startup but easier distribution)
2. **Smaller Size**: Use `--exclude-module` to remove unused packages
3. **Debug Mode**: Add `debug=True` in spec file for troubleshooting

## Advanced Configuration

Edit `hotel_dashboard.spec` to customize:
- Icon files
- Hidden imports
- Data files
- Executable name
- Bundle identifier (macOS)
- Version information

## Support

If you encounter issues:
1. Check the build logs for specific error messages
2. Ensure your virtual environment has all required packages
3. Try building in a fresh virtual environment
4. Check PyInstaller documentation for platform-specific issues

---

**Note**: The first build may take several minutes as PyInstaller analyzes dependencies. Subsequent builds are typically faster.