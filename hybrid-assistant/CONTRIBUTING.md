# Contributing

Thank you for interest in contributing to the Hybrid Assistant project!

## How to Contribute

1. **Fork the repository** and create your feature branch (`git checkout -b feature/YourFeature`).
2. **Make your changes** with clear commit messages.
3. **Test locally** — ensure the GUI runs without errors and detections work as expected.
4. **Submit a pull request** with a description of your changes.

## Code Style

- Follow PEP 8 for Python code.
- Use meaningful variable and function names.
- Add comments for complex logic.

## Reporting Issues

If you find a bug or have a feature request, please open an issue with:
- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment info (OS, Python version, GPU/CPU mode)

## Testing

Before submitting, ensure:
- `python src/detection_gui.py` launches without errors
- Camera detection works
- Teaching dialog functions correctly
- No new syntax errors (run `python -c "import src.detection_gui"`)

## Pull Request Process

1. Update documentation if you change functionality.
2. Include a brief description of the feature or fix.
3. Reference any related issues.
4. Ensure tests pass and no new errors are introduced.

Thank you for making Hybrid Assistant better!
