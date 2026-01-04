
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

# Background Corrections

A PyQt6-based GUI application for applying baseline corrections and noise reduction to spectroscopy data.

## Features

- **Multiple Baseline Correction Algorithms**: Access to 50+ baseline correction algorithms from the `pybaselines` library, with library's original categories:
  - Polynomial (poly, modpoly, imodpoly, etc.)
  - Whittaker (asls, iasls, airpls, etc.)
  - Morphological (mor, imor, rolling_ball, etc.)
  - Spline (mixture_model, pspline variants, etc.)
  - Smoothing (snip, swima, ipsa, etc.)
  - Classification (dietrich, golotvin, fastchrom, etc.)
  - Miscellaneous (beads, custom_bc, etc.)

- **Noise Reduction**: Savitzky-Golay filter with configurable parameters

- **Real-time Visualization**: Interactive plot

- **Flexible Data Input**:
  - Load NumPy binary files (*.npy)
  - Optionally, load xrflab related files (*.atx, *.pck)
  - Generate synthetic spectra for testing

- **Dynamic Parameter Adjustment**: All algorithm parameters are automatically generated from function signatures with appropriate input widgets

## Installation

### Prerequisites

- Python 3.12 or higher
- poetry package manager (pip if poetry was not available)

### Required Dependencies

```bash
poetry install
```

If poetry was not available:

```bash
pip install .
```

### xrflab
If the user has access to `xrflab` library, installing it would give the user more
options for opening spectra files. 

```bash
poetry add xrflab_url
```

## Usage

### Starting the Application

```bash
python GUI.py
```

### Basic Workflow

1. **Load or Generate Data**
   - A synthetic spectrum is generated when the app is loaded
   - Click "Open Spectrum" to load existing data files
   - Click "Generate Random Spectrum" to generate random spectrum

2. **Configure Baseline Correction**
   - Enable/disable baseline correction with the checkbox
   - Select a category from the dropdown
   - Choose a specific algorithm
   - Adjust algorithm parameters as needed

3. **Configure Noise Reduction (Optional)**
   - Enable noise reduction with the checkbox
   - Adjust Savitzky-Golay filter parameters:
     - Window length (number of points)
     - Polynomial order
     - Derivative order
     - Extension mode

4. **View Results**
   - The plot updates automatically as you adjust parameters
   - Blue line: Original spectrum
   - Green line: Computed baseline
   - Orange line: Corrected spectrum

### Supported File Formats

| Extension | Description |
|-----------|-------------|
| `.npy` | NumPy Binary Array |


If `xrflab` is available, the below extensions are also supported:
| Extension | Description |
|-----------|-------------|
| `.atx` | Antique'X Spectrum |
| `.pck` | Antique'X Packet Spectrum |

## Application Architecture

The application is organized into modular components:

### Core Classes

- **`MainWindow`**: Main application window and UI orchestration
- **`SpectrumGenerator`**: Generates synthetic test spectra with Gaussian peaks
- **`BaselineProcessor`**: Manages baseline correction algorithms and parameters
- **`NoiseReducer`**: Applies Savitzky-Golay filtering
- **`PlotManager`**: Handles plot visualization and updates
- **`FileManager`**: Manages file I/O operations
- **`ParameterWidgetFactory`**: Creates appropriate widgets for algorithm parameters
- **`ConditionDialog`**: Dialog for selecting spectrum conditions from multi-condition files

## Algorithm Categories

### Polynomial Methods
Best for simple, smooth baselines. Uses polynomial fitting with various robustness improvements.

**Recommended starting points:**
- `modpoly`: Modified polynomial, good general-purpose algorithm
- `imodpoly`: Improved modpoly with better peak handling

### Whittaker Methods
Penalized least squares approaches. Excellent for complex baselines.

**Recommended starting points:**
- `asls`: Asymmetric least squares, very popular
- `arpls`: Adaptive reweighted penalized least squares

### Morphological Methods
Based on mathematical morphology operations. Good for uneven baselines.

**Recommended starting points:**
- `mor`: Basic morphological method
- `rolling_ball`: Simulates rolling a ball under the spectrum

### Spline Methods
Use spline interpolation for smooth baseline estimation.

**Recommended starting points:**
- `mixture_model`: Statistical mixture model approach
- `pspline_asls`: P-spline version of ASLS

### Smoothing Methods
Apply smoothing operations to estimate baseline.

**Recommended starting points:**
- `snip`: Statistics-sensitive non-linear iterative peak-clipping
- `noise_median`: Median-based smoothing

### Classification Methods
Use signal classification to identify baseline regions.

**Recommended starting points:**
- `dietrich`: Simple and fast classification method
- `std_distribution`: Standard deviation-based classification

## Tips and Best Practices

### Baseline Correction

1. **Start Simple**: Begin with polynomial methods like `modpoly` before trying more complex algorithms

2. **Adjust Lambda/Alpha**: For Whittaker methods, the smoothing parameter (usually `lam`) controls baseline smoothness:
   - Smaller values (1e3 - 1e5): More flexible baseline
   - Larger values (1e6 - 1e8): Smoother baseline

3. **Peak Sensitivity**: For spectra with sharp peaks, use methods with asymmetric weighting (asls, arpls, etc.)

4. **Iterative Refinement**: Some algorithms (imodpoly, iasls) iterate to improve results. Adjust iteration parameters if needed.

### Noise Reduction

1. **Window Length**: Should be odd and larger than polynomial order
   - Smaller windows: Preserve peak shape but less smoothing
   - Larger windows: More smoothing but can distort peaks

2. **Polynomial Order**: 
   - Order 2-3: Good for most spectra
   - Higher orders: Can introduce artifacts

### Performance

- Parameter changes are debounced (300ms delay) to prevent excessive recalculation
- For large datasets, start with fast algorithms (polynomial, morphological)
- Disable real-time updates if working with very large spectra

## Troubleshooting

### Common Issues

**Error: "Algorithm 'xxx' not found"**
- Ensure `pybaselines` is properly installed
- Some algorithms may require specific dependencies

**Error: "window_length must be less than or equal to the size of x"**
- Reduce the window length parameter in noise reduction
- Ensure your spectrum has enough data points

**Error: "polyorder must be less than window_length"**
- Reduce polynomial order or increase window length

**Plot shows error message**
- Check parameter values are appropriate for your data
- Try a different algorithm
- Verify input data is valid (no NaN or infinite values)

### Getting Help

If you encounter issues:
1. Check parameter values are within reasonable ranges
2. Try the default synthetic spectrum to isolate data vs. algorithm issues
3. Consult the [pybaselines documentation](https://pybaselines.readthedocs.io/) for algorithm-specific guidance


## License

This project uses the following open-source libraries:
- PyQt6: GPL v3
- pybaselines: BSD 3-Clause License
- pyqtgraph: MIT License
- NumPy, SciPy: BSD License

Please ensure compliance with respective licenses for distribution.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional baseline correction methods
- Export functionality for corrected spectra
- Batch processing capabilities
- Parameter presets/profiles
- Undo/redo functionality
- Additional visualization options (residuals, difference plots)

## References

- **pybaselines**: [Official Documentation](https://pybaselines.readthedocs.io/)
- **Savitzky-Golay filter**: [SciPy.savgol_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) 

---

**Author**: [Bahador Rousta Jorshary]  
**Last Updated**: January 2026