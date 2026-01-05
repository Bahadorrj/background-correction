import sys
import inspect

import numpy as np
import pyqtgraph as pg

from pathlib import Path
from typing import Any
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QScrollArea,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QPushButton,
    QSplitter,
    QDialog,
    QFileDialog,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from pybaselines import Baseline
from scipy.signal import savgol_filter

try:
    import xrflab as xrf  # type: ignore

    EXTENSION_MAPPER = {
        ".atx": "Antique'X Spectrum (*.atx)",
        ".pck": "Antique'X Packet Spectrum (*.pck)",
        ".npy": "NumPy Binary(*.npy)",
    }
except ModuleNotFoundError:
    xrf = None

    EXTENSION_MAPPER = {
        ".npy": "NumPy Binary(*.npy)",
    }

pg.setConfigOption("antialias", True)


@dataclass
class SpectrumData:
    """Container for spectrum data."""

    x: np.ndarray
    y: np.ndarray


class FileManager:
    """Handles file operations for spectrum data."""

    EXTENSION_MAPPER = EXTENSION_MAPPER.copy()

    def __init__(self):
        self.current_path = Path.home()
        self.current_filter = list(EXTENSION_MAPPER.values())[0]

    def load_spectrum(self, file_path: Path) -> np.ndarray | None:
        """Load spectrum data from file."""
        ext = file_path.suffix

        if ext == ".npy":
            return np.load(file_path)

        if xrf is None:
            raise ModuleNotFoundError("xrflab not found")

        analyse = xrf.Analyse.from_file(file_path)
        condition = self._select_condition(analyse.data.keys())

        if condition:
            return analyse.data[condition].y

        return None

    def _select_condition(self, options) -> str | None:
        """Show dialog to select condition from available options."""
        try:
            sorted_options = sorted(options, key=lambda x: int(x.split(" ")[-1]))
        except (ValueError, IndexError):
            sorted_options = list(options)

        dialog = ConditionDialog(sorted_options)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_condition

        return None


class ConditionDialog(QDialog):
    """Dialog for selecting spectrum condition."""

    def __init__(self, options: list[str], parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Condition Selector")
        self._setup_ui(options)

    def _setup_ui(self, options: list[str]):
        layout = QVBoxLayout()

        # Selection row
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Select the condition you want to view:"))
        h_layout.addStretch()

        self.combo = QComboBox()
        self.combo.addItems(options)
        h_layout.addWidget(self.combo)
        layout.addLayout(h_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    @property
    def selected_condition(self) -> str:
        return self.combo.currentText()


class SpectrumGenerator:
    """Generates synthetic spectrum data."""

    def __init__(self, num_points: int = 2048):
        self.num_points = num_points

    def generate(self, seed: int | None = 42) -> SpectrumData:
        """Generate synthetic spectrum with noise and baseline."""
        if seed is not None:
            np.random.seed(seed)

        x = np.arange(self.num_points)

        # Create peaks
        spectrum = self._create_peaks(x, num_peaks=15)

        # Add baseline
        baseline = self._create_baseline(x)

        # Add noise
        noise = np.random.normal(0, 2, len(x))

        y = spectrum + baseline + noise
        return SpectrumData(x=x, y=y)

    def _create_peaks(self, x: np.ndarray, num_peaks: int) -> np.ndarray:
        """Create multiple Gaussian peaks."""
        spectrum = np.zeros_like(x, dtype=float)

        for _ in range(num_peaks):
            center = np.random.randint(100, len(x) - 100)
            height = np.random.uniform(50, 200)
            width = np.random.uniform(10, 40)
            spectrum += height * np.exp(-((x - center) ** 2) / (2 * width**2))

        return spectrum

    def _create_baseline(self, x: np.ndarray) -> np.ndarray:
        """Create exponential and polynomial baseline."""
        return 100 + 50 * np.exp(-x / 1000) + 0.01 * x


class BaselineProcessor:
    """Handles baseline correction operations."""

    # Baseline algorithm categories
    CATEGORIES = {
        "Polynomial": [
            "poly",
            "modpoly",
            "imodpoly",
            "penalized_poly",
            "quant_reg",
            "goldindec",
            "loess",
        ],
        "Whittaker": [
            "asls",
            "iasls",
            "airpls",
            "arpls",
            "drpls",
            "iarpls",
            "aspls",
            "psalsa",
            "derpsalsa",
            "mpls",
            "irsqr",
        ],
        "Morphological": [
            "mpls",
            "mor",
            "imor",
            "mormol",
            "amormol",
            "rolling_ball",
            "mwmv",
            "tophat",
            "mpspline",
            "jbcd",
        ],
        "Spline": [
            "mixture_model",
            "irsqr",
            "corner_cutting",
            "pspline_asls",
            "pspline_iasls",
            "pspline_airpls",
            "pspline_arpls",
            "pspline_drpls",
            "pspline_iarpls",
            "pspline_aspls",
            "pspline_psalsa",
            "pspline_derpsalsa",
            "pspline_mpls",
        ],
        "Smoothing": ["noise_median", "snip", "swima", "ipsa"],
        "Classification": [
            "dietrich",
            "golotvin",
            "std_distribution",
            "fastchrom",
            "cwt_br",
            "fabc",
            "rubberband",
            "rolling_std",
        ],
        "Miscellaneous": ["beads", "custom_bc", "interp_pts"],
    }

    SKIP_PARAMS = {"weights", "mask", "x_data", "z", "return_coef", "kwargs"}

    def __init__(self, x_data: np.ndarray):
        self.baseline_fitter = Baseline(x_data=x_data)

    def apply_correction(
        self, data: np.ndarray, algorithm: str, params: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply baseline correction with given algorithm and parameters."""
        method = getattr(self.baseline_fitter, algorithm)
        baseline, _ = method(data, **params)
        corrected = data - baseline
        return baseline, corrected

    def get_algorithm_parameters(self, algorithm: str) -> dict[str, Any]:
        """Get parameter information for an algorithm."""
        try:
            method = getattr(self.baseline_fitter, algorithm)
            sig = inspect.signature(method)

            params = {}
            for param_name, param in sig.parameters.items():
                if param_name == "data" or param_name in self.SKIP_PARAMS:
                    continue

                default = (
                    param.default if param.default != inspect.Parameter.empty else None
                )
                annotation = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                )

                params[param_name] = {
                    "default": default,
                    "annotation": annotation,
                    "type": type(default).__name__ if default is not None else "none",
                }

            return params

        except AttributeError:
            return {}


class NoiseReducer:
    """Handles noise reduction using Savitzky-Golay filter."""

    def apply(
        self,
        data: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0,
        axis: int = -1,
        mode: str = "interp",
        cval: float = 0.0,
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter to data."""
        return savgol_filter(
            data, window_length, polyorder, deriv, delta, axis, mode, cval
        )


class ParameterWidgetFactory:
    """Factory for creating parameter input widgets."""

    @staticmethod
    def create_widget(name: str, param_info: dict[str, Any], on_change_callback):
        """Create appropriate widget based on parameter info."""
        default = param_info["default"]
        param_type = param_info["type"]

        if param_type == "bool":
            return ParameterWidgetFactory._create_checkbox(default, on_change_callback)
        elif param_type == "int":
            return ParameterWidgetFactory._create_int_spinbox(
                default, on_change_callback
            )
        elif param_type == "float":
            return ParameterWidgetFactory._create_float_spinbox(
                name, default, on_change_callback
            )
        elif param_type == "none":
            return ParameterWidgetFactory._create_default_spinbox(
                name, on_change_callback
            )

        return None

    @staticmethod
    def _create_checkbox(default: bool, callback):
        widget = QCheckBox()
        widget.setChecked(default)
        widget.stateChanged.connect(callback)
        return widget

    @staticmethod
    def _create_int_spinbox(default: int, callback):
        widget = QSpinBox()
        widget.setRange(-1000000, 1000000)
        widget.setValue(default)
        widget.valueChanged.connect(callback)
        return widget

    @staticmethod
    def _create_float_spinbox(name: str, default: float, callback):
        widget = QDoubleSpinBox()
        widget.setRange(-1e10, 1e10)

        # Adjust precision based on value magnitude
        if abs(default) < 0.001 or abs(default) > 1000:
            widget.setDecimals(15)
            widget.setSingleStep(default / 10 if default != 0 else 0.0001)
        else:
            widget.setDecimals(10)
            widget.setSingleStep(0.01)

        widget.setValue(default)
        widget.valueChanged.connect(callback)
        return widget

    @staticmethod
    def _create_default_spinbox(name: str, callback):
        """Create spinbox for parameters with None default."""
        widget = QDoubleSpinBox()
        widget.setDecimals(10)

        # Special handling for common parameters
        if "lam" in name.lower() or "alpha" in name.lower():
            widget.setRange(0, 1e15)
            widget.setValue(1e6)
            widget.setSingleStep(1e5)
        else:
            widget.setRange(-1e10, 1e10)
            widget.setValue(0.0)
            widget.setSingleStep(0.01)

        widget.valueChanged.connect(callback)
        return widget


class PlotManager:
    """Manages plot visualization."""

    def __init__(self, plot_widget: pg.PlotWidget):
        self.plot_widget = plot_widget
        self._setup_plot()

    def _setup_plot(self):
        """Configure plot appearance."""
        self.plot_widget.setBackground("#FFFFFF")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setFrameShape(QFrame.Shape.NoFrame)

        plot_item = self.plot_widget.getPlotItem()
        assert plot_item
        plot_item.setContentsMargins(10, 10, 10, 10)
        plot_item.addLegend(
            offset=(-10, 10),
            pen=pg.mkPen(color="#E0E0E0", width=1),
            brush=pg.mkBrush(color="#F2F2F2"),
            labelTextColor="#000000",
        )

        self.plot_widget.setLabel("bottom", "Pixel")
        self.plot_widget.setLabel("left", "Intensity")

    def update_plot(
        self,
        x_data: np.ndarray,
        y_original: np.ndarray,
        baseline: np.ndarray | None = None,
        corrected: np.ndarray | None = None,
    ):
        """Update plot with data."""
        self.plot_widget.clear()

        # #1f77b4
        # #ff7f0e
        # #2ca02c

        # Plot original
        self.plot_widget.plot(
            x_data, y_original, pen=pg.mkPen(color="#1f77b4", width=2), name="Original"
        )

        # Plot baseline if available
        if baseline is not None:
            self.plot_widget.plot(
                x_data,
                baseline,
                pen=pg.mkPen(color="#2ca02c", width=2),
                name="Baseline",
            )

        # Plot corrected if available
        if corrected is not None:
            self.plot_widget.plot(
                x_data,
                corrected,
                pen=pg.mkPen(color="#ff7f0e", width=2),
                name="Corrected",
            )

        self.plot_widget.setTitle("Correction")

    def show_error(self, algorithm: str, error: Exception):
        """Display error message on plot."""
        self.plot_widget.clear()

        text = pg.TextItem(
            html=f'<p style="font-size:24px; color:red;">Error: {str(error)}</p>',
            anchor=(0.5, 0.5),
        )
        self.plot_widget.addItem(text)

        view_range = self.plot_widget.viewRange()
        x_center = (view_range[0][0] + view_range[0][1]) / 2
        y_center = (view_range[1][0] + view_range[1][1]) / 2
        text.setPos(x_center, y_center)

        self.plot_widget.setTitle(f"Error with {algorithm}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Corrections")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize components
        self.generator = SpectrumGenerator()
        self.file_manager = FileManager()

        # Generate initial data
        data = self.generator.generate()
        self.x_data = data.x
        self.y_data = data.y

        # Initialize processors
        self.baseline_processor = BaselineProcessor(self.x_data)
        self.noise_reducer = NoiseReducer()

        # State
        self.current_algorithm = None
        self.param_widgets = {}

        # Debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_plot)

        # Setup UI
        self._setup_ui()
        self._initialize_defaults()

    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create panels
        splitter.addWidget(self._create_control_panel())
        splitter.addWidget(self._create_plot_panel())
        splitter.setSizes([400, 1000])

        layout.addWidget(splitter)

    def _create_control_panel(self) -> QWidget:
        """Create control panel with all settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._create_baseline_group())
        layout.addWidget(self._create_noise_group())
        layout.addWidget(self._create_action_buttons())

        return panel

    def _create_baseline_group(self) -> QGroupBox:
        """Create baseline correction controls."""
        group = QGroupBox("Baseline Correction")
        layout = QVBoxLayout()

        h_layout = QHBoxLayout()

        # Enable checkbox
        self.baseline_enabled = QCheckBox("Enabled")
        self.baseline_enabled.setChecked(True)
        self.baseline_enabled.stateChanged.connect(self._schedule_update)
        h_layout.addWidget(self.baseline_enabled)

        open_docs_btn = QPushButton("Open Docs")
        open_docs_btn.clicked.connect(self._open_docs)
        h_layout.addWidget(open_docs_btn)

        layout.addLayout(h_layout)

        # Algorithm selection
        algo_group = QGroupBox("Algorithm Selection")
        algo_layout = QVBoxLayout()

        # Category
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(BaselineProcessor.CATEGORIES.keys())
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.category_combo)
        algo_layout.addLayout(cat_layout)

        # Algorithm
        alg_layout = QHBoxLayout()
        alg_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        alg_layout.addWidget(self.algorithm_combo)
        algo_layout.addLayout(alg_layout)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Parameters (scrollable)
        params_group = QGroupBox("Algorithm Parameters")
        params_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.params_widget = QWidget()
        self.params_layout = QFormLayout(self.params_widget)
        scroll.setWidget(self.params_widget)

        params_layout.addWidget(scroll)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        group.setLayout(layout)
        return group

    def _create_noise_group(self) -> QGroupBox:
        """Create noise reduction controls."""
        group = QGroupBox("Noise Reduction")
        layout = QVBoxLayout()

        self.noise_enabled = QCheckBox("Enabled")
        self.noise_enabled.stateChanged.connect(self._schedule_update)
        layout.addWidget(self.noise_enabled)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        # Create all noise reduction parameter widgets
        self.window_length = self._add_param_row(
            params_layout,
            "Window Length:",
            QSpinBox,
            (1, 1000, 5),
            "Filter window length",
        )

        self.polyorder = self._add_param_row(
            params_layout,
            "Polynomial Order:",
            QSpinBox,
            (0, 10, 2),
            "Polynomial order (must be < window length)",
        )

        self.deriv = self._add_param_row(
            params_layout,
            "Derivative Order:",
            QSpinBox,
            (0, 5, 0),
            "Derivative order (0 = no differentiation)",
        )

        self.delta = self._add_param_row(
            params_layout,
            "Delta:",
            QDoubleSpinBox,
            (0.001, 1000.0, 1.0, 3),
            "Sample spacing",
        )

        self.axis = self._add_param_row(
            params_layout, "Axis:", QSpinBox, (-10, 10, -1), "Array axis for filter"
        )

        self.mode = self._add_param_combo(
            params_layout,
            "Mode:",
            ["interp", "mirror", "constant", "nearest", "wrap"],
            "Extension mode",
        )

        self.cval = self._add_param_row(
            params_layout,
            "Constant Value:",
            QDoubleSpinBox,
            (-1000.0, 1000.0, 0.0, 3),
            "Edge value for 'constant' mode",
        )

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        group.setLayout(layout)

        return group

    def _add_param_row(self, layout, label, widget_class, params, tooltip):
        """Helper to add parameter row."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        widget = widget_class()
        if widget_class == QSpinBox:
            widget.setRange(params[0], params[1])
            widget.setValue(params[2])
        else:  # QDoubleSpinBox
            widget.setRange(params[0], params[1])
            widget.setValue(params[2])
            widget.setDecimals(params[3])

        widget.setToolTip(tooltip)
        widget.valueChanged.connect(self._schedule_update)

        row.addWidget(widget)
        row.addStretch()
        layout.addLayout(row)

        return widget

    def _add_param_combo(self, layout, label, items, tooltip):
        """Helper to add combo box parameter."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        combo = QComboBox()
        combo.addItems(items)
        combo.setToolTip(tooltip)
        combo.currentIndexChanged.connect(self._schedule_update)

        row.addWidget(combo)
        row.addStretch()
        layout.addLayout(row)

        return combo

    def _create_action_buttons(self) -> QWidget:
        """Create action buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        open_btn = QPushButton("Open Spectrum")
        open_btn.clicked.connect(self._open_spectrum)
        layout.addWidget(open_btn)

        gen_btn = QPushButton("Generate Random Spectrum")
        gen_btn.clicked.connect(self._generate_random_spectrum)
        layout.addWidget(gen_btn)

        return widget

    def _create_plot_panel(self) -> QWidget:
        """Create plot panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        plot_widget = pg.PlotWidget()
        self.plot_manager = PlotManager(plot_widget)
        layout.addWidget(plot_widget)

        return panel

    def _initialize_defaults(self):
        """Initialize with first algorithm."""
        first_category = list(BaselineProcessor.CATEGORIES.keys())[0]
        self.category_combo.setCurrentText(first_category)
        self._on_category_changed(first_category)

    def _on_category_changed(self, category: str):
        """Handle category selection change."""
        self.algorithm_combo.clear()
        algorithms = BaselineProcessor.CATEGORIES.get(category, [])
        self.algorithm_combo.addItems(algorithms)

        if algorithms:
            self._on_algorithm_changed(algorithms[0])

    def _on_algorithm_changed(self, algorithm: str):
        """Handle algorithm selection change."""
        self.current_algorithm = algorithm
        self._update_parameter_widgets()
        self._schedule_update()

    def _update_parameter_widgets(self):
        """Update parameter widgets for current algorithm."""
        # Clear existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item and (widget := item.widget()):
                widget.deleteLater()

        self.param_widgets.clear()

        if not self.current_algorithm:
            return

        # Get parameters
        params = self.baseline_processor.get_algorithm_parameters(
            self.current_algorithm
        )

        if not params:
            self.params_layout.addRow(
                QLabel(f"Algorithm '{self.current_algorithm}' not found")
            )
            return

        # Create widgets
        for param_name, param_info in params.items():
            widget = ParameterWidgetFactory.create_widget(
                param_name, param_info, self._schedule_update
            )
            if widget:
                self.params_layout.addRow(f"{param_name}:", widget)
                self.param_widgets[param_name] = widget

    def _open_docs(self):
        url = QUrl("https://pybaselines.readthedocs.io/en/latest/")
        QDesktopServices.openUrl(url)

    def _schedule_update(self):
        """Schedule plot update with debouncing."""
        self.update_timer.stop()
        self.update_timer.start(300)

    def _update_plot(self):
        """Update the plot with current settings."""
        if not self.current_algorithm:
            return

        try:
            baseline = None
            corrected = None

            # Apply baseline correction
            if self.baseline_enabled.isChecked():
                params = self._get_baseline_params()
                baseline, corrected = self.baseline_processor.apply_correction(
                    self.y_data, self.current_algorithm, params
                )

            # Apply noise reduction
            if self.noise_enabled.isChecked():
                data = corrected if corrected is not None else self.y_data
                corrected = self.noise_reducer.apply(
                    data,
                    window_length=self.window_length.value(),
                    polyorder=self.polyorder.value(),
                    deriv=self.deriv.value(),
                    delta=self.delta.value(),
                    axis=self.axis.value(),
                    mode=self.mode.currentText(),
                    cval=self.cval.value(),
                )

            # Update plot
            self.plot_manager.update_plot(self.x_data, self.y_data, baseline, corrected)

        except Exception as e:
            self.plot_manager.show_error(self.current_algorithm, e)

    def _get_baseline_params(self) -> dict[str, Any]:
        """Extract baseline parameters from widgets."""
        assert self.current_algorithm

        params = {}

        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()

                # Skip zero values for None defaults
                if value == 0.0 and isinstance(widget, QDoubleSpinBox):
                    method = getattr(
                        self.baseline_processor.baseline_fitter, self.current_algorithm
                    )
                    sig = inspect.signature(method)
                    if param_name in sig.parameters:
                        default = sig.parameters[param_name].default
                        if default is None:
                            continue

                params[param_name] = value

        return params

    def _generate_random_spectrum(self):
        """Regenerate spectrum with new random seed."""
        data = self.generator.generate(seed=None)
        self.y_data = data.y
        self._schedule_update()

    def _open_spectrum(self):
        """Open spectrum from file."""
        file_name, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            directory=self.file_manager.current_path.as_posix(),
            filter=";;".join(FileManager.EXTENSION_MAPPER.values()),
            initialFilter=self.file_manager.current_filter,
        )

        if not file_name:
            return

        path = Path(file_name)
        self.file_manager.current_path = path
        self.file_manager.current_filter = FileManager.EXTENSION_MAPPER[path.suffix]

        y_data = self.file_manager.load_spectrum(path)
        if y_data is not None:
            self.y_data = y_data
            self._update_plot()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
