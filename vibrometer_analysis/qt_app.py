import queue
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRunnable, Qt, QThreadPool, QTimer, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QProgressBar, QPushButton, QSizePolicy, QSlider,
                             QTableView, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from vibrometer_analysis.vibrometer import (DEV_NAME, SignalAnalysis, VibrometerCapture)

mpl.use('Qt5Agg')


class Window(QMainWindow):
    """Docstring for Window. """
    sig_stream_restart = QtCore.pyqtSignal(object)
    sig_device_reload = QtCore.pyqtSignal(object)
    sig_unlock = QtCore.pyqtSignal(object)
    sig_device_velo = QtCore.pyqtSignal(object)

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()
        self.threadpool = QThreadPool()
        self.q = queue.Queue()
        self.downsample = 20
        self.preview_time = 5
        self.wait_time = 10
        self.dev_velo = None

        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('Ready')
        # self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Vibrometer analysis Pro-30000')
        self.main_widget = QWidget(self)

        #############################################################
        # Layout
        main_layout = QVBoxLayout(self.main_widget)
        self.setLayout(main_layout)

        input_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)

        left_layout = QVBoxLayout()
        input_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        input_layout.addLayout(right_layout)

        # Layout for controls
        group_device = QGroupBox("VIB-E-220")
        layout_device = QVBoxLayout()
        form_device = QFormLayout()
        layout_buttons = QHBoxLayout()
        layout_dev = QVBoxLayout()
        group_device.setLayout(layout_device)

        layout_device.addLayout(form_device)
        layout_device.addLayout(layout_dev)
        layout_device.addLayout(layout_buttons)

        # Layout for board properties
        group_board = QGroupBox("Board properties")
        form_board = QFormLayout()
        group_board.setLayout(form_board)

        # Layout for frequency region
        group_freq = QGroupBox("Frequency range")
        grid_freq = QGridLayout()
        group_freq.setLayout(grid_freq)

        # Layout for results
        group_results = QGroupBox("Results")
        results_layout = QVBoxLayout()
        group_results.setLayout(results_layout)

        # Add layouts
        left_layout.addWidget(group_device)
        left_layout.addWidget(group_freq)

        right_layout.addWidget(group_board)
        right_layout.addWidget(group_results)

        #############################################################
        # Volicity
        self.cbox_vel = QComboBox()
        self.cbox_vel.setFixedWidth(80)
        self.cbox_vel.addItem("20")
        self.cbox_vel.addItem("100")
        self.cbox_vel.addItem("500")
        form_device.addRow(QLabel("VELO (mm/s):"), self.cbox_vel)
        # Set default value to "100"
        self.cbox_vel.setCurrentIndex(1)
        self.dev_velo = 100.0
        self.cbox_vel.currentIndexChanged.connect(self.update_device_velo)

        #############################################################
        # Duration
        self.rec_time = QDoubleSpinBox()
        self.rec_time.setMinimum(0.01)
        self.rec_time.setDecimals(2)
        self.rec_time.setSingleStep(0.1)
        self.rec_time.setValue(0.2)
        self.rec_time.setFixedWidth(80)
        self.trigger = QDoubleSpinBox()
        self.trigger.setMinimum(1e-8)
        self.trigger.setValue(0.10)
        self.trigger.setDecimals(5)
        self.trigger.setSingleStep(0.01)
        self.trigger.setFixedWidth(80)

        # Set validators
        self.only_double = QtGui.QDoubleValidator()
        self.only_double.setBottom(0)
        form_device.addRow(QLabel("Recording time (s):"), self.rec_time)
        form_device.addRow(QLabel("Trigger sensitivity (mm/s):"), self.trigger)

        # Get list of sound devices
        self.devs = []
        self.devs_ix = []
        self.devs_rate = []
        for ix, dev in enumerate(sd.query_devices()):
            if dev["default_low_input_latency"] != -1:
                self.devs.append(dev["name"])
                self.devs_ix.append(ix)
                self.devs_rate.append(dev["default_samplerate"])

        self.cbox_dev = QComboBox()
        for dev_i in self.devs:
            self.cbox_dev.addItem(dev_i)

        self.cbox_dev.currentIndexChanged.connect(self._reload_device)

        self.cbox_dev.setFixedWidth(300)
        # form_time.addRow(QLabel("Device:"), self.cbox_dev)
        layout_dev.addWidget(QLabel("Device:"))
        layout_dev.addWidget(self.cbox_dev)

        self.start = QPushButton("Start")
        self.start.clicked.connect(self.listen_for_signal)
        self.start.setStyleSheet("background-color: rgb(44, 160, 44);")
        self.preview = QPushButton("Preview")
        self.preview.clicked.connect(self.start_live_preview)
        self.preview_stop = QPushButton("Stop")
        self.preview_stop.clicked.connect(self.stop_live_preview)
        layout_buttons.addWidget(self.start)
        layout_buttons.addWidget(self.preview)
        layout_buttons.addWidget(self.preview_stop)
        self.preview_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setMaximum(self.wait_time)
        self.progress.setGeometry(0, 0, 300, 15)
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size_policy.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(size_policy)
        self.progress.setVisible(False)
        layout_device.addWidget(self.progress)

        #############################################################
        # Board properties
        self.board_w = QLineEdit("100")
        self.board_t = QLineEdit("30")
        self.board_l = QLineEdit("1000")
        self.board_kg = QLineEdit("500")
        self.board_w.setFixedWidth(70)
        self.board_t.setFixedWidth(70)
        self.board_l.setFixedWidth(70)
        self.board_kg.setFixedWidth(70)
        form_board.addRow(QLabel("Width (mm):"), self.board_w)
        form_board.addRow(QLabel("Thickness (mm):"), self.board_t)
        form_board.addRow(QLabel("Length (mm):"), self.board_l)
        form_board.addRow(QLabel("Weight (g):"), self.board_kg)
        # Set validators
        self.board_w.setValidator(self.only_double)
        self.board_t.setValidator(self.only_double)
        self.board_l.setValidator(self.only_double)
        self.board_kg.setValidator(self.only_double)

        #############################################################
        # Frequency region
        self.freq_min_slide = QSlider(QtCore.Qt.Horizontal)
        self.freq_max_slide = QSlider(QtCore.Qt.Horizontal)
        self.freq_max_slide.setInvertedAppearance(True)
        default_min = 100
        default_max = 4000
        self.min_freq = QDoubleSpinBox()
        self.min_freq.setFixedWidth(80)
        self.min_freq.setDecimals(0)
        self.min_freq.setRange(0, 20000)
        self.min_freq.setSingleStep(50)

        self.max_freq = QDoubleSpinBox()
        self.max_freq.setFixedWidth(80)
        self.max_freq.setDecimals(0)
        self.max_freq.setRange(0, 20000)
        self.max_freq.setSingleStep(50)

        self.freq_min_slide.sliderMoved[int].connect(self.update_min_freq)
        self.freq_max_slide.sliderMoved[int].connect(self.update_max_freq)
        self.min_freq.valueChanged.connect(self.update_min_freq_slider)
        self.min_freq.textChanged.connect(self.update_min_freq_region)
        self.max_freq.valueChanged.connect(self.update_max_freq_slider)
        self.max_freq.textChanged.connect(self.update_max_freq_region)
        self.freq_min_slide.sliderReleased.connect(self.update_min_freq_region)
        self.freq_max_slide.sliderReleased.connect(self.update_max_freq_region)

        # Set validators
        grid_freq.addWidget(QLabel("Min freq. (Hz):"), 0, 0)
        grid_freq.addWidget(QLabel("Max freq. (Hz):"), 2, 0)
        grid_freq.addWidget(self.min_freq, 0, 1)
        grid_freq.addWidget(self.max_freq, 2, 1)
        grid_freq.addWidget(self.freq_min_slide, 1, 0, 1, 2)
        grid_freq.addWidget(self.freq_max_slide, 3, 0, 1, 2)

        #############################################################
        # Results
        data_results = pd.DataFrame({"Freq. [Hz]": [0], "E_dyn [MPa]": [0]})
        self.data_results = TableModel(data_results)
        self.results = TableResults(self.data_results)
        results_layout.addWidget(self.results)

        #############################################################
        # Matplotlib

        self.canvas = SignalTimePlot(self.main_widget, width=5, height=2, dpi=100,
                                     prev_time=self.preview_time)
        self.canvas_f = FrequencyPlot(self.main_widget, width=5, height=2, dpi=100)

        self.init_stream()

        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.canvas_f)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Signal handling
        self.sig_stream_restart.connect(self.init_stream)
        self.sig_device_reload.connect(self.reload_device)
        self.sig_unlock.connect(self.unlock_input)
        self.sig_device_velo.connect(self.update_device_velo)

        #############################################################
        # Set some default values
        self.freq_min_slide.setValue(int(default_min * 99.0 / 20000.0))
        self.min_freq.setValue(default_min)
        self.freq_max_slide.setValue(int(99.0 - default_max * 99.0 / 20000.0))
        self.max_freq.setValue(default_max)

        self.show()

    def init_stream(self):
        dev_sel = self.cbox_dev.currentText()
        ix_sel = self.devs.index(dev_sel)
        dev_num = self.devs_ix[ix_sel]
        dev_rate = int(self.devs_rate[ix_sel])
        self.dev_rate = dev_rate
        fs = int(dev_rate)
        dev_velo = self.dev_velo

        self.mic = VibrometerCapture(dev_num, rate=fs, velo=dev_velo,
                                     downsample=self.downsample)
        self.canvas.set_mic(self.mic)

    def reload_device(self):
        self.mic.close_stream()
        self.init_stream()

    def _reload_device(self):
        self.sig_device_reload.emit("Reload device")

    def update_device_velo(self):
        """Update the valocity configured in the device."""
        self.dev_velo = float(self.cbox_vel.currentText())
        self.mic.close_stream()
        self.init_stream()

    def update_min_freq(self, val):
        freq = int(val * 20000.0 / 99.0)
        self.min_freq.setValue(freq)

    def update_max_freq(self, val):
        freq = 20000 - val * 20000.0 / 99.0
        self.max_freq.setValue(freq)

    def update_min_freq_slider(self, val):
        slider = int(val * 99.0 / 20000.0)
        max_slider = 99.0 - self.freq_max_slide.value()
        if slider >= max_slider:
            self.freq_min_slide.setValue(max_slider)
        else:
            self.min_freq.valueChanged.disconnect()
            self.freq_min_slide.setValue(slider)
            self.min_freq.valueChanged.connect(self.update_min_freq_slider)

    def update_max_freq_slider(self, val):
        slider = int((20000 - float(val)) * (99.0) / 20000.0)
        min_slider = 99.0 - self.freq_min_slide.value()
        if slider >= min_slider:
            self.freq_max_slide.setValue(min_slider)
        else:
            self.max_freq.valueChanged.disconnect()
            self.freq_max_slide.setValue(slider)
            self.max_freq.valueChanged.connect(self.update_max_freq_slider)

    def update_min_freq_region(self):
        """Update the x-axis of the matplotlib canvas."""
        if not self.freq_min_slide.isSliderDown():
            slider = self.min_freq.value()
            self.canvas_f.min_f = slider
            self.canvas_f.axes.set_xlim(left=slider)
            self.canvas_f.draw()

    def update_max_freq_region(self):
        """Update the x-axis of the matplotlib canvas."""
        if not self.freq_max_slide.isSliderDown():
            slider = self.max_freq.value()
            self.canvas_f.max_f = slider
            self.canvas_f.axes.set_xlim(right=slider)
            self.canvas_f.draw()

    def listen_for_signal(self):
        self.start.setEnabled(False)
        self.lock_input()
        self.preview.setEnabled(False)
        worker = Worker(self._listen_for_signal)
        self.threadpool.start(worker)

    def _listen_for_signal(self):
        # Close active stream
        self.mic.close_stream()
        dev_sel = self.cbox_dev.currentText()
        ix_sel = self.devs.index(dev_sel)
        dev_num = self.devs_ix[ix_sel]
        dev_rate = self.devs_rate[ix_sel]

        rec_time = self.rec_time.value()

        self.statusBar().showMessage('Waiting for impulse...')
        thress = self.trigger.value()
        velo = float(self.cbox_vel.currentText())

        vib_analysis = SignalAnalysis(device=dev_num, sample_rate=dev_rate, velo=velo)
        # Record signal after impulse
        vib_analysis.wait_and_record(duration=rec_time, total_recording=self.wait_time,
                                     thress=thress, progress=self.progress)

        max_freq = int(self.max_freq.value())
        min_freq = int(self.min_freq.value())

        freq = vib_analysis.compute_frequencies(min_freq=min_freq, max_freq=max_freq)

        l = float(self.board_l.text())
        w = float(self.board_w.text())
        t = float(self.board_t.text())
        kg = float(self.board_kg.text())

        moes = vib_analysis.calc_moe(length=l, width=w, thick=t, weight=kg)

        moes = np.round(moes, 0)
        freq = np.round(freq, 0)

        data_results = pd.DataFrame({"Freq. [Hz]": freq, "E_dyn [MPa]": moes}, dtype=int)
        self.data_results = TableModel(data_results)
        self.results.setModel(self.data_results)

        vib_analysis.make_plot_gui(self.canvas.axes, self.canvas_f.axes)

        # Send signals
        self.sig_stream_restart.emit("Restart")
        self.sig_unlock.emit("Unlock")
        self.preview.setEnabled(True)
        self.start.setEnabled(True)

        self.statusBar().showMessage('Ready')

    def start_live_preview(self):
        """Start live plotting of signal."""
        self.preview.setEnabled(False)
        self.start.setEnabled(False)
        self.cbox_dev.setEnabled(False)
        self.cbox_vel.setEnabled(False)
        self.preview_stop.setEnabled(True)
        self.statusBar().showMessage('Preview...')

        rate = int(self.dev_rate)

        self.canvas_f.compute_initial_figure()
        self.canvas.live_preview(rate, downsample=self.downsample, interval=50)

    def stop_live_preview(self):
        self.canvas.stop_live_preview()
        # self.init_canvas()
        self.mic.stop_stream()
        self.mic.close_stream()
        # Change status of buttons
        self.preview.setEnabled(True)
        self.preview_stop.setEnabled(False)
        self.start.setEnabled(True)
        self.cbox_dev.setEnabled(True)
        self.cbox_vel.setEnabled(True)
        self.statusBar().showMessage('Ready...')
        self.sig_stream_restart.emit("Restart")

    def lock_input(self):
        """Lock all the input fields."""
        input_fields = [
            self.board_l,
            self.board_w,
            self.board_t,
            self.board_kg,
        ]

        for field in input_fields:
            field.setStyleSheet("color: rgb(150, 150, 150);")
            field.setReadOnly(True)

        self.cbox_dev.setEnabled(False)

    def unlock_input(self):
        """Unlock input fields."""
        input_fields = [
            self.board_l,
            self.board_w,
            self.board_t,
            self.board_kg,
        ]

        for field in input_fields:
            field.setStyleSheet("color: rgb(0, 0, 0);")
            field.setReadOnly(False)

        self.cbox_dev.setEnabled(True)

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        # self.signal[:] = indata[::self.downsample, 0]
        self.q.put(indata[::self.downsample, mapping])


class TableResults(QTableView):

    def __init__(self, model, parent=None):
        super(TableResults, self).__init__(parent)

        rowHeight = self.fontMetrics().height()
        self.verticalHeader().setDefaultSectionSize(rowHeight)
        self.setModel(model)

    def resizeEvent(self, event):
        width = event.size().width()
        self.setColumnWidth(1, int(width * 0.5))
        self.setColumnWidth(2, int(width * 0.5))


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class Worker(QRunnable):
    """
    Worker Thread.

    Used to not block the GUI while aquiring data.

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.fn()


class MplCanvas(FigureCanvas, FuncAnimation):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        pass


class SignalTimePlot(MplCanvas):
    """Canvas used to plot the signal in the time domain.

    Parameters
    ----------
    parent : TODO
    width : TODO
    height : TODO
    dpi : TODO

    """

    def __init__(self, parent=None, width=5, height=4, dpi=100, prev_time=5):
        self.mic = None
        self.preview_time = prev_time
        super(SignalTimePlot, self).__init__(parent, width, height, dpi)

    def set_mic(self, mic):
        self.mic = mic

    def compute_initial_figure(self):
        ax1 = self.axes
        ax1.clear()

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [mm/s]")

        ax1.set_xlim(left=0, right=self.preview_time)
        ax1.grid(True)

        self.figure.tight_layout()
        self.draw()

    def live_preview(self, rate, downsample, interval):
        """TODO: Docstring for live_preview.

        Parameters
        ----------
        time : TODO
        rate : TODO
        interval : TODO

        Returns
        -------
        TODO

        """
        self.compute_initial_figure()
        show_time = self.preview_time
        length = int(show_time * rate / (downsample))

        self.live_data = np.zeros((length, 1))
        self.time = np.arange(start=0, step=float(downsample) / float(rate), stop=show_time)

        ax = self.axes
        self.draw()
        self.lines = ax.plot(self.time, self.live_data, color="C0", lw=0.7)
        ax.set_xlim(left=0, right=show_time)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
        ax.grid(True)

        self.ani = FuncAnimation(self.figure, self.update_live_preview, interval=interval,
                                 blit=True, repeat=False)

        self.mic.start_stream()

    def update_live_preview(self, frame):
        while True:
            try:
                data = self.mic.q.get_nowait()
            except queue.Empty:
                break

            shift = len(data)
            self.live_data = np.roll(self.live_data, -shift, axis=0)
            self.live_data[-shift:, :] = data

        for column, line in enumerate(self.lines):
            line.set_ydata(self.live_data[:, column])

        return self.lines

    def stop_live_preview(self):
        self.ani.event_source.stop()
        self.compute_initial_figure()


class FrequencyPlot(MplCanvas):
    """Canvas for the frequency domain of the signal.

    Parameters
    ----------
    parent : TODO
    width : TODO
    height : TODO
    dpi : TODO

    """

    def __init__(self, parent, width, height, dpi, min_f=0, max_f=20000):
        """TODO: to be defined.


        """
        self.min_f = min_f
        self.max_f = max_f

        super(FrequencyPlot, self).__init__(parent, width, height, dpi)

    def compute_initial_figure(self):
        """
        Initialize plots.
        """
        ax = self.axes
        ax.cla()

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")

        ax.set_xlim(left=self.min_f, right=self.max_f)

        self.figure.tight_layout()
        self.draw()


def main():
    app = QApplication([])
    window = Window()
    app.exec_()


if __name__ == "__main__":
    main()
