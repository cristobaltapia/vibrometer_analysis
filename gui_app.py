import queue
import random
import re
import sys
from time import sleep

import matplotlib
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRunnable, Qt, QThreadPool, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
                             QSlider, QTableView, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from vibrometer import DEV_NAME, SignalAnalysis

matplotlib.use('Qt5Agg')

mapping = [c - 1 for c in [1]]


class Window(QMainWindow):
    """Docstring for Window. """
    restart_stream = QtCore.pyqtSignal(object)
    device_reload = QtCore.pyqtSignal(object)
    sig_unlock = QtCore.pyqtSignal(object)

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()
        self.threadpool = QThreadPool()
        self.q = queue.Queue()
        self.downsample = 20
        self.preview_time = 5

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
        # form_board.addStretch()

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
        # VELO
        self.cbox_vel = QComboBox()
        self.cbox_vel.setFixedWidth(80)
        self.cbox_vel.addItem("20")
        self.cbox_vel.addItem("100")
        self.cbox_vel.addItem("500")
        form_device.addRow(QLabel("VELO (mm/s):"), self.cbox_vel)
        # Set default value to "100"
        self.cbox_vel.setCurrentIndex(1)

        #############################################################
        # Duration
        # group_velo.setLayout(form_time)
        self.rec_time = QLineEdit("0.2")
        self.rec_time.setFixedWidth(80)
        self.trigger = QLineEdit("0.02")
        self.trigger.setFixedWidth(80)

        # Set validators
        self.only_double = QtGui.QDoubleValidator()
        self.only_double.setBottom(0)
        self.rec_time.setValidator(self.only_double)
        self.trigger.setValidator(self.only_double)
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

        #############################################################
        # Board properties
        self.board_w = QLineEdit("100")
        self.board_t = QLineEdit("30")
        self.board_l = QLineEdit("1000")
        self.board_kg = QLineEdit("500")
        self.board_w.setFixedWidth(100)
        self.board_t.setFixedWidth(100)
        self.board_l.setFixedWidth(100)
        self.board_kg.setFixedWidth(100)
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
        default_min = 100
        default_max = 4000
        self.min_freq = QLineEdit(f"{default_min}")
        self.freq_min_slide.setValue(int(default_min * 99.0 / 20000.0))
        self.max_freq = QLineEdit(f"{default_max}")
        self.freq_max_slide.setValue(int(default_max * 99.0 / 20000.0))

        self.freq_min_slide.sliderMoved[int].connect(self.update_min_freq)
        self.freq_max_slide.sliderMoved[int].connect(self.update_max_freq)
        self.min_freq.textEdited.connect(self.update_min_freq_val)
        self.max_freq.textEdited.connect(self.update_max_freq_val)
        self.freq_max_slide.setInvertedAppearance(True)

        # Set validators
        self.only_ints = QtGui.QIntValidator(bottom=0, top=20000)
        self.min_freq.setValidator(self.only_ints)
        self.max_freq.setValidator(self.only_ints)

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
        # self.results.setModel(self.data_results)
        # header = self.results.horizontalHeader()
        # header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        # header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        results_layout.addWidget(self.results)

        #############################################################
        # Matplotlib
        self.canvas = MplCanvas(self.main_widget, width=5, height=2, dpi=100)
        self.canvas_f = MplCanvas(self.main_widget, width=5, height=2, dpi=100)

        self.init_canvas()

        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.canvas_f)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.init_stream()

        self.restart_stream.connect(self.init_stream)
        self.device_reload.connect(self.reload_device)
        self.sig_unlock.connect(self.unlock_input)

        self.show()

    def init_stream(self):
        dev_sel = self.cbox_dev.currentText()
        ix_sel = self.devs.index(dev_sel)
        dev_num = self.devs_ix[ix_sel]
        dev_rate = int(self.devs_rate[ix_sel])
        fs = int(dev_rate)
        self.mic = MicrophoneCapture(dev_num, rate=fs, downsample=self.downsample)

    def start_stream(self):
        self.mic.start_stream()

    def stop_stream(self):
        self.mic.stop_stream()

    def close_stream(self):
        self.mic.close_stream()

    def reload_device(self):
        self.close_stream()
        self.init_stream()

    def _reload_device(self):
        self.device_reload.emit("Reload device")

    def update_min_freq(self, val):
        freq = val * 20000.0 / 99.0
        self.min_freq.setText(f"{freq:1.0f}")

    def update_max_freq(self, val):
        freq = 20000 - val * 20000.0 / 99.0
        self.max_freq.setText(f"{freq:1.0f}")

    def update_min_freq_val(self, val):
        try:
            freq = float(val) * 99.0 / 20000.0
            self.freq_min_slide.setValue(freq)
        except:
            pass

    def update_max_freq_val(self, val):
        try:
            freq = (20000 - float(val)) * (99.0) / 20000.0
            self.freq_max_slide.setValue(freq)
        except:
            pass

    def listen_for_signal(self):
        self.start.setEnabled(False)
        self.lock_input()
        self.preview.setEnabled(False)
        worker = Worker(self._listen_for_signal)
        self.threadpool.start(worker)

    def _listen_for_signal(self):
        # Close active stream
        self.close_stream()
        dev_sel = self.cbox_dev.currentText()
        ix_sel = self.devs.index(dev_sel)
        dev_num = self.devs_ix[ix_sel]
        dev_rate = self.devs_rate[ix_sel]

        rec_time = float(self.rec_time.text())

        self.statusBar().showMessage('Waiting for impulse...')
        thress = float(self.trigger.text())
        velo = float(self.cbox_vel.currentText())

        vib_analysis = SignalAnalysis(device=dev_num, sample_rate=dev_rate, velo=velo)
        # Record signal after impulse
        vib_analysis.wait_and_record(duration=rec_time, total_recording=10, thress=thress)

        max_freq = int(self.max_freq.text())
        min_freq = int(self.min_freq.text())

        freq = vib_analysis.compute_frequencies(min_freq=min_freq, max_freq=max_freq)

        l = float(self.board_l.text())
        w = float(self.board_w.text())
        t = float(self.board_t.text())
        kg = float(self.board_kg.text())

        moes = vib_analysis.calc_moe(length=l, width=w, thick=t, weight=kg)

        moes = np.round(moes, 0)
        freq = np.round(freq, 0)

        data_results = pd.DataFrame({"Freq. [Hz]": freq, "E_dyn [MPa]": moes})
        self.data_results = TableModel(data_results)
        self.results.setModel(self.data_results)

        vib_analysis.make_plot_gui(self.canvas.axes, self.canvas_f.axes)
        # gui.status = "Idle..."
        self.statusBar().showMessage('Ready')
        self.preview.setEnabled(True)
        self.sig_unlock.emit("Unlock")
        self.restart_stream.emit("Restart")
        self.start.setEnabled(True)

    def start_live_preview(self):
        """Start live plotting of signal."""
        self.preview.setEnabled(False)
        self.start.setEnabled(False)
        self.cbox_dev.setEnabled(False)
        self.preview_stop.setEnabled(True)

        self.init_canvas()
        # Get selected device
        dev_sel = self.cbox_dev.currentText()
        # Get index of the device
        ix_sel = self.devs.index(dev_sel)
        dev_rate = int(self.devs_rate[ix_sel])

        self.statusBar().showMessage('Preview...')
        fs = int(dev_rate)
        rec_time = self.preview_time

        # Plot
        length = int(rec_time * fs / (self.downsample))
        self.live_data = np.zeros((length, 1))
        self.time = np.arange(start=0, step=float(self.downsample) / float(fs),
                              stop=rec_time)

        ax = self.canvas.axes
        self.lines = ax.plot(self.time, self.live_data, color="C0", lw=0.7)
        ax.set_xlim(left=0, right=self.preview_time)

        ax.grid(True)

        self.timer_prev = QtCore.QTimer()
        self.timer_prev.timeout.connect(self.update_live_preview)

        self.start_stream()
        self.timer_prev.start(50)

    def stop_live_preview(self):
        self.timer_prev.setSingleShot(True)
        self.timer_prev.stop()
        self.timer_prev.deleteLater()
        self.init_canvas()
        self.stop_stream()
        self.close_stream()
        # Change status of buttons
        self.preview.setEnabled(True)
        self.preview_stop.setEnabled(False)
        self.start.setEnabled(True)
        self.cbox_dev.setEnabled(True)
        self.statusBar().showMessage('Ready...')
        self.restart_stream.emit("Restart")

    def update_live_preview(self):
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

        self.canvas.draw()

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

    def init_canvas(self):
        """
        Initialize plots.
        """
        ax1 = self.canvas.axes
        ax2 = self.canvas_f.axes

        ax1.cla()
        ax2.cla()

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [mm/s]")

        ax1.set_xlim(left=0, right=self.preview_time)
        ax1.grid(True)

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD")

        self.canvas.figure.tight_layout()
        self.canvas_f.figure.tight_layout()

        self.canvas.draw()
        self.canvas_f.draw()

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
        self.setColumnWidth(1, width * 0.5)
        self.setColumnWidth(2, width * 0.5)


class MicrophoneCapture:
    """Manages the stream input.

    Parameters
    ----------
    device : TODO
    rate : TODO

    """
    def __init__(self, device, rate, downsample):
        """TODO: to be defined.



        """
        self.device = device
        self.rate = rate
        self.q = queue.Queue()
        self.downsample = downsample

        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=self.rate,
                                     callback=self.audio_callback)

    def start_stream(self):
        self.stream.start()

    def stop_stream(self):
        self.stream.stop()

    def close_stream(self):
        self.stream.close()

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        # self.signal[:] = indata[::self.downsample, 0]
        self.q.put(indata[::self.downsample, mapping])


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

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    app.exec_()
