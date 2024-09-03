import queue
import re
import sys
from time import sleep, time

from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from PyQt6.QtWidgets import QProgressBar

import matplotlib as mpl  # isort: skip
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

# Add the following befor importing pyplot:
rc_params_gruyter = {
    "figure.figsize": (2.8, 2.0),
    # "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 6,
    "legend.fontsize": 5,
    "axes.labelsize": 6,
    # "lines.linewidth": 1.0,
    # "lines.markersize": 5,
    # "pgf.rcfonts": False,
}
mpl.rcParams.update(rc_params_gruyter)

mapping = [c - 1 for c in [1]]

# Board dimensions
W = 100.0
T = 20.0
L = 1000.0
WEIGHT = 300.0
RHO = WEIGHT / (L * T * W)

# Vibrometer parameters
VELO = 100.0
# Thresshold for detection of impulse
THRESSHOLD = 1e-3 * VELO / 4.0
# Recording time after impulse detection
REC_TIME = 0.4
# Name of the device to use
DEV_NAME = "default"


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class SignalAnalysis:
    """A class used to analyse the data from the vibrometer.

    Parameters
    ----------
    device : int
        Sound device used to listen for the signal.
    sample_rate : int
        Sample rate used capture the signal.
    velo : float
        Maximum velocity captured by the device.

    """

    _data: ArrayLike
    _time: ArrayLike
    _freq: NDArray[np.int64]
    _n_points: int
    _psd: NDArray[np.float64]
    _peaks_ix: NDArray[np.int64]

    def __init__(self, device: int, sample_rate: float, velo: float):
        self._device = device
        self._sample_rate = int(sample_rate)
        self._velo = velo
        self._spacing = 1.0 / sample_rate
        self.prev_sig_ix = 0

    def wait_and_record(
        self,
        duration: float,
        total_recording: int,
        thress: float,
        progress: QProgressBar | None = None,
    ):
        """Start recording and return the signal after an impulse is detected.

        Parameters
        ----------
        duration : float
            Recording time (used for the analysis)
        total_recording : float
            Total recording time
        thress : float
            Thresshold of the signal to be used to mark as the start of the measurement
        progress : bool
            Whether progress information should be displayed in the terminal

        Returns
        -------
        impact_detected : bool
            Whether an impact was detacted

        """
        total_recording = int(total_recording)

        rate = int(self._sample_rate)

        length = int((total_recording + 1) * rate)
        self.live_data = np.zeros((length, 1))

        # Initialize a Vibrometer Capture object
        self.vibro = VibrometerCapture(
            device=self._device, rate=rate, velo=self._velo, downsample=1
        )
        self.vibro.start_stream()

        # Show the progress bar if we are running the QT app
        if progress:
            print("------------------")
            print("Recording...")
            print("Waiting for impulse...")
            progress.setVisible(True)

        extra_time = 0.005

        t_init = time()
        impact_detected = False

        # Wait until an impulse is detected
        while True:
            self.update_signal()
            if progress:
                progress.setValue(int(time() - t_init))

            ix = np.argmax(np.abs(self.live_data[:, 0]))
            val = np.abs(self.live_data[ix, 0])

            if val > thress:
                print("Impulse detected!...")
                impact_detected = True
                play_detected()
                break
            if time() - t_init > total_recording:
                break
            sleep(0.01)

        ix = ix + int(rate * extra_time)
        sleep(duration + extra_time)

        self.vibro.stop_stream()
        self.vibro.close_stream()
        # One last call to update the signal results
        self.update_signal()

        print("Stop recording...")

        data = self.live_data[ix : ix + int(rate * duration) :, 0]
        data = data.reshape((-1,))
        # Remove mean
        data = data - np.mean(data)
        time_ = np.arange(start=0, step=1.0 / rate, stop=len(data) / rate)

        self._data = data
        self._time = time_
        self._n_points = len(data)

        if progress:
            # Reset progress bar
            progress.setValue(0)
            progress.setVisible(False)

        return impact_detected

    def compute_frequencies(self, min_freq: int = 0, max_freq: int = 20000):
        """Compute eigen frequencies from the signal.

        Parameters
        ----------
        min_freq : int
            Minimum frequency to consider for the analysis.
        max_freq : int
            Maximum frequency to consider for the analysis.

        Returns
        -------

        freq : ArrayLike
            Frequencies detected. These consider the frequencies with a PSD larger
            than 10% of the maximum detected PSD.

        """
        n_points = self._n_points
        data = self._data
        spacing = self._spacing

        # Apply window to reduce noise
        window: ArrayLike = hamming(n_points)

        # Apply FFT
        psd_ = fft(data * window)[0 : n_points // 2]
        freq = fftfreq(n_points, d=spacing)[0 : n_points // 2]

        psd_ = np.real(psd_)
        psd = 1.0 / n_points * np.abs(psd_)

        # Narrow the frequency space to the needed one
        ix_min = np.argmin(np.abs(freq - min_freq))
        ix_max = np.argmin(np.abs(freq - max_freq))

        psd = psd[ix_min : ix_max + 1]
        freq = freq[ix_min : ix_max + 1]

        # Obtain the three most relevant frequencies
        self.psd_max_limit: float = float(np.max(psd)) * 0.3
        peaks_ix, _ = find_peaks(psd, distance=100, height=self.psd_max_limit)

        # Sort peaks
        p_sort = np.argsort(psd[peaks_ix])[::-1]
        peaks_ix_sort = peaks_ix[p_sort]

        print("Frequencies detected:")

        self._psd = psd
        self._freq = freq
        self._peaks_ix = peaks_ix_sort

        return freq[peaks_ix_sort]

    def make_plot(self):
        v = self._data
        t = self._time
        freq = self._freq
        psd = self._psd
        peaks_ix = self._peaks_ix

        fig = plt.figure(figsize=(5.8, 6.0))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(t, v)

        ax2.plot(freq, psd)
        ax2.scatter(freq[peaks_ix], psd[peaks_ix], marker="o", color="r")

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [mm/s]")

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Power spectrum")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

        for pi in peaks_ix:
            x_i = freq[pi]
            y_i = psd[pi]
            ax2.annotate(
                f"{x_i} Hz",
                xy=(x_i, y_i),
                xytext=(20, -10),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "connectionstyle": "arc"},
            )

        fig.tight_layout()
        plt.show()

    def make_plot_gui(self, ax1: Axes, ax2: Axes):
        """Generate the plot for the signal and frequency space.

        Parameters
        ----------
        ax1 : Axes
            Axes object for the signal data.
        ax2 : Axes
            Axes object for the frequency domain.

        """
        v = self._data
        t = self._time
        freq = self._freq
        psd = self._psd
        peaks_ix = self._peaks_ix

        ax1.clear()
        ax2.clear()

        ax1.plot(t, v, color="C0", lw=0.7)

        # window = hamming(len(v))
        # ax1.plot(t, v*window, color="C3", lw=0.7)

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [mm/s]")

        ax2.plot(freq, psd, color="C0", lw=0.7)
        ax2.scatter(freq[peaks_ix], psd[peaks_ix], marker="o", color="r")
        ax2.axhline(self.psd_max_limit, color="C3", ls="--", lw=0.5)
        ax2.annotate(
            "detect threshold",
            xy=(freq[-1], self.psd_max_limit),
            color="C3",
            fontsize=6,
            ha="right",
            va="bottom",
        )

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD")

        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

        for pi in peaks_ix:
            x_i = freq[pi]
            y_i = psd[pi]
            ax2.annotate(
                f"{x_i:1.0f} Hz",
                xy=(x_i, y_i),
                xytext=(20, -10),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "connectionstyle": "arc"},
            )

        ax1.figure.tight_layout()
        ax1.figure.canvas.draw()
        ax2.figure.tight_layout()
        ax2.figure.canvas.draw()

    def update_signal(self) -> None:
        """Update the signal that is analysed."""
        while True:
            try:
                data = self.vibro.q.get_nowait()
            except queue.Empty:
                break

            ix_last = self.prev_sig_ix
            self.live_data[ix_last : (ix_last + len(data)), :] = data
            self.prev_sig_ix += len(data)

    @property
    def n_points(self):
        """Length of the data."""
        return self._n_points

    def calc_moe(
        self, length: float, width: float, thick: float, weight: float
    ) -> float:
        """Compute the MOE according to beam theory.

        Parameters
        ----------
        length : float
            Length of the specimen.
        width : float
            Width of the cross-section.
        thick : float
            Thickness of the cross-section.
        weight : float
            Weight of the specimen in grams.

        Returns
        -------
        moe : float
            Modulus of elasticity

        """
        freq = self._freq
        peaks_ix = self._peaks_ix

        rho = weight / (length * width * thick)

        return 4 * length**2 * freq[peaks_ix] ** 2 * rho * 1e-6


class VibrometerCapture:
    """Manages the stream input.

    Parameters
    ----------
    device : int
        Device used to capture the signal.
    rate : int
        Rate used to sample the signal.
    velo : float
        Maximum velocity captured by the device.

    """

    def __init__(self, device: int, rate: int, velo: float, downsample: float):
        self.device = device
        self.rate = rate
        self.velo = velo
        self.q = queue.Queue()
        self.downsample = downsample

        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.rate,
            callback=self.audio_callback,
        )

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
        scale_factor = self.velo / 4.0
        self.q.put(indata[:: self.downsample, mapping] * scale_factor)


def play_detected():
    """Play a sound when impulse is detected."""
    fs = 44100
    total_t = 0.1
    freq_1 = 400
    freq_2 = 600
    freq_3 = 900
    t = np.linspace(0, total_t, int(fs * total_t))
    audio_1 = 3 * np.sin(t * 2.0 * np.pi * freq_1)
    audio_2 = 3 * np.sin(t * 2.0 * np.pi * freq_2)
    audio_3 = 3 * np.sin(t * 2.0 * np.pi * freq_3)

    audio = np.concatenate((audio_1, audio_2, audio_3))

    sd.play(audio, fs)
    sd.wait()


def main():
    # Parameters
    dev_num = None

    for ix, dev in enumerate(sd.query_devices()):
        match_name = re.match(DEV_NAME, dev["name"])
        if match_name:
            dev_num = ix
            dev_rate = dev["default_samplerate"]
            break

    vib_analysis = SignalAnalysis(device=dev_num, sample_rate=dev_rate, velo=VELO)
    # Record signal after impulse
    vib_analysis.wait_and_record(
        duration=REC_TIME, total_recording=20, thress=THRESSHOLD
    )

    vib_analysis.compute_frequencies()

    vib_analysis.make_plot()


if __name__ == "__main__":
    main()
    # play_detected()
