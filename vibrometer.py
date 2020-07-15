import re
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import blackman, find_peaks, hanning, hamming

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
    return 1 if x == 0 else 2**(x - 1).bit_length()


class SignalAnalysis:
    """Docstring for SignalAnalysis.

    Parameters
    ----------
    device : TODO
    sample_rate : TODO
    duration : TODO
    total_recording : TODO

    """
    def __init__(self, device, sample_rate, velo):
        self._device = device
        self._sample_rate = int(sample_rate)
        self._velo = velo
        self._spacing = 1.0 / sample_rate
        self._n_points = None
        self._data = None
        self._time = None
        self._freq = None
        self._psd = None
        self._peaks_ix = None

    def wait_and_record(self, duration, total_recording, thress):
        """Start recording and return the signal after an impulse is given.

        Parameters
        ----------
        duration : TODO
        total_recording : TODO

        Returns
        -------
        TODO

        """
        total_recording = int(total_recording)

        # Parameters
        sd.default.device = self._device

        fs = int(self._sample_rate)

        recording = sd.rec(total_recording * fs, samplerate=fs, channels=1, blocking=False)
        print("------------------")
        print("Recording...")

        print("Waiting for impulse...")
        t_i = time()
        while True:
            ix = np.argmax(np.abs(recording[:, 0]))
            val = recording[ix, 0]
            if val > thress:
                print("Impulse detected!...")
                break
            if time() - t_i > total_recording:
                break
            sleep(0.01)

        ix = ix - int(fs * 0.001)
        sleep(duration)

        sd.stop()
        print("Stop recording...")

        data = recording[ix:ix + int(duration * fs)]
        data = data.reshape((-1, ))

        # Remove mean
        data = data - np.mean(data)

        # Scale of the signal to obtain mm/s
        data = data * self._velo / 4.0

        time_ = np.arange(start=0, step=1 / fs, stop=len(data) / fs)

        self._data = data
        self._time = time_
        self._n_points = len(data)

        return data, time_

    def compute_frequencies(self):
        """TODO: Docstring for compute_frequencies.
        Returns
        -------
        TODO

        """
        n_points = self._n_points
        data = self._data
        spacing = self._spacing

        # Apply window to reduce noise
        window = hamming(n_points)

        # Apply FFT
        psd_ = fft(data * window)[1:n_points // 2]
        freq = fftfreq(n_points, d=spacing)[1:n_points // 2]

        psd_ = np.real(psd_)
        psd = 1.0 / n_points * np.abs(psd_)
        psd = psd[1:]
        freq = freq[1:]

        # Obtain the three most relevant frequencies
        height = np.max(psd) * 0.3
        peaks_ix, _ = find_peaks(psd, distance=200, height=height)

        # Sort peaks
        p_sort = np.argsort(psd[peaks_ix])
        peaks_ix_sort = peaks_ix[p_sort]

        print("Frequencies detected:")
        for kx, pi in enumerate(peaks_ix_sort):
            moe_i = compute_moe(freq[pi])
            print(f"\tf = {freq[pi]} Hz; \t E = {moe_i:1.2f} [MPa]")
            if kx >= 2:
                break

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
            ax2.annotate(f"{x_i} Hz", xy=(x_i, y_i), xytext=(20, -10),
                         textcoords="offset points", arrowprops={
                             "arrowstyle": "->",
                             "connectionstyle": "arc"
                         })

        fig.tight_layout()
        plt.show()

    def make_plot_gui(self, gui):
        v = self._data
        t = self._time
        freq = self._freq
        psd = self._psd
        peaks_ix = self._peaks_ix

        ax = gui.plot.ax
        ax2 = gui.plot_f.ax

        ax.clear()
        ax2.clear()

        ax.plot(t, v)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [mm/s]")

        ax2.plot(freq, psd)
        ax2.scatter(freq[peaks_ix], psd[peaks_ix], marker="o", color="r")

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD")

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Power spectrum")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

        for pi in peaks_ix:
            x_i = freq[pi]
            y_i = psd[pi]
            ax2.annotate(f"{x_i} Hz", xy=(x_i, y_i), xytext=(20, -10),
                         textcoords="offset points", arrowprops={
                             "arrowstyle": "->",
                             "connectionstyle": "arc"
                         })

        ax.figure.canvas.draw()
        ax.figure.tight_layout()
        ax2.figure.canvas.draw()
        ax2.figure.tight_layout()

    @property
    def n_points(self):
        """TODO: Docstring for n_points.
        Returns
        -------
        TODO

        """
        return self._n_points

    def calc_moe(self, length, width, thick, weight):
        """TODO: Docstring for calc_moe.

        Parameters
        ----------
        length : TODO
        width : TODO
        thick : TODO
        weight : TODO

        Returns
        -------
        TODO

        """
        freq = self._freq
        peaks_ix = self._peaks_ix

        rho = weight / (length * width * thick)

        return 4 * length**2 * freq[peaks_ix]**2 * rho


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
    vib_data, time_ = vib_analysis.wait_and_record(duration=REC_TIME,
            total_recording=20, thress=THRESSHOLD)

    vib_analysis.compute_frequencies()

    vib_analysis.make_plot()


def compute_moe(freq):
    """Compute the MOE based on the fequency.

    Parameters
    ----------
    freq : TODO

    Returns
    -------
    TODO

    """
    return 4 * L**2 * freq**2 * RHO


if __name__ == "__main__":
    main()
