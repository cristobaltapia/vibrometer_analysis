import queue
import re
import sys
from time import sleep, time
from threading import Timer

import matplotlib as mpl  # isort: skip
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from numpy.fft import fft, fftfreq
from scipy.signal import blackman, find_peaks, hamming, hanning

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
        self.prev_sig_ix = 0

    def wait_and_record(self, duration, total_recording, thress, progress=None):
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

        rate = int(self._sample_rate)

        length = int((total_recording + 1) * rate)
        self.live_data = np.zeros((length, 1))

        # Initialize a Vibrometer Capture object
        self.vibro = VibrometerCapture(device=self._device, rate=rate, velo=self._velo,
                                       downsample=1)
        self.vibro.start_stream()

        timer_capture = Timer(interval=1, function=self.update_signal)
        timer_capture.start()

        print("------------------")
        print("Recording...")
        print("Waiting for impulse...")
        if progress:
            # Reset progress bar
            progress.setVisible(True)

        extra_time = 0.005

        t_init = time()
        impact_detected = False

        while True:
            if progress:
                progress.setValue(time() - t_init)

            ix = np.argmax(np.abs(self.live_data[:, 0]))
            val = np.abs(self.live_data[ix, 0])

            if val > thress:
                print("Impulse detected!...")
                impact_detected = True
                break
            if time() - t_init > total_recording:
                break
            sleep(0.01)

        ix = ix + int(rate * extra_time)
        sleep(duration + extra_time)

        self.vibro.stop_stream()
        self.vibro.close_stream()
        timer_capture.cancel()
        # One last call to update the signal results
        self.update_signal()

        print("Stop recording...")

        data = self.live_data[ix:ix + int(rate * duration):, 0]
        data = data.reshape((-1, ))
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

    def compute_frequencies(self, min_freq=0, max_freq=20000):
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
        psd_ = fft(data * window)[0:n_points // 2]
        freq = fftfreq(n_points, d=spacing)[0:n_points // 2]

        psd_ = np.real(psd_)
        psd = 1.0 / n_points * np.abs(psd_)

        # Narrow the frequency space to the needed one
        ix_min = np.argmin(np.abs(freq - min_freq))
        ix_max = np.argmin(np.abs(freq - max_freq))

        psd = psd[ix_min:ix_max + 1]
        freq = freq[ix_min:ix_max + 1]

        # Obtain the three most relevant frequencies
        self.psd_max_limit = np.max(psd) * 0.3
        peaks_ix, _ = find_peaks(psd, distance=100, height=self.psd_max_limit)

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

    def make_plot_gui(self, ax1, ax2):
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
        ax2.annotate("detect threshold", xy=(freq[-1], self.psd_max_limit), color="C3",
                     fontsize=6, ha="right", va="bottom")

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("PSD")

        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

        for pi in peaks_ix:
            x_i = freq[pi]
            y_i = psd[pi]
            ax2.annotate(f"{x_i:1.0f} Hz", xy=(x_i, y_i), xytext=(20, -10),
                         textcoords="offset points", arrowprops={
                             "arrowstyle": "->",
                             "connectionstyle": "arc"
                         })

        ax1.figure.tight_layout()
        ax1.figure.canvas.draw()
        ax2.figure.tight_layout()
        ax2.figure.canvas.draw()

    def update_signal(self):
        """TODO: Docstring for update_signal.
        Returns
        -------
        TODO

        """
        while True:
            try:
                data = self.vibro.q.get_nowait()
            except queue.Empty:
                break

            ix_last = self.prev_sig_ix
            self.live_data[ix_last:(ix_last + len(data)), :] = data
            self.prev_sig_ix += len(data)

        return True

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

        return 4 * length**2 * freq[peaks_ix]**2 * rho * 1e-6


class VibrometerCapture:
    """Manages the stream input.

    Parameters
    ----------
    device : TODO
    rate : TODO

    """
    def __init__(self, device, rate, velo, downsample):
        """TODO: to be defined.



        """
        self.device = device
        self.rate = rate
        self.velo = velo
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
        scale_factor = self.velo / 4.0
        self.q.put(indata[::self.downsample, mapping] * scale_factor)


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
    vib_analysis.wait_and_record(duration=REC_TIME, total_recording=20, thress=THRESSHOLD)

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
