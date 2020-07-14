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
REC_TIME = 0.04
# Name of the device to use
DEV_NAME = "default"


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def main():
    # Parameters
    dev_num = None

    for ix, dev in enumerate(sd.query_devices()):
        match_name = re.match(DEV_NAME, dev["name"])
        if match_name:
            dev_num = ix
            dev_rate = dev["default_samplerate"]
            break

    sd.default.device = dev_num
    fs = int(dev_rate)
    # fs = 2000
    spacing = 1.0 / fs
    duration = 6

    recording = sd.rec(duration * fs, samplerate=fs, channels=1, blocking=False)
    print("------------------")
    print("Recording...")

    t_i = time()
    while True:
        ix = np.argmax(recording)
        val = recording[ix]
        if val > THRESSHOLD:
            print("Impulse detected!...")
            break
        if time() - t_i > duration:
            break
        sleep(0.01)

    ix = ix - int(fs * 0.005)
    sleep(REC_TIME)

    sd.stop()
    print("Stop recording...")

    vib_data = recording[ix:ix + int(REC_TIME * fs)]
    vib_data = vib_data.reshape((-1, ))

    # Remove mean
    vib_data = vib_data - np.mean(vib_data)

    # Scale of the signal to obtain mm/s
    vib_data = vib_data * VELO / 4.0

    time_ = np.arange(start=0, step=1 / fs, stop=len(vib_data) / fs)

    sample_pnts = len(vib_data)
    if (sample_pnts % 2) == 0:
        pass
    else:
        sample_pnts = sample_pnts - 1

    vib_data = vib_data[:sample_pnts]

    # Apply window to reduce noise
    window = hamming(sample_pnts)
    pow_spec_ = fft(vib_data * window)[1:sample_pnts // 2]

    freq = fftfreq(sample_pnts, d=spacing)[1:sample_pnts // 2]
    pow_spec_ = np.real(pow_spec_)

    pow_spec = 1.0 / sample_pnts * np.abs(pow_spec_)

    pow_spec = pow_spec[1:]
    freq = freq[1:]
    # Obtain the three most relevant frequencies
    height = np.max(pow_spec) * 0.3
    peaks_ix, _ = find_peaks(pow_spec, distance=200, height=height)
    # Sort peaks
    p_sort = np.argsort(pow_spec[peaks_ix])
    peaks_ix_sort = peaks_ix[p_sort]

    print("Frequencies detected:")
    for kx, pi in enumerate(peaks_ix_sort):
        moe_i = compute_moe(freq[pi])
        print(f"\tf = {freq[pi]} Hz; \t E = {moe_i:1.2f} [MPa]")
        if kx >= 2:
            break

    make_plot(vib_data, time_, pow_spec, freq, peaks_ix)


def make_plot(velo, t, spec, freq, peaks):
    """Make plot with measured data.

    Parameters
    ----------
    velo : TODO
    time : TODO
    spec : TODO
    freq : TODO

    Returns
    -------
    TODO

    """
    fig = plt.figure(figsize=(5.8, 6.0))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(t, velo)

    ax2.plot(freq, spec)
    ax2.scatter(freq[peaks], spec[peaks], marker="o", color="r")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [mm/s]")

    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Power spectrum")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

    for pi in peaks:
        x_i = freq[pi]
        y_i = spec[pi]
        ax2.annotate(f"{x_i} Hz", xy=(x_i, y_i), xytext=(20, -10),
                     textcoords="offset points", arrowprops={
                         "arrowstyle": "->",
                         "connectionstyle": "arc"
                     })

    fig.tight_layout()
    plt.show()


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
