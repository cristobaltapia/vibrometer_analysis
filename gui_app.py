import re
from guietta import Gui, _, B, E, L, R, M, ___, III, VS, HS, HSeparator, VSeparator
from vibrometer import SignalAnalysis
import sounddevice as sd
from vibrometer import DEV_NAME

VELO = None
# DEV_NAME = 'VIB-E-220 *SN0763'

def listen_for_signal(gui, *args):
    # Parameters
    dev_num = None

    rec_time = float(gui.rec_time)

    for ix, dev in enumerate(sd.query_devices()):
        match_name = re.findall(DEV_NAME, dev["name"])
        if match_name:
            dev_num = ix
            dev_rate = dev["default_samplerate"]
            break

    gui.widgets['status'].setText("Waiting for impulse...")
    thress = float(gui.thress)

    vib_analysis = SignalAnalysis(device=dev_num, sample_rate=dev_rate, velo=VELO)
    # Record signal after impulse
    vib_analysis.wait_and_record(duration=rec_time, total_recording=10, thress=thress)

    max_freq = int(gui.max_freq)
    min_freq = int(gui.min_freq)

    freq = vib_analysis.compute_frequencies(min_freq=min_freq, max_freq=max_freq)
    print(freq)

    l = float(gui.l)
    w = float(gui.w)
    t = float(gui.t)
    kg = float(gui.kg)

    moes = vib_analysis.calc_moe(length=l, width=w, thick=t, weight=kg)

    gui.results = {f"f_{k} :": f"{val:1.0f} Hz" for k, val in enumerate(freq) if k < 3}
    gui.moes = {f"E_{k} :": f"{val:1.0f} MPa" for k, val in enumerate(moes) if k < 3}

    vib_analysis.make_plot_gui(gui)
    gui.status = "Idle..."


def toggle_velo(gui, *args):
    """TODO: Docstring for toogle_velo.

    Parameters
    ----------
    gui : TODO
    *args : TODO

    Returns
    -------
    TODO

    """
    if gui.velo_1.isChecked():
        VELO = 20
    elif gui.velo_2.isChecked():
        VELO = 100
    elif gui.velo_3.isChecked():
        VELO = 500


if __name__ == "__main__":
    gui = Gui(
        ['VELO [mm/s]:',       R('velo_1'),    'Width (mm)',         '__w__'],
        [_,                    R('velo_2'),    'Thickness (mm)',     '__t__'],
        [_,                    R('velo_3'),    'Length (mm)',        '__l__'],
        [_,                    _,              'Weight (kg)',        '__kg__'],
        ['Recording time ',    '__rec_time__', ['Start'],            _],
        ['Thresshold ',        '__thress__',   _,                    _],
        ['Status: ',           'status',       'results',            'moes'],
        [L('min freq. (Hz):'), '__min_freq__', L('max freq. (Hz):'), '__max_freq__'],
        [HS('min_freq_s'),     ___,            HS('max_freq_s'),     ___],
        [M('plot'),            ___,            ___,                  ___],
        [M('plot_f'),          ___,            ___,                  ___],
    )

    gui.widgets["velo_1"].setText("20")
    gui.widgets["velo_2"].setText("100")
    gui.widgets["velo_3"].setText("500")

    gui.velo_1 = toggle_velo
    gui.velo_2 = toggle_velo
    gui.velo_3 = toggle_velo
    gui.velo_2.toggle()
    VELO = 100.
    gui.status = "Idle..."

    gui.min_freq_s = 0
    gui.max_freq_s = 99

    with gui.max_freq_s:
        freq_str = gui.max_freq_s * 20000.0 / 99.0
        gui.widgets["max_freq"].setText(f"{freq_str:1.0f}")

    with gui.min_freq_s:
        freq_min_str = gui.min_freq_s * 20000.0 / 99.0
        gui.widgets["min_freq"].setText(f"{freq_min_str:1.0f}")

    gui.rec_time = 0.5
    gui.thress = 2e-2
    gui.w = 90.0
    gui.t = 20.0
    gui.l = 640.0
    gui.kg = 0.552

    gui.results = {
        "f₁:": None,
        "f₂:": None,
        "f₃:": None,
    }
    gui.moes = {
        "E₁:": None,
        "E₂:": None,
        "E₃:": None,
    }
    gui.Start = listen_for_signal
    gui.run()
