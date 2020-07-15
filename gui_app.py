import re
from guietta import Gui, _, B, E, L, R, M, ___, III, VS, HS, HSeparator, VSeparator
from vibrometer import SignalAnalysis
import sounddevice as sd
from vibrometer import DEV_NAME, REC_TIME

VELO = None


def listen_for_signal(gui, *args):
    # Parameters
    dev_num = None

    rec_time = gui.rec_time

    for ix, dev in enumerate(sd.query_devices()):
        match_name = re.match(DEV_NAME, dev["name"])
        if match_name:
            dev_num = ix
            dev_rate = dev["default_samplerate"]
            break

    gui.status = "Waiting for impulse..."

    vib_analysis = SignalAnalysis(device=dev_num, sample_rate=dev_rate, velo=VELO)
    # Record signal after impulse
    vib_data, time_ = vib_analysis.wait_and_record(duration=REC_TIME, total_recording=10)

    freq = vib_analysis.compute_frequencies()

    gui.results = {f"f_{k}": f"{val:1.2f} Hz" for k, val in enumerate(freq)}

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
        ['VELO [mm/s]:',    R('velo_1'),    _,         _],
        [_,                 R('velo_2'),    _,         _],
        [_,                 R('velo_3'),    _,         _],
        ['Recording time ', '__rec_time__', ['Start'], _],
        ['Status: ',        'status',       'results', ___],
        [M('plot'),         ___,            ___,       ___],
        [M('plot_f'),         ___,            ___,       ___],
    )

    gui.widgets["velo_1"].setText("20")
    gui.widgets["velo_2"].setText("100")
    gui.widgets["velo_3"].setText("500")

    # gui.velo_1 = ("isChecked", toggle_velo)
    gui.velo_1 = toggle_velo
    gui.velo_2 = toggle_velo
    gui.velo_3 = toggle_velo
    gui.velo_1.toggle()
    gui.status = "Idle..."
    gui.rec_time = 0.5
    gui.results = {
        "f₁:": None,
        "f₂:": None,
        "f₃:": None,
    }
    gui.Start = listen_for_signal
    gui.run()
