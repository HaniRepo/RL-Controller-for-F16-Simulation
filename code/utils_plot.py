import matplotlib.pyplot as plt
import numpy as np

def plot_vt(times, vt, sp, title="Airspeed Tracking"):
    times = np.asarray(times)
    vt = np.asarray(vt)
    plt.figure(figsize=(6,3))
    plt.plot(times, vt, label="Vt")
    plt.axhline(sp, linestyle="--", label="setpoint")
    plt.xlabel("Time [s]"); plt.ylabel("Vt [ft/s]")
    plt.title(title); plt.legend(); plt.tight_layout()
    return plt.gcf()