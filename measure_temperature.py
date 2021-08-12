# -*- coding: utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.interpolate import interp1d
from n2_fit_2 import Spectrum_D2

L = 0.1
frac = 0.995


def read_ch(csv_name):
    time = []
    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []

    with open(csv_name + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            try:
                time.append(float(row[0]))
                ch1.append(float(row[1]))
                ch2.append(float(row[2]))
                ch3.append(float(row[3]))
                ch4.append(float(row[4]))
            except:
                pass

    time = np.array(time)
    ch = [ch3, ch1, ch2, ch4]

    for i in [1, 2, 3]:
        ch[i] = np.array(ch[i])
        ch[i] = savgol_filter(ch[i], 2001, 3)
        ch[i] = ch[i] - np.min(ch[i])
        ch[i] = ch[i] / np.max(ch[i])

    return time, ch


def peak(piezo, abso):
    abso = abso - np.min(abso)
    abso = abso / np.max(abso)
    maxi = find_peaks(abso, height=(0, 0.3),distance=2000, width=100)[0]

    plt.plot(piezo, abso)
    plt.plot(piezo[maxi], abso[maxi], 'o')
    plt.show()

    num = int(input("Quel est le bon pic ? "))
    num -= 1
    mini_1 = maxi[num]

    orig = piezo[mini_1]

    return orig


def fabry(temps, ch):
    abso = ch[2]
    maxi = find_peaks(abso, height=(0.4, 1.3),distance=5000, width=500, prominence=0.2)[0]

    """plt.plot(temps, abso)
    #plt.plot(temps, ch[3])

    plt.plot(temps[maxi], abso[maxi], 'o')
    plt.show()"""

    freq = np.linspace(0, len(maxi) * 0.67, len(maxi))
    f = interp1d(temps[maxi], freq, kind='quadratic')
    temps_2 = f(temps[maxi[0]: maxi[-1]])

    orig = peak(temps_2, ch[3][maxi[0]: maxi[-1]])
    temps_2 = temps_2 - orig

    temps = np.linspace(temps_2[0], temps_2[-1], len(temps_2))

    for i in range(0, len(ch)):
        ch[i] = ch[i][maxi[0]: maxi[-1]]

        f = interp1d(temps_2, ch[i])
        ch[i] = f(np.linspace(temps_2[0], temps_2[-1], len(ch[i])))

    """plt.plot(temps, ch[2])
    plt.xlabel('detuning $\Delta$')
    plt.ylabel('normalized intensities after the SAS and de FP')
    plt.plot(temps, ch[3])
    plt.show()"""

    return temps, ch


def fit_temp(T, detun_min, detun_max, exp):
    D2 = Spectrum_D2(T=T, detun_min=detun_min, detun_max=detun_max, L=L, frac=frac, waist=1e-3)

    f = interp1d(np.linspace(detun_min, detun_max, len(exp)), exp, kind='quadratic')
    diff = D2.trans() / np.max(D2.trans()) - f(np.linspace(detun_min, detun_max, 10000))

    return np.mean(diff ** 2)


if __name__ == "__main__":
    path = 'Data/tangui2'
    name = 'tangui2_4'

    temps, ch_tot = read_ch(str(path)+'/'+str(name))

    #ch_tot[0] = savgol_filter(ch_tot[0], 10001, 3)
    ch_tot[0] = ch_tot[0]/np.max(ch_tot[0])
    ch_tot[1] = ch_tot[1]/ch_tot[0]
    plt.plot(ch_tot[1])
    plt.show()
    temps, ch_tot = fabry(temps, ch_tot)

    np.savetxt(path+'/'+'temps_{}.txt'.format(name), np.column_stack((temps)))
    np.savetxt(path+'/'+'ch_{}.txt'.format(name), np.column_stack((ch_tot)))
    print('Fichiers txt sauvegardés')

    temps = np.loadtxt(path + '/' + 'temps_{}.txt'.format(name))
    ch = np.transpose(np.loadtxt(path + '/' + 'ch_{}.txt'.format(name)))[1]
    f = interp1d(temps, ch, kind='quadratic')
    ch = savgol_filter(f(np.linspace(temps[0], temps[-1], 10000)), 101, 3)
    ch = ch / np.max(ch)

    detun_min, detun_max = temps[0] * 1e9, temps[-1] * 1e9

    res = minimize(fit_temp, x0=[390], args=(detun_min, detun_max, ch), bounds=((360, 440),))
    T = res.x[0]
    D2 = Spectrum_D2(T=T, detun_min=detun_min, detun_max=detun_max, L=L, frac=frac, waist=1e-3)

    temps_2 = np.linspace(detun_min * 1e-9, detun_max * 1e-9, 10000)

    plt.plot(temps_2[::10], ch[::10] * np.max(D2.trans()), label='Experiment', color='b')
    plt.plot(temps_2[::10], D2.trans()[::10], '--', label='Theoretical fit : {:.03}°C'.format(T-273), color='r')
    plt.xlabel('Detuning (GHz)')
    plt.ylabel('Transmission')
    plt.legend()

    # tikz.save('temperature.tex')

    plt.show()