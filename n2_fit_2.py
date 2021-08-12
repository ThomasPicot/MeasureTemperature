#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: Guillaume Brochier, May 2021
Computes the transmission profile and the non linear index thanks to the open three level model.
All the constants are in SI units.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, hbar, c, mu_0, Boltzmann, m_n, e
from scipy.special import wofz


def V(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def N(T):
    P_v = 10 ** (15.88253 - 4529.635 / T + 0.00058663 * T - 2.99138 * np.log10(T))
    return 133.323 * P_v / (Boltzmann * T)


def beta_n(T):
    return 2 * np.pi * 1.03e-13 * N(T=T)


def C_f(F_g, F_e, at, D='D2'):
    """
    Takes the transition and returns the coefficient C_f_2 along with the total degenerescence
    """

    if D == 'D2':
        if at == 87:
            F_g_tot = [1, 2]
            F_e_tot = [0, 1, 2, 3]

            C_f_2 = np.array([[1 / 9, 5 / 18, 5 / 18, 0], [0, 1 / 18, 5 / 18, 7 / 9]])
            return C_f_2[F_g_tot.index(F_g), F_e_tot.index(F_e)], 8

        elif at == 85:
            F_g_tot = [2, 3]
            F_e_tot = [1, 2, 3, 4]

            C_f_2 = np.array([[1 / 3, 35 / 81, 28 / 81, 0], [0, 10 / 81, 35 / 81, 1]])
            return C_f_2[F_g_tot.index(F_g), F_e_tot.index(F_e)], 12

    elif D == 'D1':
        if at == 87:
            F_g_tot = [1, 2]
            F_e_tot = [1, 2]

            C_f_2 = np.array([[1 / 18, 5 / 18], [5 / 18, 5 / 18]])
            return C_f_2[F_g_tot.index(F_g), F_e_tot.index(F_e)], 8

        elif at == 85:
            F_g_tot = [2, 3]
            F_e_tot = [2, 3]

            C_f_2 = np.array([[10 / 81, 35 / 81], [35 / 81, 28 / 81]])
            return C_f_2[F_g_tot.index(F_g), F_e_tot.index(F_e)], 12


class Spectrum_D2:
    def __init__(self, T, detun_min, detun_max, L, frac, waist=0.01):
        """
        detun_min /detun_max : plage sur laquelle il faut réaliser le spectre. Le zero correspond ici à  F_g = 3 --> F_e = 3  du RB 85. 
        L : longueur de la cellule
        frac : fraction de Rubidium 87 dans la cellule
        waist : waist du faisceau, utile seulement pour n2
        """

        self.T = T
        self.L = L
        self.frac = frac
        self.waist = waist
        self.detun = np.linspace(detun_min, detun_max, 10000)

        # On prend les nombres donnés dans Siddons et al (2008) et on décale les
        # fréquences pour placer le zéro sur F_g : 3 -> F_e : 3 du Rb 85

        self.detun_87 = np.array(
            [[4027.403e6, 4099.625e6, 4256.57e6, 0], [0, -2735.05e6, -2578.11e6, -2311.26e6]]) + 1307.87e6
        self.detun_85 = np.array(
            [[1635.454e6, 1664.714e6, 1728.134e6, 0], [0, -1371.29e6, -1307.87e6, -1186.91e6]]) + 1307.87e6

        self.wl = 780.241e-9
        self.k = 2 * np.pi / self.wl
        self.Gamma = 2 * np.pi * 6.065e6
        self.m87 = 1.44316060e-25
        self.m85 = 1.44316060e-25 - 2 * m_n
        self.u = np.sqrt(2 * Boltzmann * T / (self.frac * self.m87 + (1 - self.frac) * self.m85))
        self.d = np.sqrt(9 * epsilon_0 * hbar * self.Gamma * self.wl ** 3 / (8 * np.pi ** 2))
        self.gamma_t = self.u / self.waist * 2 / np.sqrt(np.pi)  # formule à compléter!
        self.gamma = self.Gamma / 2 + self.gamma_t + beta_n(self.T) / 2

    def freq_trans(self, F_g, F_e, at):
        if at == 87:
            F_g_tot = [1, 2]
            F_e_tot = [0, 1, 2, 3]
            return self.detun_87[F_g_tot.index(F_g), F_e_tot.index(F_e)]

        elif at == 85:
            F_g_tot = [2, 3]
            F_e_tot = [1, 2, 3, 4]
            return self.detun_85[F_g_tot.index(F_g), F_e_tot.index(F_e)]

    def alpha_0(self, F_g, F_e, at):

        C_f2, deg = C_f(F_g, F_e, at=at, D='D2')
        mu_2 = C_f2 * self.d ** 2 / (2 * F_g + 1)

        return (2 * F_g + 1) / deg * N(self.T) / (epsilon_0 * hbar) * mu_2 / self.gamma

    def E_sat(self, F_g, F_e, at=87):

        mu = np.sqrt(C_f(F_g, F_e, at=at, D='D2')[0] / (2 * F_g + 1)) * self.d

        E_satur = hbar * self.gamma / mu * np.sqrt(
            2 * self.gamma_t / self.gamma * (1 + self.Gamma / (2 * self.gamma)) / (1 + self.gamma_t / self.gamma))

        return E_satur

    def chi_1(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        return self.alpha_0(F_g, F_e, at=at) * (1j - detun_trans / self.gamma) / (1 + (detun_trans / self.gamma) ** 2)

    def chi_3(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        return -1 / self.E_sat(F_g, F_e, at=at) ** 2 * self.alpha_0(F_g, F_e, at=at) * (
                    1j - detun_trans / self.gamma) / (1 + (detun_trans / self.gamma) ** 2) ** 2

    def chi_1_doppler(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        b = self.gamma / (self.k * self.u)
        a = detun_trans / (self.k * self.u)
        return self.alpha_0(F_g, F_e, at=at) * b * V(a + 1j * b)

    def chi_3_doppler(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        b = self.gamma / (self.k * self.u)
        a = detun_trans / (self.k * self.u)
        return -1j / self.E_sat(F_g, F_e, at=at) ** 2 * self.alpha_0(F_g, F_e, at=at) * b ** 2 / 2 * (
                    (a + 1j * b) * V(a + 1j * b) + (a - 1j * b) * V(-a + 1j * b))
        # return -1/self.E_sat(F_g, F_e, at=at)**2 *self.alpha_0(F_g, F_e, at=at)* b*(-b*0 + (-1/4+b**2-1j*b*a)*V(a+1j*b) - 1/4*V(-a+1j*b))

    def trans(self):
        at_tot = [85, 87]
        alpha = 0
        for at in at_tot:
            if at == 87:
                F_tot = [[1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3]]
                frac_at = self.frac

            elif at == 85:
                F_tot = [[2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [3, 4]]
                frac_at = 1 - self.frac

            for F_g, F_e in F_tot:
                alpha += self.k * frac_at * self.chi_1_doppler(F_g, F_e, at=at).imag
                # plt.plot(self.detun*1e-9, np.exp(-self.k * frac_at* self.chi_1_doppler(F_g, F_e, at=at).imag*self.L), '--', color='r')
        return np.exp(-alpha * self.L)

    def n2(self):
        at_tot = [85, 87]
        n_2 = 0
        n_2_2 = 0
        for at in at_tot:
            if at == 87:
                F_tot = [[1, 0], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3]]
                frac_at = self.frac

            elif at == 85:
                F_tot = [[2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [3, 4]]
                frac_at = 1 - self.frac

            for F_g, F_e in F_tot:
                n_2 += frac_at * self.chi_3_doppler(F_g, F_e, at=at).real / (epsilon_0 * c)

        return n_2


class Spectrum_D1:
    def __init__(self, T, detun_min, detun_max, L, frac, waist=0.01):
        """
        detun_min /detun_max : plage sur laquelle il faut réaliser le spectre. Le zero correspond ici à  F_g = 3 --> F_e = 3  du RB 85. 
        L : longueur de la cellule
        frac : fraction de Rubidium 87 dans la cellule
        waist : waist du faisceau, utile seulement pour n2
        """

        self.T = T
        self.L = L
        self.frac = frac
        self.waist = waist
        self.detun = np.linspace(detun_min, detun_max, 10000)

        # On prend les nombres donnés dans Siddons et al (2008) et on décale les fréquences pour placer
        # le zéro sur F_g : 3 -> F_e : 3 du Rb 85

        self.detun_87 = np.array([[3820.046e6, 4632.339e6], [-3012.644e6, -2202.381e6]]) + 1307.87e6
        self.detun_85 = np.array([[1538.063e6, 1900.087e6], [-1497.657e6, -1135.721e6]]) + 1307.87e6

        self.wl = 794.979e-9
        self.k = 2 * np.pi / self.wl
        self.Gamma = 2 * np.pi * 5.746e6
        self.m87 = 1.44316060e-25
        self.m85 = 1.44316060e-25 - 2 * m_n
        self.u = np.sqrt(2 * Boltzmann * T / (self.frac * self.m87 + (1 - self.frac) * self.m85))
        self.d = np.sqrt(9 * epsilon_0 * hbar * self.Gamma * self.wl ** 3 / (8 * np.pi ** 2))
        self.gamma_t = self.u / self.waist * 2 / np.sqrt(np.pi)  # formule à compléter!
        self.gamma = self.Gamma / 2 + self.gamma_t + beta_n(self.T) / 2 * 0.73 / 1.03

    def freq_trans(self, F_g, F_e, at):
        if at == 87:
            F_g_tot = [1, 2]
            F_e_tot = [1, 2]
            return self.detun_87[F_g_tot.index(F_g), F_e_tot.index(F_e)]

        elif at == 85:
            F_g_tot = [2, 3]
            F_e_tot = [2, 3]
            return self.detun_85[F_g_tot.index(F_g), F_e_tot.index(F_e)]

    def alpha_0(self, F_g, F_e, at):

        C_f2, deg = C_f(F_g, F_e, at=at, D='D1')
        mu_2 = C_f2 * self.d ** 2 / (2 * F_g + 1)

        return (2 * F_g + 1) / deg * N(self.T) / (epsilon_0 * hbar) * mu_2 / self.gamma

    def E_sat(self, F_g, F_e, at=87):

        mu = np.sqrt(C_f(F_g, F_e, at=at, D='D1')[0] / (2 * F_g + 1)) * self.d

        E_satur = hbar * self.gamma / mu * np.sqrt(
            2 * self.gamma_t / self.gamma * (1 + self.Gamma / (2 * self.gamma)) / (1 + self.gamma_t / self.gamma))

        return E_satur

    def chi_1(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        return self.alpha_0(F_g, F_e, at=at) * (1j - detun_trans / self.gamma) / (1 + (detun_trans / self.gamma) ** 2)

    def chi_3(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        return -1 / self.E_sat(F_g, F_e, at=at) ** 2 * self.alpha_0(F_g, F_e, at=at) * (
                    1j - detun_trans / self.gamma) / (1 + (detun_trans / self.gamma) ** 2) ** 2

    def chi_1_doppler(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        b = self.gamma / (self.k * self.u)
        a = detun_trans / (self.k * self.u)
        return self.alpha_0(F_g, F_e, at=at) * b * V(a + 1j * b)

    def chi_3_doppler(self, F_g, F_e, at=87):
        detun_trans = 2 * np.pi * (self.detun - self.freq_trans(F_g, F_e, at))
        b = self.gamma / (self.k * self.u)
        a = detun_trans / (self.k * self.u)

        return -1j / self.E_sat(F_g, F_e, at=at) ** 2 * self.alpha_0(F_g, F_e, at=at) * b ** 2 / 2 * (
                    (a + 1j * b) * V(a + 1j * b) + (a - 1j * b) * V(-a + 1j * b))
        # return -1/self.E_sat(F_g, F_e, at=at)**2 *self.alpha_0(F_g, F_e, at=at)* b*(-b*0 + (-1/4+b**2-1j*b*a)*V(a+1j*b) - 1/4*V(-a+1j*b))

    def trans(self):
        at_tot = [85, 87]
        alpha = 0
        for at in at_tot:
            if at == 87:
                F_g_tot = [1, 2]
                F_e_tot = [1, 2]
                frac_at = self.frac

            elif at == 85:
                F_g_tot = [2, 3]
                F_e_tot = [2, 3]
                frac_at = 1 - self.frac

            for F_g in F_g_tot:
                for F_e in F_e_tot:
                    alpha += self.k * frac_at * self.chi_1_doppler(F_g, F_e, at=at).imag
                    # plt.plot(self.detun*1e-9, np.exp(-self.k * frac_at* self.chi_1_doppler(F_g, F_e, at=at).imag*self.L), '--', color='r')
        return np.exp(-alpha * self.L)

    def n2(self):
        at_tot = [85, 87]
        n_2 = 0
        for at in at_tot:
            if at == 87:
                F_g_tot = [1, 2]
                F_e_tot = [1, 2]
                frac_at = self.frac

            elif at == 85:
                F_g_tot = [2, 3]
                F_e_tot = [2, 3]
                frac_at = 1 - self.frac

            for F_g in F_g_tot:
                for F_e in F_e_tot:
                    n_2 += frac_at * self.chi_3_doppler(F_g, F_e, at=at).real / (epsilon_0 * c)
        return n_2


if __name__ == "__main__":
    detun_min, detun_max = -10e9, 12e9
    Spectrum = Spectrum_D2(273 + 155, detun_min, detun_max, 0.1, 0.995, 2e-3)
    detuning = np.linspace(detun_min * 1e-9, detun_max * 1e-9, 10000)

    trans = np.transpose(np.loadtxt('/home/guillaume/Documents/cours/M2/stage/mesures/n2/transmission_nl.txt'))
    plt.plot(detuning, Spectrum.trans())
    plt.plot(trans[0] - 1.2, trans[1] - trans[1, -1])
    plt.xlabel('Detuning (GHz)')
    plt.ylabel('Transmission')

    plt.figure(2)
    plt.plot(detuning, -1 * Spectrum.n2())
    plt.xlabel('Detuning (GHz)')
    plt.ylabel(r'$n_2$ (W/m$^2$)')
    plt.yscale('log')

    plt.show()
