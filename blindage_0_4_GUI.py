import matplotlib.pyplot as plt
import math
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import ttk

from numpy import *

from reportlab.pdfgen import canvas
from tkinter import filedialog
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import Color, black, blue, red
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Table
from reportlab.lib.pagesizes import A4
from reportlab.platypus import TableStyle
from reportlab.lib import colors

from PIL import Image
from openpyxl import load_workbook
import xlwings as xw
import xlsxwriter

from io import BytesIO
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

root = Tk()

list_nappe = ['O', 'N']
v1 = StringVar()
list_options_nappe = ttk.OptionMenu(root, v1, list_nappe[0], list_nappe[1], list_nappe[0]).grid(row=9, column=1)
L_nappe = ttk.Label(root, text='Nappe: ').grid(row=9, column=0)

list_type_coeff = ['A', 'R']
v2 = StringVar()
list_options_coeff = ttk.OptionMenu(root, v2, list_type_coeff[0], list_type_coeff[0], list_type_coeff[1]).grid(row=11, column=1)
L_Coeff = ttk.Label(root, text='Type Coeff: ').grid(row=11, column=0)

#nombre_couches = int(input("Entrer le nombre de couches (1, 2, 3 ou 4): "))

#presence_surcharge = input("Y' a t-il une surcharge? (O ou N): ")

e1 = tk.StringVar()
E_TN = tk.Entry(root, textvariable=e1, bd=3).grid(row=0, column=1)
L_TN = ttk.Label(root, text='Cote_TN (m): ').grid(row=0, column=0)

e2 = tk.StringVar()
E_epaisseur = tk.Entry(root, textvariable=e2, bd=3).grid(row=1, column=1)
L_epaisseur = ttk.Label(root, text='Epaisseur couche(m): ').grid(row=1, column=0)


e3 = tk.StringVar()
E_angle_frot1 = tk.Entry(root, textvariable=e3, bd=3).grid(row=2, column=1)
L_angle_frot1 = ttk.Label(root, text='Angle frottement(°): ').grid(row=2, column=0)

e4 = tk.StringVar()
E_cohesion1 = tk.Entry(root, textvariable=e4, bd=3).grid(row=3, column=1)
L_cohesion1 = ttk.Label(root, text='Cohesion(kpa): ').grid(row=3, column=0)

e5 = tk.StringVar()
E_surcharge = tk.Entry(root, textvariable=e5, bd=3).grid(row=4, column=1)
L_surcharge = ttk.Label(root, text='Surcharge(kpa): ').grid(row=4, column=0)

e6 = tk.StringVar()
E_distance = tk.Entry(root, textvariable=e6, bd=3).grid(row=5, column=1)
L_distance = ttk.Label(root, text='Distance d(m): ').grid(row=5, column=0)

e7 = tk.StringVar()
E_largeur = tk.Entry(root, textvariable=e7, bd=3).grid(row=6, column=1)
L_largeur = ttk.Label(root, text='Largeur B(m): ').grid(row=6, column=0)

e8 = tk.StringVar()
E_gamma_sat = tk.Entry(root, textvariable=e8, bd=3).grid(row=7, column=1)
L_gamma_sat = ttk.Label(root, text='Gamma_sat(kN/m3): ').grid(row=7, column=0)

e9 = tk.StringVar()
E_gamma_sec = tk.Entry(root, textvariable=e9, bd=3).grid(row=8, column=1)
L_gamma_sec = ttk.Label(root, text='Gamma_sec(kN/m3): ').grid(row=8, column=0)

e10 = tk.StringVar()
E_cote_nappe = tk.Entry(root, textvariable=e10, bd=3).grid(row=10, column=1)
L_cote_nappe = ttk.Label(root, text='Cote nappe(m): ').grid(row=10, column=0)

def calculer():
    global y_plot_h_f, x_plot_h_f, x_plot_h_total, x_plot_h_q, y_plot_h_total
    global z1, z2, z3, z4
    global presence_nappe
    presence_nappe = v1.get()
    global cote_TN
    cote_TN = float(e1.get()) # cote du terrain naturel
    global type_coefficient
    type_coefficient = v2.get()

    global epaisseur1, angle_frottement1, cohesion1
    epaisseur1 = float(e2.get())
    angle_frottement1 = math.radians(float(e3.get()))
    cohesion1 = float(e4.get())

    global surcharge
    surcharge = float(e5.get())

    global cote_inf1, cote_sup1
    cote_sup1 = cote_TN
    cote_inf1 = cote_sup1 - epaisseur1

    # DIFFUSION DE KREY
    angle_frottement_diffusion = math.degrees(angle_frottement1)
    global d, B
    d = float(e6.get())
    B = float(e7.get())

    if type_coefficient.upper() == 'A':
        global ka1
        ka1 = (1 - math.sin(angle_frottement1)) / (1 + math.sin(angle_frottement1))
    else:
        global k0_1
        k0_1 = 1 - math.sin(angle_frottement1)
    global z1, z2, z3, z4
    z1 = d * math.tan(math.radians(angle_frottement_diffusion))
    z2 = d * math.tan((math.radians(angle_frottement_diffusion / 2) + math.radians(180 / 4)))
    z3 = (d + B) * math.tan(math.radians(angle_frottement_diffusion))
    z4 = (d + B) * math.tan(math.radians(angle_frottement_diffusion / 2) + math.radians(180 / 4))
    if z2 < z3:
        if type_coefficient.upper() == 'A':
            z3_p = ((2 * B) * math.tan(math.radians(180 / 4 - angle_frottement_diffusion / 2))) / ka1 + z1 + z3 - z4
            if z3_p < z3:
                z3_p = z3

        else:
            z3_p = ((2 * B) * math.tan(math.radians(180 / 4 - angle_frottement_diffusion / 2))) / k0_1 + z1 + z3 - z4
            if z3_p < z3:
                z3_p = z3
    else:
        if type_coefficient.upper() == 'A':
            z3_p = ((2 * B) * math.tan(math.radians(180 / 4 - angle_frottement_diffusion / 2))) / ka1 + z1 + z2 - z4
            if z3_p < z2:
                z3_p = z2

        else:
            z3_p = ((2 * B) * math.tan(math.radians(180 / 4 - angle_frottement_diffusion / 2))) / k0_1 + z1 + z2 - z4
            if z3_p < z2:
                z3_p = z2

    global cote_z1, cote_z2, cote_z3, cote_z4
    cote_z1 = cote_TN - z1
    cote_z2 = cote_TN - z2
    cote_z3 = cote_TN - z3
    cote_z4 = cote_TN - z4
    cote_z3_p = cote_TN - z3_p

    if min(cote_z1, cote_z2, cote_z3, cote_z4) < cote_inf1:
        cote_inf1 = min(cote_z1, cote_z2, cote_z3, cote_z4)

    global sigma_max
    if z2 < z3:
        sigma_max = (2 * B * surcharge * math.tan(
            math.radians(180 / 4) - math.radians(angle_frottement_diffusion / 2))) / (
                            (z3 + z4) - (z1 + z2))
        # print('sigma max: ' + str(sigma_max))
    else:
        sigma_max = (2 * B * surcharge * math.tan(
            math.radians(180 / 4) - math.radians(angle_frottement_diffusion / 2))) / (z4 - z1)
        #print('sigma max: ' + str(sigma_max))

    if sigma_max < ka1 * surcharge:
        #print('z1: ', z1, 'z2: ', z2, 'z3: ', z3, 'z4: ', z4)

        # if cote_inf1 > z4_plot: # Cette condition permet d'imposer qu'on ait reellement tous les z sur la couche 1
        if z2 < z3:
            sigma_q_z1 = 0
            sigma_q_z2 = sigma_max
            sigma_q_z3 = sigma_max
            sigma_q_z4 = 0
        else:
            sigma_q_z1 = 0
            sigma_q_z3 = sigma_max
            sigma_q_z4 = 0
            sigma_q_z2 = ((sigma_q_z4 - sigma_q_z3) / (cote_z4 - cote_z3)) * (cote_z2 - cote_z3) + sigma_q_z3

        global gamma_sat1, gamma_sec1, cote_nappe
        if presence_nappe.upper() == "O":
            gamma_sat1 = float(e8.get())
            gamma_sec1 = float(e9.get())  # Poids volumique au dessus de la nappe
            gamma_eau = 10
            cote_nappe = float(e10.get())  # cette valeur doit etre évidemment inférieure à la cote du terrain naturel


        else:
            gamma_sat1 = 0
            gamma_sec1 = float(e9.get())  # Poids volumique au dessus de la nappe
            gamma_eau = 0
            cote_nappe = cote_inf1

        # if cote_nappe > cote_TN:
        #     print("La cote de la nappe doit etre inferieure à la cote du terrain naturel!!")

        # CALCUL DE LA CONTRAINTE TOTALE,EFFECTIVE ET PRESSION INTERSTITIELLE (TOP AND BOTTOM)
        sigma_v_sup1 = surcharge
        sigma_v_inf1 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_inf1) + surcharge

        pression_eau_sup1 = 0
        pression_eau_inf1 = gamma_eau * (cote_nappe - cote_inf1)

        sigma_p_sup1 = sigma_v_sup1 - pression_eau_sup1  # Le 'p' signifie que prime. IL s'agit de la contrainte effective
        sigma_p_inf1 = sigma_v_inf1 - pression_eau_inf1

        # CALCUL DE LA CONTRAINTE HORIZONTALE

        if type_coefficient.upper() == 'A':
            # Contrainte horizontale sans surcharge (At the top and bottom of the layer)
            sigma_h_sol_inf1 = (sigma_p_inf1 - surcharge) * ka1 - 2 * cohesion1 * (ka1) ** 0.5
            sigma_h_sol_sup1 = (sigma_p_sup1 - surcharge) * ka1 - 2 * cohesion1 * (ka1) ** 0.5

            sigma_h_total_inf1 = sigma_h_sol_inf1 + pression_eau_inf1
            sigma_h_total_sup1 = sigma_h_sol_sup1 + pression_eau_sup1
        else:
            sigma_h_sol_inf1 = (sigma_p_inf1 - surcharge) * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5
            sigma_h_sol_sup1 = (sigma_p_sup1 - surcharge) * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5

            sigma_h_total_inf1 = sigma_h_sol_inf1 + pression_eau_inf1
            sigma_h_total_sup1 = sigma_h_sol_sup1 + pression_eau_sup1

        # Contraintes totale, effective et U aux points caractéristiques de Krey sans surcharge

        if cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 > cote_nappe and cote_z4 > cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_z4)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z4 = sigma_v_z4

            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                # SHOW 2 PLOTS ON THE SAME FIGURE

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 > cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]



                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 < cote_nappe and cote_z3 < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

            figure = plt.figure()
            axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
            axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
            axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

            axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
            # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
            # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
            axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

            axes.set_xlabel('sigma_horizontal(kpa)')
            axes.set_ylabel('prof(m)')
            axes.legend(loc="upper right")
            axes.grid()

            y_plot_k_q = [cote_sup1, cote_inf1]
            x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

            axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
            axes2.patch.set_color('lightyellow')
            axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
            axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
            axes2.grid()
            axes2.legend(loc="lower right")
            plt.savefig('sigma.png')
            plt.show()


        elif cote_z1 > cote_nappe and cote_z2 < cote_nappe and cote_z3 > cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z3 = sigma_v_z3
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]



                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z2 = sigma_h_f_z3 + pression_eau_z2
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        else:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)
            pression_eau_z1 = gamma_eau * (cote_nappe - cote_z1)

            sigma_p_z1 = sigma_v_z1 - pression_eau_z1
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]
                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1 + pression_eau_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()



            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1 + pression_eau_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


    else:
        sigma_max = ka1 * surcharge
        # print('z1: ', z1, 'z2: ', z2, 'z3: ', z3, "z'3: ", z3_p, 'z4: ', z4)

        # if cote_inf1 > z4_plot: # Cette condition permet d'imposer qu'on ait reellement tous les z sur la couche 1
        if z2 < z3:
            sigma_q_z1 = 0
            sigma_q_z2 = sigma_max
            sigma_q_z3 = sigma_max
            sigma_q_z4 = 0
            sigma_q_z3_p = sigma_max
        else:
            sigma_q_z1 = 0
            sigma_q_z3 = sigma_max
            sigma_q_z4 = 0
            sigma_q_z2 = sigma_max
            sigma_q_z3_p = sigma_max

        if presence_nappe.upper() == "O":
            gamma_sat1 = float(e8.get())
            gamma_sec1 = float(e9.get())  # Poids volumique au dessus de la nappe
            gamma_eau = 10
            cote_nappe = float(e10.get())  # cette valeur doit etre évidemment inférieure à la cote du terrain naturel


        else:
            gamma_sat1 = 0
            gamma_sec1 = float(e9.get())  # Poids volumique au dessus de la nappe
            gamma_eau = 0
            cote_nappe = cote_inf1

        # if cote_nappe > cote_TN:
        #     print("La cote de la nappe doit etre inferieure à la cote du terrain naturel!!")

        # CALCUL DE LA CONTRAINTE TOTALE,EFFECTIVE ET PRESSION INTERSTITIELLE (TOP AND BOTTOM)
        sigma_v_sup1 = surcharge
        sigma_v_inf1 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_inf1) + surcharge

        pression_eau_sup1 = 0
        pression_eau_inf1 = gamma_eau * (cote_nappe - cote_inf1)

        sigma_p_sup1 = sigma_v_sup1 - pression_eau_sup1  # Le 'p' signifie que prime. IL s'agit de la contrainte effective
        sigma_p_inf1 = sigma_v_inf1 - pression_eau_inf1

        # CALCUL DE LA CONTRAINTE HORIZONTALE

        if type_coefficient.upper() == 'A':
            # Contrainte horizontale sans surcharge (At the top and bottom of the layer)
            sigma_h_sol_inf1 = (sigma_p_inf1 - surcharge) * ka1 - 2 * cohesion1 * (ka1) ** 0.5
            sigma_h_sol_sup1 = (sigma_p_sup1 - surcharge) * ka1 - 2 * cohesion1 * (ka1) ** 0.5

            sigma_h_total_inf1 = sigma_h_sol_inf1 + pression_eau_inf1
            sigma_h_total_sup1 = sigma_h_sol_sup1 + pression_eau_sup1
        else:
            sigma_h_sol_inf1 = (sigma_p_inf1 - surcharge) * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5
            sigma_h_sol_sup1 = (sigma_p_sup1 - surcharge) * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5

            sigma_h_total_inf1 = sigma_h_sol_inf1 + pression_eau_inf1
            sigma_h_total_sup1 = sigma_h_sol_sup1 + pression_eau_sup1

        # Contraintes totale, effective et U aux points caractéristiques de Krey sans surcharge

        if cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 > cote_nappe and cote_z3_p > cote_nappe and cote_z4 > cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_z4)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_z3_p)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z4 = sigma_v_z4
            sigma_p_z3_p = sigma_v_z3_p

            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4
                sigma_h_total_z3_p = sigma_h_f_z3_p

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z3_p, sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()

            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z4 = sigma_h_f_z4
                sigma_h_total_z3_p = sigma_h_f_z3_p

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3,
                                      sigma_h_total_z3_p, sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 > cote_nappe and cote_z3_p > cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z3_p = sigma_v_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()

        elif cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 > cote_nappe and cote_z3_p < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3_p = gamma_eau * (cote_nappe - cote_z3_p)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z3_p = sigma_v_z3_p - pression_eau_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()

            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 > cote_nappe and cote_z3 < cote_nappe and cote_z3_p < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3_p = gamma_eau * (cote_nappe - cote_z3_p)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z3_p = sigma_v_z3_p - pression_eau_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]



                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 < cote_nappe and cote_z3 > cote_nappe and cote_z3_p < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3_p = gamma_eau * (cote_nappe - cote_z3_p)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z3 = sigma_v_z3
            sigma_p_z3_p = sigma_v_z3_p - pression_eau_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]


                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()

            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        elif cote_z1 > cote_nappe and cote_z2 < cote_nappe and cote_z3 < cote_nappe and cote_z3_p < cote_nappe and cote_z4 < cote_nappe:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_nappe)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3_p = gamma_eau * (cote_nappe - cote_z3_p)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)

            sigma_p_z1 = sigma_v_z1
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z3_p = sigma_v_z3_p - pression_eau_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]



                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


        else:
            sigma_v_z1 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z1)
            sigma_v_z2 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z2)
            sigma_v_z3 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3)
            sigma_v_z3_p = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z3_p)
            sigma_v_z4 = gamma_sec1 * (cote_sup1 - cote_nappe) + gamma_sat1 * (cote_nappe - cote_z4)

            pression_eau_z4 = gamma_eau * (cote_nappe - cote_z4)
            pression_eau_z3_p = gamma_eau * (cote_nappe - cote_z3_p)
            pression_eau_z3 = gamma_eau * (cote_nappe - cote_z3)
            pression_eau_z2 = gamma_eau * (cote_nappe - cote_z2)
            pression_eau_z1 = gamma_eau * (cote_nappe - cote_z1)

            sigma_p_z1 = sigma_v_z1 - pression_eau_z1
            sigma_p_z2 = sigma_v_z2 - pression_eau_z2
            sigma_p_z3 = sigma_v_z3 - pression_eau_z3
            sigma_p_z3_p = sigma_v_z3_p - pression_eau_z3_p
            sigma_p_z4 = sigma_v_z4 - pression_eau_z4

            # contrainte horizontale aux points caractéristiques de Krey (ne prend pas en compte la poussée des eaux)
            if type_coefficient.upper() == 'A':
                sigma_h_f_z1 = sigma_p_z1 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * ka1 - 2 * cohesion1 * (ka1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1 + pression_eau_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()


            else:
                sigma_h_f_z1 = sigma_p_z1 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z1
                sigma_h_f_z2 = sigma_p_z2 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z2
                sigma_h_f_z3 = sigma_p_z3 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3
                sigma_h_f_z3_p = sigma_p_z3_p * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z3_p
                sigma_h_f_z4 = sigma_p_z4 * k0_1 - 2 * cohesion1 * (k0_1) ** 0.5 + sigma_q_z4

                if z2 < z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z2, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]

                if z2 > z3:
                    y_plot_h_f = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_f = [sigma_h_sol_sup1, sigma_h_f_z1, sigma_h_f_z3, sigma_h_f_z3_p, sigma_h_f_z4,
                                  sigma_h_sol_inf1]


                y_plot_v_f = [cote_sup1, cote_inf1]
                x_plot_v_f = [sigma_v_sup1, sigma_v_inf1]

                y_plot_p_f = [cote_sup1, cote_inf1]
                x_plot_p_f = [sigma_p_sup1, sigma_p_inf1]

                y_plot_u_f = [cote_nappe, cote_inf1]
                x_plot_u_f = [pression_eau_sup1, pression_eau_inf1]

                y_plot_scatter_z = [cote_z1, cote_z2, cote_z3, cote_z3_p, cote_z4]
                x_plot_scatter_z = [0, 0, 0, 0, 0]

                # sommes de toutes les contraintes horizontales (earth pressure + pore water pressure)

                sigma_h_total_z1 = sigma_h_f_z1 + pression_eau_z1
                sigma_h_total_z2 = sigma_h_f_z2 + pression_eau_z2
                sigma_h_total_z3 = sigma_h_f_z3 + pression_eau_z3
                sigma_h_total_z3_p = sigma_h_f_z3_p + pression_eau_z3_p
                sigma_h_total_z4 = sigma_h_f_z4 + pression_eau_z4

                if z2 < z3:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z2, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z2, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                else:
                    y_plot_h_total = [cote_sup1, cote_z1, cote_z3, cote_z3_p, cote_z4, cote_inf1]
                    x_plot_h_total = [sigma_h_total_sup1, sigma_h_total_z1, sigma_h_total_z3, sigma_h_total_z3_p,
                                      sigma_h_total_z4, sigma_h_total_inf1]

                if z2 < z3:
                    y_plot_h_q = [cote_z1, cote_z2, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z2, sigma_q_z3_p, sigma_q_z4]
                else:
                    y_plot_h_q = [cote_z1, cote_z3, cote_z3_p, cote_z4]
                    x_plot_h_q = [sigma_q_z1, sigma_q_z3, sigma_q_z3_p, sigma_q_z4]

                figure = plt.figure()
                axes = figure.add_subplot(111)  # Renvoie un objet AxesSubplot, sous classe de Axes
                axes.plot(x_plot_h_total, y_plot_h_total, 'r', label='H_total')
                axes.scatter(x_plot_scatter_z, y_plot_scatter_z, label="z1, z2, z3, z4")

                axes.plot(x_plot_h_f, y_plot_h_f, label='sigma_H', linestyle='--')
                # plt.plot(x_plot_v_f, y_plot_v_f, linestyle='--', label='Contrainte verticale')
                # plt.plot(x_plot_p_f, y_plot_p_f, linestyle='--', label='Contrainte effective')
                axes.plot(x_plot_u_f, y_plot_u_f, 'b', linestyle='--', label='U')

                axes.set_xlabel('sigma_horizontal(kpa)')
                axes.set_ylabel('prof(m)')
                axes.legend(loc="upper right")
                axes.grid()

                y_plot_k_q = [cote_sup1, cote_inf1]
                x_plot_k_q = [ka1 * surcharge, ka1 * surcharge]

                axes2 = figure.add_axes([0.35, 0.7, 0.3, 0.3])  # renvoie un objet Axes
                axes2.patch.set_color('lightyellow')
                axes2.plot(x_plot_h_q, y_plot_h_q, linestyle='--', label="Krey")
                axes2.plot(x_plot_k_q, y_plot_k_q, label="k*q")
                axes2.grid()
                axes2.legend(loc="lower right")
                plt.savefig('sigma.png')
                plt.show()

# PARTIE DU RAPPORT (ReportLab)

def hello(c):
    try:
        c.drawString(520, 825, "sigma v0.3")

        c.drawInlineImage("logo_Cerema.jpg", 250, 810, width=100, height=20)
        c.line(100, 805, 495, 805)

        # c.setFont('Cambria', 14)
        c.drawString(50, 780, "HYPOTHESES:")
        c.line(50, 778, 135, 778)

        red50transparent = Color(100, 0, 0, alpha=0.5)
        c.setFillColor(red50transparent)
        c.circle(150,  784, 3, fill=1)
        c.setFillColor(black)
        c.drawString(160, 780, "Diffusion de Krey (NF P 94-282 Annexe D.3.2)")

        red50transparent = Color(100, 0, 0, alpha=0.5)
        c.setFillColor(red50transparent)
        c.circle(150, 764, 3, fill=1)
        c.setFillColor(black)
        c.drawString(160, 760, "Terrain homogène")

        red50transparent = Color(100, 0, 0, alpha=0.5)
        c.setFillColor(red50transparent)
        c.circle(150, 744, 3, fill=1)
        c.setFillColor(black)
        c.drawString(160, 740, "Conditions drainées")

        # Draw a table for data
        c.drawString(50, 710, "DONNEES:")
        c.line(50, 708, 112, 708)

        if presence_nappe == 'O':
            data = [
                ['cote_TN (m)', 'cote nappe (m)', 'e (m)', 'phi (°)', "c'(kpa)", 'g_sat (kN/m3) ', 'g_sec (kN/m3)', 'q (kpa)', 'd (m)', 'B (m)'],
                [cote_TN, cote_nappe, epaisseur1, round(math.degrees(angle_frottement1), 2), cohesion1, gamma_sat1, gamma_sec1, surcharge, d, B]
            ]

        else:
            data = [
                ['cote_TN (m)', 'e (m)', 'phi (°)', "c'(kpa)", 'g_sat (kN/m3)', 'g_sec (kN/m3)', 'q (kpa)', 'd (m)','B (m)'],
                [cote_TN, epaisseur1, round(math.degrees(angle_frottement1), 2), cohesion1, gamma_sat1, gamma_sec1, surcharge, d, B]
            ]

        # Draw a table for data
        c.drawString(50, 640, "RESULTATS:")
        c.line(50, 638, 122, 638)

        list_z = ['contrainte (kpa)', 'Cote_TN', 'z1', 'z2', 'z3', 'z4', 'cote_inf']
        list_h_f = ['effective_h_sol']
        list_h_total = ['Horizontale_totale']

        for j in x_plot_h_f:
            list_h_f.append(round(j, 2))

        for i in x_plot_h_total:
                list_h_total.append(round(i, 2))

        results = [list_z, list_h_f, list_h_total]

        t2 = Table(results)

        # 3) Add borders
        ts = TableStyle([
                ('BOX', (0, 0), (-1, -1), 2, colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])

        t1 = Table(data)
        t1.setStyle(ts)
        t1.wrapOn(c, 450, 300)
        t1.drawOn(c, 30, 665)

        t2.setStyle(ts)
        t2.wrapOn(c, 450, 300)
        t2.drawOn(c, 30, 575)

        # Drawing the graph
        c.drawInlineImage("sigma.png", 275, 150, width=320, height=400)

    #     Draw the Krey distribution scheme
        c.line(50, 192, 50, 488) #repere vertical
        c.line(260, 488, 50, 488) #repere horizontal

        c.line(240, 488, 240, 510) # Cotation verticale 1, de reference
        c.line(50, 488, 50, 510)


        c.line(240 - B * 190/ (d + B), 505, 240, 505)
        c.line(240 - B * 190 / (d + B) , 488,240 - B * 190 / (d + B), 510) # ligne de séparation verticale de d et B
        c.line(50,505, d * 190 / (d + B) + 50,505)

        c.drawString(240 - B * 190 / (d + B)+5, 511, f"B = {B} m") # Ecrire B
        c.drawString( 55, 511, f"d = {d} m")  # Ecrire d

        # creer les petites flèches pour représenter la surcharge
        i = 240
        while i > 240 - B * 190/ (d + B):
            i -= 8
            c.line(i, 488, i, 501)

        # Tracer la diffusion de Krey
        c.line(50, 488 - 296 * (cote_sup1 - cote_z1) /(cote_sup1 -cote_inf1) , 240 - B * 190 / (d + B),  488)
        c.line(50, 488 - 296 * (cote_sup1 - cote_z2) / (cote_sup1 -cote_inf1), 240 - B * 190 / (d + B), 488)
        c.line(50, 488 - 296 * (cote_sup1 - cote_z3) / (cote_sup1 -cote_inf1), 240, 488)
        c.line(50, 488 - 296 * (cote_sup1 - cote_z4) / (cote_sup1 -cote_inf1), 240, 488)

        c.drawString(0, 488 - 296 * (cote_sup1 - cote_z1) /(cote_sup1 -cote_inf1), f"z1 = {round(y_plot_h_f[1],1)}")
        c.drawString(0, 488 - 296 * (cote_sup1 - cote_z2) / (cote_sup1 - cote_inf1), f"z2 = {round(y_plot_h_f[2],1)}")
        c.drawString(0, 488 - 296 * (cote_sup1 - cote_z3) / (cote_sup1 - cote_inf1), f"z3 = {round(y_plot_h_f[3],1)}")
        c.drawString(0, 488 - 296 * (cote_sup1 - cote_z4) / (cote_sup1 - cote_inf1), f"z4 = {round(y_plot_h_f[4],1)}")

        c.showPage()
        c.save()
    except NameError:
        c.drawString(520, 825, "sigma v0.3")

        c.drawInlineImage("logo_Cerema.jpg", 250, 810, width=100, height=20)
        c.line(100, 805, 495, 805)
        c.drawString(50, 600, "Erreur: Soit vous n'avez pas fourni toutes les données ou vous n'avez pas encore calculé")

        c.showPage()
        c.save()


def rapport():
    c = canvas.Canvas("Note de calcul.pdf", pagesize=(595.27, 841.89))
    hello(c)

def browse_excel():
    # #Je n'ai finalement plus utilisé pd dans cette section car xlsxwriter s'est avéré etre meilleur pour travailler sur excel
    ##file = filedialog.askopenfile(parent=root, mode='rb', title='choose a file')
    #df1 = pd.DataFrame({'cote_TN': [round(y_plot_h_total[0],2), round(x_plot_h_f[0],2),  0, round(x_plot_h_total[0],2)],
                       #'z1': [round(y_plot_h_total[1],2), round(x_plot_h_f[1],2),  round(x_plot_h_q[0],2), round(x_plot_h_total[1],2)],
                       #'z2': [round(y_plot_h_total[2],2), round(x_plot_h_f[2],2),  round(x_plot_h_q[1],2), round(x_plot_h_total[2],2)],
                       #'z3': [round(y_plot_h_total[3],2), round(x_plot_h_f[3],2),  round(x_plot_h_q[2],2), round(x_plot_h_total[3],2)],
                       #'z4': [round(y_plot_h_total[4],2), round(x_plot_h_f[4],2),  round(x_plot_h_q[3],2), round(x_plot_h_total[4],2)],
                       #'cote_inf': [round(y_plot_h_total[5],2), round(x_plot_h_f[5],2),  0, round(x_plot_h_total[5],2)]
                       #})
    #index_ = ['cote (m)', 'effective_h_sol(kpa)', 'increment charge_q(kpa)', 'Horizontale totale (kpa)']
    #df1.index = index_
    #df1.loc['cote (m)', 'Name'] = 'Dimitri'

    if presence_nappe == 'O':
        df2 = pd.DataFrame([{'cote_TN (m)':cote_TN,
                            'cote nappe (m)':cote_nappe,
                            'e (m)':epaisseur1,
                            'phi (°)':round(math.degrees(angle_frottement1)),
                            "c'(kpa)": cohesion1,
                            'g_sat (kN/m3) ':gamma_sat1,
                            'g_sec (kN/m3)':gamma_sec1,
                            'q (kpa)':surcharge,
                            'd (m)':d,
                            'B (m)': B
                            }])

    else:
        df2 = pd.DataFrame([{'cote_TN (m)': cote_TN,
                            'e (m)': epaisseur1,
                            'phi (°)': round(math.degrees(angle_frottement1)),
                            "c'(kpa)": cohesion1,
                            'g_sat (kN/m3) ': gamma_sat1,
                            'g_sec (kN/m3)': gamma_sec1,
                            'q (kpa)': surcharge,
                            'd (m)': d,
                            'B (m)': B
                            }])

    # DANS CETTE SECTION  J'AI PREFERE TRAVAILLER AVEC Xlsxwriter qui semble plus interéssant que pandas pour exporter les données vers excel
    path = 'blindage - Copie.xlsx'
    workbook = xlsxwriter.Workbook(path)
    sheet_name1 = workbook.add_worksheet()
    sheet_name2 = workbook.add_worksheet()
    #sheet_name1 = 'Feuil1'
    #sheet_name2 = 'Sheet1'
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    #df1.to_excel(writer, sheet_name= sheet_name1, startcol=0, startrow=16, index=True)

    #df1.to_excel(writer, sheet_name=sheet_name2, startcol=0, startrow=16, index=True)
    #df2.to_excel(writer, sheet_name= sheet_name2,startcol=0, startrow=12, index=False)

    # Add image logo

    #url = 'logo_Cerema.jpg'
    #sheet_name1.insert_image('A1', url, {'image_data': image_data})

    if sigma_max < ka1*surcharge:
        if z2 < z3:

            headings_results = ['cote (m)',
                        'effective_h_sol(kpa)',
                        'increment charge_q(kpa)',
                        'Horizontale totale (kpa)']
            index_results = ['cote_TN', 'z1', 'z2', 'z3', 'z4', 'cote_inf']

            data_results = [
                [round(y_plot_h_total[0], 2), round(y_plot_h_total[1],2), round(y_plot_h_total[2], 2), round(y_plot_h_total[3], 2), round(y_plot_h_total[4], 2), round(y_plot_h_total[5], 2)],
                [round(x_plot_h_f[0], 2), round(x_plot_h_f[1], 2), round(x_plot_h_f[2], 2), round(x_plot_h_f[3], 2), round(x_plot_h_f[4], 2), round(x_plot_h_f[5], 2)],
                [0, round(x_plot_h_q[0], 2), round(x_plot_h_q[1],2), round(x_plot_h_q[2], 2), round(x_plot_h_q[3], 2), 0],
                [round(x_plot_h_total[0], 2), round(x_plot_h_total[1], 2), round(x_plot_h_total[2], 2), round(x_plot_h_total[3], 2), round(x_plot_h_total[4], 2), round(x_plot_h_total[5], 2)]
            ]

        else:
            headings_results = ['cote (m)',
                                'effective_h_sol(kpa)',
                                'increment charge_q(kpa)',
                                'Horizontale totale (kpa)']
            index_results = ['cote_TN', 'z1', 'z3', 'z4', 'cote_inf']

            data_results = [
                [round(y_plot_h_total[0], 2), round(y_plot_h_total[1], 2), round(y_plot_h_total[2], 2),round(y_plot_h_total[3], 2), round(y_plot_h_total[4], 2)],
                [round(x_plot_h_f[0], 2), round(x_plot_h_f[1], 2), round(x_plot_h_f[2], 2), round(x_plot_h_f[3], 2),round(x_plot_h_f[4], 2)],
                [0, round(x_plot_h_q[0], 2), round(x_plot_h_q[1], 2), round(x_plot_h_q[2], 2), 0],
                [round(x_plot_h_total[0], 2), round(x_plot_h_total[1], 2), round(x_plot_h_total[2], 2),round(x_plot_h_total[3], 2), round(x_plot_h_total[4], 2)]
            ]

    else:
        if z2 < z3:

            headings_results = ['cote (m)',
                                'effective_h_sol(kpa)',
                                'increment charge_q(kpa)',
                                'Horizontale totale (kpa)']
            index_results = ['cote_TN', 'z1', 'z2', 'z3_p', 'z4', 'cote_inf']

            data_results = [
                [round(y_plot_h_total[0], 2), round(y_plot_h_total[1],2), round(y_plot_h_total[2], 2), round(y_plot_h_total[3], 2), round(y_plot_h_total[4], 2), round(y_plot_h_total[5], 2)],
                [round(x_plot_h_f[0], 2), round(x_plot_h_f[1], 2), round(x_plot_h_f[2], 2), round(x_plot_h_f[3], 2), round(x_plot_h_f[4], 2), round(x_plot_h_f[5], 2)],
                [0, round(x_plot_h_q[0], 2), round(x_plot_h_q[1],2), round(x_plot_h_q[2], 2), round(x_plot_h_q[3], 2), 0],
                [round(x_plot_h_total[0], 2), round(x_plot_h_total[1], 2), round(x_plot_h_total[2], 2), round(x_plot_h_total[3], 2), round(x_plot_h_total[4], 2), round(x_plot_h_total[5], 2)]
            ]

        else:
            headings_results = ['cote (m)',
                                'effective_h_sol(kpa)',
                                'increment charge_q(kpa)',
                                'Horizontale totale (kpa)']
            index_results = ['cote_TN', 'z1', 'z3','z3_p', 'z4', 'cote_inf']

            data_results = [
                [round(y_plot_h_total[0], 2), round(y_plot_h_total[1], 2), round(y_plot_h_total[2], 2),round(y_plot_h_total[3], 2), round(y_plot_h_total[4], 2), round(y_plot_h_total[5],2)],
                [round(x_plot_h_f[0], 2), round(x_plot_h_f[1], 2), round(x_plot_h_f[2], 2), round(x_plot_h_f[3], 2),round(x_plot_h_f[4], 2), round(x_plot_h_f[5], 2)],
                [0, round(x_plot_h_q[0], 2), round(x_plot_h_q[1], 2), round(x_plot_h_q[2], 2), round(x_plot_h_q[3], 2),0],
                [round(x_plot_h_total[0], 2), round(x_plot_h_total[1], 2), round(x_plot_h_total[2], 2),round(x_plot_h_total[3], 2), round(x_plot_h_total[4], 2), round(x_plot_h_total[5], 2)]
            ]

    bold = workbook.add_format({'bold': 1})

    sheet_name1.write_row('B10', headings_results, bold)
    sheet_name1.write_column('A11', index_results)
    sheet_name1.write_column('B11', data_results[0])
    sheet_name1.write_column('C11', data_results[1])
    sheet_name1.write_column('D11', data_results[2])
    sheet_name1.write_column('E11', data_results[3])


    # s'il y'a la nappe alors on affiche les données dans excel avec la cote de la nappe, sinon on ne l'affiche pas
    if presence_nappe == 'O':
        headings_data = [   'cote_TN (m)',
                            'cote nappe (m)',
                            'e (m)',
                            'φ (°)',
                            "c'(kpa)",
                            'γ_sat (kN/m3) ',
                            'γ_sec (kN/m3)',
                            'q (kpa)',
                            'd (m) ',
                            'B (m)'
                            ]

        data_data = [
                    [cote_TN],
                    [cote_nappe],
                    [epaisseur1],
                    [round(math.degrees(angle_frottement1))],
                    [cohesion1],
                    [gamma_sat1],
                    [gamma_sec1],
                    [surcharge],
                    [d],
                    [B]
                    ]
    else:
        headings_data = [   'cote_TN (m)',
                            'e (m)',
                            'φ (°)',
                            "c'(kpa)",
                            'γ_sat (kN/m3) ',
                            'γ_sec (kN/m3)',
                            'q (kpa)',
                            'd (m) ',
                            'B (m)'
                            ]

        data_data = [
                    [cote_TN],
                    [epaisseur1],
                    [round(math.degrees(angle_frottement1))],
                    [cohesion1],
                    [gamma_sat1],
                    [gamma_sec1],
                    [surcharge],
                    [d],
                    [B]
                    ]



    # Export des données puis des résultats vers excel

    sheet_name1.write_row('A6', headings_data, bold)
    sheet_name1.write_column('A7', data_data[0])
    sheet_name1.write_column('B7', data_data[1])
    sheet_name1.write_column('C7', data_data[2])
    sheet_name1.write_column('D7', data_data[3])
    sheet_name1.write_column('E7', data_data[4])
    sheet_name1.write_column('F7', data_data[5])
    sheet_name1.write_column('G7', data_data[6])
    sheet_name1.write_column('H7', data_data[7])
    sheet_name1.write_column('I7', data_data[8])
    sheet_name1.write_column('J7', data_data[9])


    # create a chart
    chart1 = workbook.add_chart({'type': 'line'})

    # Adding first serie 1

    """chart1.add_series({
        'name': '=Sheet1!$D$10',
        'categories': '=Sheet1!$B$11:$B$16',
        'values': '=Sheet1!$D$11:$D$16',
    })"""

    chart1 = workbook.add_chart({'type': 'line'})
    chart1.add_series({
        'name': ['Sheet1', 9, 2],
        'categories': ['Sheet1', 10, 1, 15, 1],
        'values': ['Sheet1', 10, 2, 15, 2],
    })

    chart2 = workbook.add_chart({'type': 'line'})
    chart2.add_series({
        'name': ['Sheet1', 9, 4],
        'categories': ['Sheet1', 10, 1, 15, 1],
        'values': ['Sheet1', 10, 4, 15, 4],
    })

    chart1.combine(chart2)


    sheet_name1.insert_chart('A19', chart1, {'x_offset': 10, 'y_offset': 10})
    workbook.close()
    # # Access the Xlsxwriter workbook and worksheet objects from the dataframe
    # workbook = writer.book
    # worksheet = writer.sheets[sheet_name1]
    #
    # data = [x_plot_h_total, y_plot_h_total]
    # worksheet.write_column('A1', data[0])
    # #worksheet.write_column('B1', data[1])
    #
    # # Create a chart object.
    # chart = workbook.add_chart({'type': 'line'})
    # chart.add_series({
    #     'values': '=$A$1:$A$5'
    # })
    #
    # worksheet.insert_chart('C1', chart)
    #
    # # writer.save()
    # workbook.close()




    #df1 = pd.read_excel(file)
    # df1 = pd.read_excel(file, sheet_name=1)
    #
    # cote_TN = df1['cote_TN (m)'][0]



calcul = ttk.Button(root, text=f"Calculer", command=calculer).grid(row=13, column=1)
report = ttk.Button(root, text=f"Rapport", command=rapport).grid(row=13, column=2)
browse = ttk.Button(root, text=f"Exporter xls", command=browse_excel).grid(row=13, column=0)


root.mainloop()





