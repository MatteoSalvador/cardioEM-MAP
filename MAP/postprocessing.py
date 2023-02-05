import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import numpy as np

def plot_elastances_MAP(PE, axes = None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))

    fig.suptitle(r'Iteration %d' % PE.iteration, fontsize=16)

    axes[0, 1].plot(PE.times_last_cycle, PE.E_RA(PE.times_last_cycle)), axes[0, 1].set_title(r'Right atrium')
    axes[1, 1].plot(PE.times_last_cycle, PE.E_RV(PE.times_last_cycle)), axes[1, 1].set_title(r'Right ventricle')
    axes[0, 0].plot(PE.times_last_cycle, PE.E_LA(PE.times_last_cycle)), axes[0, 0].set_title(r'Left atrium')
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel(r'Time [s]')
            axes[i,j].set_ylabel(r'Elastance [mmHg/mL]')

def plot_history_MAP(PE, axes = None):
    if axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(10,15))

    fig.suptitle(r'Iteration %d' % PE.iteration, fontsize=16)

    ax = axes[0,1]
    ax.plot(PE.times_last_cycle, PE.p_VEN_SYS, 'k--', label = r'$p_\mathrm{VEN, SYS}$')
    ax.plot(PE.times_last_cycle, PE.p_RA, 'b--', label = r'$p_{\mathrm{RA}}$ (MAP)')
    ax.plot(PE.times_last_cycle, PE.p_RV, 'g--', label = r'$p_\mathrm{RV}$ (MAP)')
    ax.plot(PE.times_last_cycle, PE.p_AR_PUL , 'r--', label = r'$p_\mathrm{AR, PUL}$')
    ax.plot(PE.times_last_cycle, PE.p_VEN_PUL, 'c--', label = r'$p_\mathrm{VEN, PUL}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Pressure [mmHg]')

    ax = axes[0,0]
    ax.plot(PE.times_last_cycle, PE.p_VEN_PUL, 'k--', label = r'$p_\mathrm{VEN, PUL}$')
    ax.plot(PE.times_last_cycle, PE.p_LA, 'b--', label = r'$p_\mathrm{LA}$ (MAP)')
    ax.plot(PE.times_last_cycle, PE.p_LV, 'g--', label = r'$p_\mathrm{LV}$ (MAP)')
    ax.plot(PE.times_last_cycle, PE.p_AR_SYS, 'r--', label = r'$p_\mathrm{AR, SYS}$ (MAP)')
    if (PE.weight_p_AR_SYS >= 1e-15):
        ax.plot(PE.times_last_cycle, PE.p_AR_SYS_opt, 'r', label = r'$p_\mathrm{AR, SYS}$ (target)')
    ax.plot(PE.times_last_cycle, PE.p_VEN_SYS, 'c--', label = r'$p_\mathrm{VEN, SYS}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Pressure [mmHg]')

    ax = axes[1,1]
    ax.plot(PE.times_last_cycle, PE.V_RA, 'k--', label = r'$V_\mathrm{RA}$ (MAP)')
    if (PE.weight_V_RA >= 1e-15):
        ax.plot(PE.times_last_cycle, PE.V_RA_opt, 'k', label = r'$V_\mathrm{RA}$ (target)')
    ax.plot(PE.times_last_cycle, PE.V_RV, 'b--', label = r'$V_\mathrm{RV}$ (MAP)')
    if (PE.weight_V_RV >= 1e-15):
        ax.plot(PE.times_last_cycle, PE.V_RV_opt, 'b', label = r'$V_\mathrm{RV}$ (target)')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Volume [mL]')

    ax = axes[1,0]   
    ax.plot(PE.times_last_cycle, PE.V_LA, 'k--', label = r'$V_\mathrm{LA}$ (MAP)')
    if (PE.weight_V_LA >= 1e-15):
        ax.plot(PE.times_last_cycle, PE.V_LA_opt, 'k', label = r'$V_\mathrm{LA}$ (target)')
    ax.plot(PE.times_last_cycle, PE.V_LV, 'b--', label = r'$V_\mathrm{LV}$ (MAP)')
    if (PE.weight_V_LV >= 1e-15):
        ax.plot(PE.times_last_cycle, PE.V_LV_opt, 'b', label = r'$V_\mathrm{LV}$ (target)')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Volume [mL]')

    ax = axes[2,1]
    ax.plot(PE.times_last_cycle, PE.Q_VEN_SYS, 'k--', label = r'$Q_\mathrm{VEN, SYS}$')
    ax.plot(PE.times_last_cycle, PE.Q_TV     , 'b--', label = r'$Q_\mathrm{TV}$')
    ax.plot(PE.times_last_cycle, PE.Q_PV     , 'g--', label = r'$Q_\mathrm{PV}$')
    ax.plot(PE.times_last_cycle, PE.Q_AR_PUL , 'r--', label = r'$Q_\mathrm{AR, PUL}$')
    ax.plot(PE.times_last_cycle, PE.Q_VEN_PUL, 'c--', label = r'$Q_\mathrm{VEN, PUL}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Flux [mL/s]')

    ax = axes[2,0]
    ax.plot(PE.times_last_cycle, PE.Q_VEN_PUL, 'k--', label = r'$Q_\mathrm{VEN, PUL}$')
    ax.plot(PE.times_last_cycle, PE.Q_MV     , 'b--', label = r'$Q_\mathrm{MV}$')
    ax.plot(PE.times_last_cycle, PE.Q_AV     , 'g--', label = r'$Q_\mathrm{AV}$')
    ax.plot(PE.times_last_cycle, PE.Q_AR_SYS , 'r--', label = r'$Q_\mathrm{AR, SYS}$')
    ax.plot(PE.times_last_cycle, PE.Q_VEN_SYS, 'c--', label = r'$Q_\mathrm{VEN, SYS}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Flux [mL/s]')

def plot_PV_loops_MAP(PE, axes = None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))

    fig.suptitle(r'Iteration %d' % PE.iteration, fontsize=16)

    axes[0, 1].plot(PE.V_RA, PE.p_RA, 'k--', label = r'MAP'), axes[0, 1].set_title(r'Right atrium')
    axes[0, 1].legend(loc = 'upper right')                

    axes[1, 1].plot(PE.V_RV, PE.p_RV, 'k--', label = r'MAP'), axes[1, 1].set_title(r'Right ventricle')
    axes[1, 1].legend(loc = 'upper right')                

    axes[0, 0].plot(PE.V_LA, PE.p_LA, 'k--', label = r'MAP'), axes[0, 0].set_title(r'Left atrium')
    axes[0, 0].legend(loc = 'upper right')                

    axes[1, 0].plot(PE.V_LV, PE.p_LV, 'k--', label = r'MAP'), axes[1, 0].set_title(r'Left ventricle')
    axes[1, 0].legend(loc = 'upper right')                

    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel(r'Volume [mL]')
            axes[i,j].set_ylabel(r'Pressure [mmHg]')