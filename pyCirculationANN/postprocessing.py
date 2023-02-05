import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def plot_PV_loops(history, axes = None):
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(10,10))

    axes[0, 1].plot(history.V_RA, history.p_RA, 'r'), axes[0, 1].set_title(r'Right atrium')
    axes[1, 1].plot(history.V_RV, history.p_RV, 'r'), axes[1, 1].set_title(r'Right ventricle')
    axes[0, 0].plot(history.V_LA, history.p_LA, 'r'), axes[0, 0].set_title(r'Left atrium')
    axes[1, 0].plot(history.V_LV, history.p_LV, 'r'), axes[1, 0].set_title(r'Left ventricle')
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel('Volume [mL]')
            axes[i,j].set_ylabel('Pressure [mmHg]')

def plot_elastances(history, axes = None):
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(10,10))

    axes[0, 1].plot(history.t, history.E_RA), axes[0, 1].set_title(r'Right atrium')
    axes[1, 1].plot(history.t, history.E_RV), axes[1, 1].set_title(r'Right ventricle')
    axes[0, 0].plot(history.t, history.E_LA), axes[0, 0].set_title(r'Left atrium')
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel('Time [s]')
            axes[i,j].set_ylabel('Elastance [mmHg/mL]')

def plot_history(history, axes = None):
    if axes is None:
        _, axes = plt.subplots(3, 2, figsize=(10,10))

    ax = axes[0,1]
    ax.plot(history.t, history.p_VEN_SYS, 'k', label = r'$p_\mathrm{VEN, SYS}$')
    ax.plot(history.t, history.p_RA     , 'b', label = r'$p_\mathrm{RA}$')
    ax.plot(history.t, history.p_RV     , 'g', label = r'$p_\mathrm{RV}$')
    ax.plot(history.t, history.p_AR_PUL , 'r', label = r'$p_\mathrm{AR, PUL}$')
    ax.plot(history.t, history.p_VEN_PUL, 'c', label = r'$p_\mathrm{VEN, PUL}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Pressure [mmHg]')

    ax = axes[0,0]
    ax.plot(history.t, history.p_VEN_PUL, 'k', label = r'$p_\mathrm{VEN, PUL}$')
    ax.plot(history.t, history.p_LA     , 'b', label = r'$p_\mathrm{LA}$')
    ax.plot(history.t, history.p_LV     , 'g', label = r'$p_\mathrm{LV}$')
    ax.plot(history.t, history.p_AR_SYS , 'r', label = r'$p_\mathrm{AR, SYS}$')
    ax.plot(history.t, history.p_VEN_SYS, 'c', label = r'$p_\mathrm{VEN, SYS}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Pressure [mmHg]')

    ax = axes[1,1]
    ax.plot(history.t, history.V_RA     , 'k', label = r'$V_\mathrm{RA}$')
    ax.plot(history.t, history.V_RV     , 'b', label = r'$V_\mathrm{RV}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Volume [mL]')

    ax = axes[1,0]
    ax.plot(history.t, history.V_LA     , 'k', label = r'$V_\mathrm{LA}$')
    ax.plot(history.t, history.V_LV     , 'b', label = r'$V_\mathrm{LV}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Volume [mL]')

    ax = axes[2,1]
    ax.plot(history.t, history.Q_VEN_SYS, 'k', label = r'$Q_\mathrm{VEN, SYS}$')
    ax.plot(history.t, history.Q_TV     , 'b', label = r'$Q_\mathrm{TV}$')
    ax.plot(history.t, history.Q_PV     , 'g', label = r'$Q_\mathrm{PV}$')
    ax.plot(history.t, history.Q_AR_PUL , 'r', label = r'$Q_\mathrm{AR, PUL}$')
    ax.plot(history.t, history.Q_VEN_PUL, 'c', label = r'$Q_\mathrm{VEN, PUL}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Flux [mL/s]')

    ax = axes[2,0]
    ax.plot(history.t, history.Q_VEN_PUL, 'k', label = r'$Q_\mathrm{VEN, PUL}$')
    ax.plot(history.t, history.Q_MV     , 'b', label = r'$Q_\mathrm{MV}$')
    ax.plot(history.t, history.Q_AV     , 'g', label = r'$Q_\mathrm{AV}$')
    ax.plot(history.t, history.Q_AR_SYS , 'r', label = r'$Q_\mathrm{AR, SYS}$')
    ax.plot(history.t, history.Q_VEN_SYS, 'c', label = r'$Q_\mathrm{VEN, SYS}$')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(r'Flux [mL/s]')