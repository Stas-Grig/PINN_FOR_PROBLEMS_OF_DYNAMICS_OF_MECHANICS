import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def plotting_contour(x_y, func, flag=0, graph_name=None):
    show = True
    flags_meaning = {0: "$u_{x}$", 1: "$u_{y}$", 2: r"$\sigma_{xx}$", 3: r"$\sigma_{yy}$", 4: r"$\sigma_{xy}$"}
    x_coord = x_y[:, 0].detach().numpy()
    y_coord = x_y[:, 1].detach().numpy()
    func_coord = func[:, flag].detach().numpy()
    xi = np.linspace(np.min(x_coord), np.max(x_coord), 1_000)
    yi = np.linspace(np.min(y_coord), np.max(y_coord), 1_000)
    zi = griddata((x_coord, y_coord), func_coord, (xi[None, :], yi[:, None]), method='cubic')

    if not graph_name:
        graph_name = flags_meaning[flag]
        show = False
    elif graph_name == "FEM - PINN":
        graph_name = r"$\vert$" + flags_meaning[flag] + "$^{FEM}$ - " + flags_meaning[flag] + "$^{PINN}$" + r"$\vert$"
    else:
        graph_name = flags_meaning[flag] + "$^{" + graph_name + "}$"
    fig = plt.figure(graph_name, figsize=(10, 2), dpi=200)
    ax = fig.add_subplot()
    ax.set_title(graph_name)
    cont = ax.contourf(xi, yi, zi, cmap="jet", vmin=np.min(func_coord), vmax=np.max(func_coord), levels=20)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()

    ax.set_xlim([np.min(x_coord), np.max(x_coord)])
    ax.set_ylim([np.min(y_coord), np.max(y_coord)])
    cbar = fig.colorbar(cont, format=lambda x, _: f"{x:.2e}" if x < 0.1 else f"{x:.3f}")
    cbar.set_ticks(np.linspace(np.min(func_coord), np.max(func_coord), 4))
    if not show:
        plt.show()
