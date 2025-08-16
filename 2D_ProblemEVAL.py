import torch
import torch.nn as nn

from ResultFromANSYS import coord_and_true_result, get_result
from Plotting import plotting_contour

import matplotlib.pyplot as plt

device = torch.device("cpu")


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.layers = nn.Sequential(  # Построение слоев нейронной сети
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        return self.layers(x)  # прямой проход


def get_dependence_on_t(quantity, x_show, y_show):
    fig = plt.figure(quantity + "(t)", figsize=(6.5, 5), dpi=150)
    ax = fig.add_subplot()
    # ax.set_title(quantity)
    dict_quantity = {"u_x": 0, "u_y": 1, "sigma_x": 2, "sigma_y": 3, "sigma_xy": 4}
    dict_y_name = {"u_x": "$u_{x}$", "u_y": "$u_{y}$", "sigma_x": r"$\sigma_{xx}$", "sigma_y": r"$\sigma_{yy}$",
                   "sigma_xy": r"$\sigma_{xy}$"}
    quantity_show = dict_quantity[quantity]
    x_train_show, y_train_show, t_train_show, result_show = get_result(x_train, y_train, t_train, result_True, x=x_show,
                                                                       y=y_show)
    ax.plot(t_train_show.detach().numpy(), result_show[:, quantity_show].detach().numpy(), 'r', linewidth=2)

    x_predict, y_predict, t_predict = x_train_show / l_c, y_train_show / l_c, t_train_show / t_c
    x_y_t_predict = torch.stack((x_predict, y_predict, t_predict), -1)

    model_predict = model(x_y_t_predict).clone()

    model_predict[:, 0] = model_predict[:, 0] * u_x_c
    model_predict[:, 1] = model_predict[:, 1] * u_y_c
    model_predict[:, 2] = model_predict[:, 2] * sigma_xx_c
    model_predict[:, 3] = model_predict[:, 3] * sigma_yy_c
    model_predict[:, 4] = model_predict[:, 4] * sigma_xy_c

    ax.plot(t_train_show.detach().numpy(), model_predict[:, quantity_show].detach().numpy(), 'b', linewidth=2)

    ax.set_xlabel("t")
    ax.set_ylabel(dict_y_name[quantity])
    ax.legend([dict_y_name[quantity] + "$^{FEM}$", dict_y_name[quantity] + "$^{PINN}$"])
    fig.tight_layout()
    ax.grid()
    plt.show()


def get_contour(quantity, t_show, all_graph=False):
    dict_quantity = {"u_x": 0, "u_y": 1, "sigma_x": 2, "sigma_y": 3, "sigma_xy": 4}
    x_predict, y_predict, t_predict = get_result(x_train, y_train, t_train, t=t_show)
    x_predict, y_predict, t_predict = x_predict / l_c, y_predict / l_c, t_predict / t_c
    x_y_t_predict = torch.stack((x_predict, y_predict, t_predict), -1)

    model_predict = model(x_y_t_predict).clone()

    model_predict[:, 0] = model_predict[:, 0] * u_x_c
    model_predict[:, 1] = model_predict[:, 1] * u_y_c
    model_predict[:, 2] = model_predict[:, 2] * sigma_xx_c
    model_predict[:, 3] = model_predict[:, 3] * sigma_yy_c
    model_predict[:, 4] = model_predict[:, 4] * sigma_xy_c

    quantity_show = dict_quantity[quantity]
    x_train_show, y_train_show, t_train_show, result_show = get_result(x_train, y_train, t_train, result_True, t=t_show)
    x_y_show = torch.stack((x_train_show, y_train_show), -1)
    if not all_graph:
        plotting_contour(x_y_show, result_show, flag=quantity_show)
        plotting_contour(x_y_show, model_predict, flag=quantity_show)
        plotting_contour(x_y_show, torch.abs(result_show - model_predict), flag=quantity_show)
    else:
        plotting_contour(x_y_show, result_show, flag=quantity_show, graph_name="FEM")
        plotting_contour(x_y_show, model_predict, flag=quantity_show, graph_name="PINN")
        plotting_contour(x_y_show, torch.abs(result_show - model_predict), flag=quantity_show, graph_name="FEM - PINN")
        plt.show()


model = SimpleModel()
model.load_state_dict(torch.load("./Models/model.pth", weights_only=True))
model.eval()

x_train, y_train, t_train, result_True = coord_and_true_result()
x_y_t_train = torch.stack((x_train, y_train, t_train), -1)

lyambda = 115_385
mu = 76_923
ro = 7.85e-06

x_c = torch.max(torch.abs(x_train)).data
y_c = torch.max(torch.abs(y_train)).data
t_c = torch.max(torch.abs(t_train)).data
u_x_c = torch.max(torch.abs(result_True[:, 0])).data
u_y_c = torch.max(torch.abs(result_True[:, 1])).data
sigma_xx_c = torch.max(torch.abs(result_True[:, 2])).data
sigma_yy_c = torch.max(torch.abs(result_True[:, 3])).data
sigma_xy_c = torch.max(torch.abs(result_True[:, 4])).data
lyambda_c = lyambda
ro_c = ro
l_c = torch.max(x_c, y_c)


get_contour("u_x", 0.005, True)
get_contour("u_y", 0.005, True)
get_contour("sigma_x", 0.005, True)
get_contour("sigma_y", 0.005, True)
get_contour("sigma_xy", 0.005, True)


get_dependence_on_t("u_x", 120, 5)
get_dependence_on_t("u_y", 120, 5)
get_dependence_on_t("sigma_x", 120, 5)
get_dependence_on_t("sigma_y", 120, 5)
get_dependence_on_t("sigma_xy", 120, 5)

