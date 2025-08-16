import torch
import torch.nn as nn
import torch.optim as optim

import time

from ResultFromANSYS import coord_and_true_result, get_boundary_data
import torch.utils.data as data

start_time = time.time()
device = torch.device("cpu")


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.layers = nn.Sequential(  # Построение слоев нейронной сети
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # 1
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # 2
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # 3
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # 4
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # 5
            nn.Tanh(),
            nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        return self.layers(x)  # прямой проход


class MyDataset(data.Dataset):
    def __init__(self, x, y, t, result_data):
        self.x = x
        self.y = y
        self.t = t
        self.result_data = result_data

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.t[item], self.result_data[item]

    def __len__(self):
        return len(self.x)


def pde_xx(tens_output, x, y):
    u_x = tens_output[:, 0]
    u_y = tens_output[:, 1]
    sigma_xx = tens_output[:, 2]

    du_x_dx = \
        torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(x), create_graph=True,
                            retain_graph=True)[0]
    du_y_dy = \
        torch.autograd.grad(outputs=u_y, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True,
                            retain_graph=True)[0]

    koef_1 = sigma_xx_c * l_c / (u_y_c * lyambda_c)
    koef_2 = u_x_c / u_y_c
    f = koef_1 * sigma_xx - (koef_2 * (lyambda + 2 * mu) * du_x_dx + lyambda * du_y_dy)
    return f


def pde_yy(tens_output, x, y):
    u_x = tens_output[:, 0]
    u_y = tens_output[:, 1]
    sigma_yy = tens_output[:, 3]

    du_x_dx = \
        torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(x), create_graph=True,
                            retain_graph=True)[0]
    du_y_dy = \
        torch.autograd.grad(outputs=u_y, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True,
                            retain_graph=True)[0]

    koef_1 = sigma_yy_c * l_c / (u_y_c * lyambda_c)
    koef_2 = u_x_c / u_y_c
    f = koef_1 * sigma_yy - ((lyambda + 2 * mu) * du_y_dy + koef_2 * lyambda * du_x_dx)
    return f


def pde_xy(tens_output, x, y):
    u_x = tens_output[:, 0]
    u_y = tens_output[:, 1]
    sigma_xy = tens_output[:, 4]

    du_x_dy = \
        torch.autograd.grad(outputs=u_x, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True,
                            retain_graph=True)[0]
    du_y_dx = \
        torch.autograd.grad(outputs=u_y, inputs=x, grad_outputs=torch.ones_like(x), create_graph=True,
                            retain_graph=True)[0]

    koef_1 = sigma_xy_c * l_c / (u_y_c * lyambda_c)
    koef_2 = u_x_c / u_y_c
    f = koef_1 * sigma_xy - mu * (koef_2 * du_x_dy + du_y_dx)
    return f


def dop_eq_1(tens_output, x, y, t):
    u_x = tens_output[:, 0]
    sigma_xx = tens_output[:, 2]
    sigma_xy = tens_output[:, 4]

    du_x_dt = \
        torch.autograd.grad(outputs=u_x, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True,
                            retain_graph=True)[0]
    ddu_x_dt_dt = \
        torch.autograd.grad(outputs=du_x_dt, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True,
                            retain_graph=True)[0]
    dsigma_xx_dx = \
        torch.autograd.grad(outputs=sigma_xx, inputs=x, grad_outputs=torch.ones_like(x), create_graph=True,
                            retain_graph=True)[0]
    dsigma_xy_dy = \
        torch.autograd.grad(outputs=sigma_xy, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    koef_1 = sigma_xy_c / sigma_xx_c
    koef_2 = (u_x_c * ro_c * l_c) / (sigma_xx_c * t_c ** 2)
    f = dsigma_xx_dx + koef_1 * dsigma_xy_dy - koef_2 * ro * ddu_x_dt_dt
    return f


def dop_eq_2(tens_output, x, y, t):
    u_y = tens_output[:, 1]
    sigma_xy = tens_output[:, 4]
    sigma_yy = tens_output[:, 3]

    du_y_dt = \
        torch.autograd.grad(outputs=u_y, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True,
                            retain_graph=True)[0]
    ddu_y_dt_dt = \
        torch.autograd.grad(outputs=du_y_dt, inputs=t, grad_outputs=torch.ones_like(t), create_graph=True,
                            retain_graph=True)[0]
    dsigma_xy_dx = \
        torch.autograd.grad(outputs=sigma_xy, inputs=x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dsigma_yy_dy = \
        torch.autograd.grad(outputs=sigma_yy, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    koef_1 = sigma_xy_c / sigma_xx_c
    koef_2 = sigma_yy_c / sigma_xx_c
    koef_3 = (u_y_c * ro_c * l_c) / (sigma_xx_c * t_c ** 2)
    f = koef_1 * dsigma_xy_dx + koef_2 * dsigma_yy_dy - koef_3 * ro * ddu_y_dt_dt
    return f


def result_loss(x, y, t, result_field, init_loss=False):
    x_y_t = torch.stack((x, y, t), -1)
    output = model(x_y_t)
    govern_eq_xx = pde_xx(output, x, y)
    govern_eq_yy = pde_yy(output, x, y)
    govern_eq_xy = pde_xy(output, x, y)

    u_x_True = result_field[:, 0]
    u_y_True = result_field[:, 1]
    sigma_xx_True = result_field[:, 2]
    sigma_yy_True = result_field[:, 3]
    sigma_xy_True = result_field[:, 4]

    dop_eq_11 = dop_eq_1(output, x, y, t)
    dop_eq_22 = dop_eq_2(output, x, y, t)

    loss_data = (loss_func(output[:, 0], u_x_True) + loss_func(output[:, 1], u_y_True) +
                loss_func(output[:, 2], sigma_xx_True) + loss_func(output[:, 3], sigma_yy_True) +
                loss_func(output[:, 4], sigma_xy_True)) / loss_data_0

    loss_gov_eq_xx = loss_func(govern_eq_xx, torch.zeros_like(govern_eq_xx)) / loss_gov_eq_xx_0
    loss_gov_eq_yy = loss_func(govern_eq_yy, torch.zeros_like(govern_eq_yy)) / loss_gov_eq_yy_0
    loss_gov_eq_xy = loss_func(govern_eq_xy, torch.zeros_like(govern_eq_xy)) / loss_gov_eq_xy_0

    loss_dop_eq_11 = loss_func(dop_eq_11, torch.zeros_like(dop_eq_11)) / loss_dop_eq_11_0
    loss_dop_eq_12 = loss_func(dop_eq_22, torch.zeros_like(dop_eq_22)) / loss_dop_eq_12_0

    total_loss = 1e7 * loss_data + 1 * (loss_gov_eq_xx + loss_gov_eq_yy + loss_gov_eq_xy + loss_dop_eq_11 + loss_dop_eq_12)
    if init_loss:
        return (loss_data.item(), loss_gov_eq_xx.item(), loss_gov_eq_yy.item(), loss_gov_eq_xy.item(),
                loss_dop_eq_11.item(), loss_dop_eq_12.item())
    return total_loss


lyambda = 115_385
mu = 76_923
ro = 7.85e-06

model = SimpleModel()

x_train, y_train, t_train, result_True_import = coord_and_true_result()
result_True = result_True_import.clone()

x_c = torch.max(torch.abs(x_train)).data
y_c = torch.max(torch.abs(y_train)).data
t_c = torch.max(torch.abs(t_train)).data  # 1 ДЛЯ САМОЙ ПЕРВОЙ ЗАДАЧИ
u_x_c = torch.max(torch.abs(result_True[:, 0])).data
u_y_c = torch.max(torch.abs(result_True[:, 1])).data
sigma_xx_c = torch.max(torch.abs(result_True[:, 2])).data
sigma_yy_c = torch.max(torch.abs(result_True[:, 3])).data
sigma_xy_c = torch.max(torch.abs(result_True[:, 4])).data
lyambda_c = lyambda
ro_c = ro
l_c = torch.max(x_c, y_c)


x_train, y_train, t_train, result_True = get_boundary_data(x_train, y_train, t_train,
                                                           result_True)  # data only from boundary
x_train, y_train, t_train = x_train / l_c, y_train / l_c, t_train / t_c

x_train.requires_grad = True
y_train.requires_grad = True
t_train.requires_grad = True

lyambda, mu, ro = lyambda / lyambda_c, mu / lyambda_c, ro / ro_c

result_True[:, 0] = result_True[:, 0] / u_x_c
result_True[:, 1] = result_True[:, 1] / u_y_c
result_True[:, 2] = result_True[:, 2] / sigma_xx_c
result_True[:, 3] = result_True[:, 3] / sigma_yy_c
result_True[:, 4] = result_True[:, 4] / sigma_xy_c

loss_func = torch.nn.MSELoss()

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1501, 3001, 5001, 6501], gamma=0.1)

model.train()  # правило хорошего тона

point_step = 3
data_set = MyDataset(x_train[::point_step], y_train[::point_step], t_train[::point_step], result_True[::point_step])
data_loader = data.DataLoader(data_set, batch_size=1024, shuffle=True, drop_last=False)

(loss_data_0, loss_gov_eq_xx_0, loss_gov_eq_yy_0,
 loss_gov_eq_xy_0, loss_dop_eq_11_0, loss_dop_eq_12_0) = result_loss(x_train[::point_step],
                                                                     y_train[::point_step],
                                                                     t_train[::point_step],
                                                                     result_True[::point_step],
                                                                     init_loss=True)

loss_history = {}

for epoch in range(8_001):
    for x_tr, y_tr, t_tr, res_tr in data_loader:
        loss = result_loss(x_tr, y_tr, t_tr, res_tr)

        if epoch not in loss_history:
            loss_history[epoch] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step()

    if epoch % 10 == 0:
        print(f"total loss: {(loss_history[epoch] / loss_history[0]).item():.3e}, epoch = {epoch}")

end_time = time.time() - start_time
print(f"Время обучения: {end_time // 60} минут и {end_time % 60:.2f} секунд")


torch.save(model.state_dict(), "./Models/model.pth")
