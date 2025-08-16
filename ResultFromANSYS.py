import torch
import numpy as np


def coord_and_true_result():
    def export_data(filename, export_coord=False):
        with open(filename, encoding="utf-8") as file:
            data = file.readlines()

        data = data[1:]

        x_coord = np.zeros(len(data))
        y_coord = np.zeros(len(data))
        func_coord = np.zeros(len(data))
        for i in range(len(data)):
            new_represent_data = data[i].replace('\n', '').split('\t')
            x_coord[i] = float(new_represent_data[1])
            y_coord[i] = float(new_represent_data[2])
            func_coord[i] = float(new_represent_data[4])

        if not export_coord:
            return func_coord

        return x_coord, y_coord, func_coord

    t_0 = 5e-4

    x_res = None
    y_res = None
    t_res = None
    result_res = None

    while t_0 <= 5e-3:
        file_name = r".\ANSYS_Data\\"
        x, y, u_x = export_data(file_name + f"{t_0}_def_x.txt", export_coord=True)
        u_y = export_data(file_name + f"{t_0}_def_y.txt")
        sigma_x = export_data(file_name + f"{t_0}_sigma_x.txt")
        sigma_y = export_data(file_name + f"{t_0}_sigma_y.txt")
        sigma_xy = export_data(file_name + f"{t_0}_sigma_xy.txt")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        t = torch.full_like(x, t_0, dtype=torch.float32)
        u_x = torch.tensor(u_x, dtype=torch.float32)
        u_y = torch.tensor(u_y, dtype=torch.float32)
        sigma_x = torch.tensor(sigma_x, dtype=torch.float32)
        sigma_y = torch.tensor(sigma_y, dtype=torch.float32)
        sigma_xy = torch.tensor(sigma_xy, dtype=torch.float32)

        result = torch.stack((u_x, u_y, sigma_x, sigma_y, sigma_xy), -1)

        if x_res == None:
            x_res = x
            y_res = y
            t_res = t
            result_res = result
        else:
            x_res = torch.cat((x_res, x), 0)
            y_res = torch.cat((y_res, y), 0)
            t_res = torch.cat((t_res, t), 0)
            result_res = torch.cat((result_res, result), 0)
        t_0 = round(t_0 + 5e-4, 5)

    return x_res, y_res, t_res, result_res


def get_result(x_coord, y_coord, t_coord, model_data=torch.empty(0), *, x=None, y=None, t=None):
    x_y_t_data = torch.stack((x_coord, y_coord, t_coord), -1)
    if not model_data.numel():
        if t is not None:
            x_y_t_data = x_y_t_data[x_y_t_data[:, 2] == t]
        else:
            x_y_t_data = x_y_t_data[x_y_t_data[:, 0] == x]
            x_y_t_data = x_y_t_data[x_y_t_data[:, 1] == y]
        return x_y_t_data[:, 0], x_y_t_data[:, 1], x_y_t_data[:, 2]
    all_data = torch.cat((x_y_t_data, model_data), -1)
    if t is not None:
        all_data = all_data[all_data[:, 2] == t]
    else:
        all_data = all_data[all_data[:, 0] == x]
        all_data = all_data[all_data[:, 1] == y]
    return all_data[:, 0], all_data[:, 1], all_data[:, 2], all_data[:, 3:]


def get_boundary_data(x_coord, y_coord, t_coord, model_data):
    x_y_t_data = torch.stack((x_coord, y_coord, t_coord), -1)
    all_data = torch.cat((x_y_t_data, model_data), -1)
    x_min, x_max = torch.min(x_coord), torch.max(x_coord)
    y_min, y_max = torch.min(y_coord), torch.max(y_coord)
    all_data_x_max = all_data[x_coord == x_max]
    all_data_y_min = all_data[y_coord == y_min]
    all_data_y_max = all_data[y_coord == y_max]
    all_data_boundary = torch.cat((all_data_x_max, all_data_y_min, all_data_y_max), 0)
    return all_data_boundary[:, 0], all_data_boundary[:, 1], all_data_boundary[:, 2], all_data_boundary[:, 3:]


if __name__ == "__main__":
    x_res, y_res, t_res, result_res = coord_and_true_result()
    print(get_result(x_res, y_res, t_res, x=120, y=3)[0:3])
