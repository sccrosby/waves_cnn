import torch.nn as nn
import torch.nn.functional as F
import torch
from data_utils.constants import A1, A2, B1, B2, E


class ConvNN(nn.Module):
    def __init__(self, v_filter=1, h_filter=3, d_filter=None, u_filter=(8, 8, 16, 16, 32, 32),
                 forecast=12, history=24, offset=6, relu_e=False, moment_constraint=None):
        #  history_len, forecast_len, channels
        super(ConvNN, self).__init__()

        self.relu_e = relu_e
        if moment_constraint == 'tanh':
            self.m_constraint = nn.Tanh()
        elif moment_constraint == 'hardtanh':
            self.m_constraint = nn.Hardtanh()
        else:
            self.m_constraint = None

        in_size = 2 * history + 2 * forecast
        out_size = forecast

        self.conv_steps = [in_size]
        if d_filter is not None:
            self.conv_steps.extend(d_filter)
        self.conv_steps.extend(u_filter)

        # self.conv_steps = [60, 30, 8, 8, 16, 16, 32, 32, 64, 64]
        k_size = (v_filter, h_filter)
        pad = (v_filter // 2, h_filter // 2)

        self.convolutions = []

        # add convolutions
        for i in range(1, len(self.conv_steps)):
            attr_name = "conv" + str(i + 1)
            print("\t--> %s in=%d out=%d" % (attr_name, self.conv_steps[i-1], self.conv_steps[i]))
            setattr(self, attr_name, nn.Conv2d(in_channels=self.conv_steps[i - 1],
                                               out_channels=self.conv_steps[i],
                                               kernel_size=k_size, padding=pad))

            self.convolutions.append(getattr(self, attr_name))

        print("\t--> %s in=%d out=%d" % ('conv_out', self.conv_steps[-1], out_size))
        self.conv_out = nn.Conv2d(in_channels=self.conv_steps[-1], out_channels=out_size, kernel_size=(1, 1), padding=0)

    def apply_moment_constraints(self, mean_e, std_e, for_e, model_e, std_x, mean_x, for_x, model_x):
        # 0425 constraint model
        # cutoff any negative energy
        e_min = -1.0 * mean_e / std_e
        e_hat = torch.max(for_e + model_e, e_min).unsqueeze(2)

        # create standardized space to put bounds on x
        std_coeff = std_e / std_x  # b
        mu_coeff = (mean_e - mean_x) / std_x  # m
        bound = model_e.unsqueeze(2) * std_coeff + mu_coeff  # [48, 24, 4, 28]

        # apply bounds to predictions
        x_pred = for_x + model_x
        x_hat = torch.max(torch.min(x_pred, bound), -1.0 * bound)

        return torch.cat((x_hat, e_hat), 2)

    def constrain_energy(self, pred_energy, mean_e, std_e):
        e_min = -1.0 * mean_e / std_e
        e_hat = torch.max(pred_energy, e_min).unsqueeze(2)
        return e_hat

    def forward(self, inputs, y_for, mean, std):
        # iterate through the convolutions
        # x.size = [mb, hist + forecast, 5, freq]
        pred = inputs
        for i, conv in enumerate(self.convolutions):
            pred = conv(pred)
            if i < len(self.convolutions):
                pred = F.leaky_relu(pred)

        # add residuals to forecast
#        print(min(self.conv_out(pred)))
#        print(max(self.conv_out(pred)))
#        print(avg(self.conv_out(pred)))
        pred = self.conv_out(pred) + y_for

        # slice energy components
        pred_energy = pred[:, :, 4, :]
        mean_e = mean[4, :]
        std_e = std[4, :]

        # slice moment components
        pred_moments = pred[:, :, 0:4, :]

        # apply constraints to energy
        e_hat = self.constrain_energy(pred_energy, mean_e, std_e)

        # Constraints to keep values between -1 and 1
        if self.m_constraint is not None:
            moments = self.m_constraint(pred_moments)
        else:
            moments = pred_moments

        # 0531 compute the mean direction vector on a1, b1
        md = torch.atan2(moments[:, :, 2, :] * std[2, :] + mean[2, :],
                         moments[:, :, 0, :] * std[0, :] + mean[0, :])

        return torch.cat((moments, e_hat), 2), md

