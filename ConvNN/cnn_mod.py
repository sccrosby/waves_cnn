import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvNN(nn.Module):
    def __init__(self, v_filter=1, h_filter=3, d_filter=None, u_filter=(8, 8, 16, 16, 32, 32),
                 forecast=12, history=24, offset=6, relu_e=False):
        #  history_len, forecast_len, channels
        super(ConvNN, self).__init__()

        self.relu_e = relu_e

        in_size = 2 * history + forecast
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

#	self.out_conv = 

    def forward(self, x):
        # x.size = [mb, hist + forecast, 5, freq]
        for i, conv in enumerate(self.convolutions):
            x = conv(x)
            if i < len(self.convolutions):
                x = F.relu(x)
                #if self.relu_e:
                #    F.relu(x[:, :, 4, :], inplace=True)
                #else:
                    

        # 
        x = self.conv_out(x)

        return x
