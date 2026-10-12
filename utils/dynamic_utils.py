import torch
from torch import nn
import torch.nn.functional as F


class unicycle(torch.nn.Module):

    def __init__(self, train_timestamp, centers, eulers, heights=None):
        super(unicycle, self).__init__()
        self.train_timestamp = train_timestamp
        self.delta = torch.diff(self.train_timestamp)

        self.input_a = centers[:, 0].clone()
        self.input_b = centers[:, 1].clone()

        self.a = nn.Parameter(centers[:, 0])
        self.b = nn.Parameter(centers[:, 1])

        diff_a = torch.diff(centers[:, 0]) / self.delta
        diff_b = torch.diff(centers[:, 1]) / self.delta
        v = torch.sqrt(diff_a ** 2 + diff_b**2)

        self.v = nn.Parameter(F.pad(v, (0, 1), 'constant', v[-1].item()))
        self.pitchroll = eulers[:, :2]
        self.yaw = nn.Parameter(eulers[:, -1])

        if heights is None:
            self.h = torch.zeros_like(train_timestamp).float()
        else:
            self.h = heights

    def acc_omega(self):
        acc = torch.diff(self.v) / self.delta
        omega = torch.diff(self.yaw) / self.delta
        acc = F.pad(acc, (0, 1), 'constant', acc[-1].item())
        omega = F.pad(omega, (0, 1), 'constant', omega[-1].item())
        return acc, omega

    def forward(self, timestamp):
        if timestamp < self.train_timestamp[0]:
            delta_t = self.train_timestamp[0] - timestamp
            a = self.a[0] - delta_t * torch.cos(self.yaw[0]) * self.v[0]
            b = self.b[0] - delta_t * torch.sin(self.yaw[0]) * self.v[0]
            return a, b, self.v[0], self.pitchroll[0], self.yaw[0], self.h[0]
        elif timestamp > self.train_timestamp[-1]:
            delta_t = timestamp - self.train_timestamp[-1]
            a = self.a[-1] + delta_t * torch.cos(self.yaw[-1]) * self.v[-1]
            b = self.b[-1] + delta_t * torch.sin(self.yaw[-1]) * self.v[-1]
            return a, b, self.v[-1], self.pitchroll[-1], self.yaw[-1], self.h[-1]
        idx = torch.searchsorted(self.train_timestamp, timestamp, side='left')
        if self.train_timestamp[idx] == timestamp:
            return self.a[idx], self.b[idx], self.v[idx], self.pitchroll[idx], self.yaw[idx], self.h[idx]
        else:
            prev_timestamps = self.train_timestamp[idx-1]
            delta_t = timestamp - prev_timestamps
            prev_a, prev_b = self.a[idx-1], self.b[idx-1]
            prev_v, prev_yaw = self.v[idx-1], self.yaw[idx-1]
            acc, omega = self.acc_omega()
            v = prev_v + acc[idx-1] * delta_t
            yaw = prev_yaw + omega[idx-1] * delta_t
            a = prev_a + prev_v * ((torch.sin(yaw) - torch.sin(prev_yaw)) / (omega[idx-1] + 1e-6))
            b = prev_b - prev_v * ((torch.cos(yaw) - torch.cos(prev_yaw)) / (omega[idx-1] + 1e-6))
            h = self.h[idx-1]
        return a, b, v, self.pitchroll[idx-1], yaw, h


    @classmethod
    def restore(cls, model_args):
        (
            a,
            b,
            v,
            pitchroll,
            yaw,
            h,
            train_timestamp,
            delta
        ) = model_args
        model = cls(train_timestamp,
                    torch.stack([a.clone().detach(),b.clone().detach()], dim=-1),
                    torch.concat([pitchroll.clone().detach(), yaw.clone().detach()[:, None]], dim=-1),
                    h.clone().detach())
        model.a = a
        model.b = b
        model.v = v
        model.pitchroll = pitchroll
        model.yaw = yaw
        model.h = h
        model.train_timestamp = train_timestamp
        model.delta = delta
        return model
