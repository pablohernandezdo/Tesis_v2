import torch.nn as nn
import torch.nn.functional as F


# 1 hidden layer models
class OneHidden6k(nn.Module):
    def __init__(self):
        super(OneHidden6k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden5k(nn.Module):
    def __init__(self):
        super(OneHidden5k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden4k(nn.Module):
    def __init__(self):
        super(OneHidden4k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden3k(nn.Module):
    def __init__(self):
        super(OneHidden3k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden2k(nn.Module):
    def __init__(self):
        super(OneHidden2k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden1k(nn.Module):
    def __init__(self):
        super(OneHidden1k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden5h(nn.Module):
    def __init__(self):
        super(OneHidden5h, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden1h(nn.Module):
    def __init__(self):
        super(OneHidden1h, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden10(nn.Module):
    def __init__(self):
        super(OneHidden10, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


class OneHidden1(nn.Module):
    def __init__(self):
        super(OneHidden1, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = self.l2(wave)
        return self.sigmoid(wave)


# 2 hidden layer models
class TwoHidden6k6k(nn.Module):
    def __init__(self):
        super(TwoHidden6k6k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k5k(nn.Module):
    def __init__(self):
        super(TwoHidden6k5k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k4k(nn.Module):
    def __init__(self):
        super(TwoHidden6k4k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k3k(nn.Module):
    def __init__(self):
        super(TwoHidden6k3k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k2k(nn.Module):
    def __init__(self):
        super(TwoHidden6k2k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k1k(nn.Module):
    def __init__(self):
        super(TwoHidden6k1k, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k5h(nn.Module):
    def __init__(self):
        super(TwoHidden6k5h, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k1h(nn.Module):
    def __init__(self):
        super(TwoHidden6k1h, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k10(nn.Module):
    def __init__(self):
        super(TwoHidden6k10, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden6k1(nn.Module):
    def __init__(self):
        super(TwoHidden6k1, self).__init__()

        self.l1 = nn.Linear(6000, 6000)
        self.l2 = nn.Linear(6000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k6k(nn.Module):
    def __init__(self):
        super(TwoHidden5k6k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k5k(nn.Module):
    def __init__(self):
        super(TwoHidden5k5k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k4k(nn.Module):
    def __init__(self):
        super(TwoHidden5k4k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k3k(nn.Module):
    def __init__(self):
        super(TwoHidden5k3k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k2k(nn.Module):
    def __init__(self):
        super(TwoHidden5k2k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k1k(nn.Module):
    def __init__(self):
        super(TwoHidden5k1k, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k5h(nn.Module):
    def __init__(self):
        super(TwoHidden5k5h, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k1h(nn.Module):
    def __init__(self):
        super(TwoHidden5k1h, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k10(nn.Module):
    def __init__(self):
        super(TwoHidden5k10, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5k1(nn.Module):
    def __init__(self):
        super(TwoHidden5k1, self).__init__()

        self.l1 = nn.Linear(6000, 5000)
        self.l2 = nn.Linear(5000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k6k(nn.Module):
    def __init__(self):
        super(TwoHidden4k6k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k5k(nn.Module):
    def __init__(self):
        super(TwoHidden4k5k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k4k(nn.Module):
    def __init__(self):
        super(TwoHidden4k4k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k3k(nn.Module):
    def __init__(self):
        super(TwoHidden4k3k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k2k(nn.Module):
    def __init__(self):
        super(TwoHidden4k2k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k1k(nn.Module):
    def __init__(self):
        super(TwoHidden4k1k, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k5h(nn.Module):
    def __init__(self):
        super(TwoHidden4k5h, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k1h(nn.Module):
    def __init__(self):
        super(TwoHidden4k1h, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k10(nn.Module):
    def __init__(self):
        super(TwoHidden4k10, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden4k1(nn.Module):
    def __init__(self):
        super(TwoHidden4k1, self).__init__()

        self.l1 = nn.Linear(6000, 4000)
        self.l2 = nn.Linear(4000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k6k(nn.Module):
    def __init__(self):
        super(TwoHidden3k6k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k5k(nn.Module):
    def __init__(self):
        super(TwoHidden3k5k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k4k(nn.Module):
    def __init__(self):
        super(TwoHidden3k4k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k3k(nn.Module):
    def __init__(self):
        super(TwoHidden3k3k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k2k(nn.Module):
    def __init__(self):
        super(TwoHidden3k2k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k1k(nn.Module):
    def __init__(self):
        super(TwoHidden3k1k, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k5h(nn.Module):
    def __init__(self):
        super(TwoHidden3k5h, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k1h(nn.Module):
    def __init__(self):
        super(TwoHidden3k1h, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k10(nn.Module):
    def __init__(self):
        super(TwoHidden3k10, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden3k1(nn.Module):
    def __init__(self):
        super(TwoHidden3k1, self).__init__()

        self.l1 = nn.Linear(6000, 3000)
        self.l2 = nn.Linear(3000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k6k(nn.Module):
    def __init__(self):
        super(TwoHidden2k6k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k5k(nn.Module):
    def __init__(self):
        super(TwoHidden2k5k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k4k(nn.Module):
    def __init__(self):
        super(TwoHidden2k4k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k3k(nn.Module):
    def __init__(self):
        super(TwoHidden2k3k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k2k(nn.Module):
    def __init__(self):
        super(TwoHidden2k2k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k1k(nn.Module):
    def __init__(self):
        super(TwoHidden2k1k, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k5h(nn.Module):
    def __init__(self):
        super(TwoHidden2k5h, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k1h(nn.Module):
    def __init__(self):
        super(TwoHidden2k1h, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k10(nn.Module):
    def __init__(self):
        super(TwoHidden2k10, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden2k1(nn.Module):
    def __init__(self):
        super(TwoHidden2k1, self).__init__()

        self.l1 = nn.Linear(6000, 2000)
        self.l2 = nn.Linear(2000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k6k(nn.Module):
    def __init__(self):
        super(TwoHidden1k6k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k5k(nn.Module):
    def __init__(self):
        super(TwoHidden1k5k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k4k(nn.Module):
    def __init__(self):
        super(TwoHidden1k4k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k3k(nn.Module):
    def __init__(self):
        super(TwoHidden1k3k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k2k(nn.Module):
    def __init__(self):
        super(TwoHidden1k2k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k1k(nn.Module):
    def __init__(self):
        super(TwoHidden1k1k, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k5h(nn.Module):
    def __init__(self):
        super(TwoHidden1k5h, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k1h(nn.Module):
    def __init__(self):
        super(TwoHidden1k1h, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k10(nn.Module):
    def __init__(self):
        super(TwoHidden1k10, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1k1(nn.Module):
    def __init__(self):
        super(TwoHidden1k1, self).__init__()

        self.l1 = nn.Linear(6000, 1000)
        self.l2 = nn.Linear(1000, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h6k(nn.Module):
    def __init__(self):
        super(TwoHidden5h6k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h5k(nn.Module):
    def __init__(self):
        super(TwoHidden5h5k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h4k(nn.Module):
    def __init__(self):
        super(TwoHidden5h4k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h3k(nn.Module):
    def __init__(self):
        super(TwoHidden5h3k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h2k(nn.Module):
    def __init__(self):
        super(TwoHidden5h2k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h1k(nn.Module):
    def __init__(self):
        super(TwoHidden5h1k, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h5h(nn.Module):
    def __init__(self):
        super(TwoHidden5h5h, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h1h(nn.Module):
    def __init__(self):
        super(TwoHidden5h1h, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h10(nn.Module):
    def __init__(self):
        super(TwoHidden5h10, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden5h1(nn.Module):
    def __init__(self):
        super(TwoHidden5h1, self).__init__()

        self.l1 = nn.Linear(6000, 500)
        self.l2 = nn.Linear(500, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h6k(nn.Module):
    def __init__(self):
        super(TwoHidden1h6k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h5k(nn.Module):
    def __init__(self):
        super(TwoHidden1h5k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h4k(nn.Module):
    def __init__(self):
        super(TwoHidden1h4k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h3k(nn.Module):
    def __init__(self):
        super(TwoHidden1h3k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h2k(nn.Module):
    def __init__(self):
        super(TwoHidden1h2k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h1k(nn.Module):
    def __init__(self):
        super(TwoHidden1h1k, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h5h(nn.Module):
    def __init__(self):
        super(TwoHidden1h5h, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h1h(nn.Module):
    def __init__(self):
        super(TwoHidden1h1h, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h10(nn.Module):
    def __init__(self):
        super(TwoHidden1h10, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1h1(nn.Module):
    def __init__(self):
        super(TwoHidden1h1, self).__init__()

        self.l1 = nn.Linear(6000, 100)
        self.l2 = nn.Linear(100, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_6k(nn.Module):
    def __init__(self):
        super(TwoHidden10_6k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_5k(nn.Module):
    def __init__(self):
        super(TwoHidden10_5k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_4k(nn.Module):
    def __init__(self):
        super(TwoHidden10_4k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_3k(nn.Module):
    def __init__(self):
        super(TwoHidden10_3k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_2k(nn.Module):
    def __init__(self):
        super(TwoHidden10_2k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_1k(nn.Module):
    def __init__(self):
        super(TwoHidden10_1k, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_5h(nn.Module):
    def __init__(self):
        super(TwoHidden10_5h, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_1h(nn.Module):
    def __init__(self):
        super(TwoHidden10_1h, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_10(nn.Module):
    def __init__(self):
        super(TwoHidden10_10, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden10_1(nn.Module):
    def __init__(self):
        super(TwoHidden10_1, self).__init__()

        self.l1 = nn.Linear(6000, 10)
        self.l2 = nn.Linear(10, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_6k(nn.Module):
    def __init__(self):
        super(TwoHidden1_6k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 6000)
        self.l3 = nn.Linear(6000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_5k(nn.Module):
    def __init__(self):
        super(TwoHidden1_5k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 5000)
        self.l3 = nn.Linear(5000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_4k(nn.Module):
    def __init__(self):
        super(TwoHidden1_4k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 4000)
        self.l3 = nn.Linear(4000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_3k(nn.Module):
    def __init__(self):
        super(TwoHidden1_3k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 3000)
        self.l3 = nn.Linear(3000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_2k(nn.Module):
    def __init__(self):
        super(TwoHidden1_2k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 2000)
        self.l3 = nn.Linear(2000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_1k(nn.Module):
    def __init__(self):
        super(TwoHidden1_1k, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 1000)
        self.l3 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_5h(nn.Module):
    def __init__(self):
        super(TwoHidden1_5h, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 500)
        self.l3 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_1h(nn.Module):
    def __init__(self):
        super(TwoHidden1_1h, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 100)
        self.l3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_10(nn.Module):
    def __init__(self):
        super(TwoHidden1_10, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)


class TwoHidden1_1(nn.Module):
    def __init__(self):
        super(TwoHidden1_1, self).__init__()

        self.l1 = nn.Linear(6000, 1)
        self.l2 = nn.Linear(1, 1)
        self.l3 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        wave = F.relu(self.l1(wave))
        wave = F.relu(self.l2(wave))
        wave = self.l3(wave)
        return self.sigmoid(wave)
