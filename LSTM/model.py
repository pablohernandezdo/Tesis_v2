import torch
import torch.nn as nn
import torch.nn.functional as F

# 1 linear output layer

# 16

class Lstm_16_16_1_1(nn.Module):
    def __init__(self):
        super(Lstm_16_16_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_16_2_1(nn.Module):
    def __init__(self):
        super(Lstm_16_16_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_16_5_1(nn.Module):
    def __init__(self):
        super(Lstm_16_16_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_16_10_1(nn.Module):
    def __init__(self):
        super(Lstm_16_16_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_16_20_1(nn.Module):
    def __init__(self):
        super(Lstm_16_16_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_32_1_1(nn.Module):
    def __init__(self):
        super(Lstm_16_32_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_32_2_1(nn.Module):
    def __init__(self):
        super(Lstm_16_32_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_32_5_1(nn.Module):
    def __init__(self):
        super(Lstm_16_32_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_32_10_1(nn.Module):
    def __init__(self):
        super(Lstm_16_32_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_32_20_1(nn.Module):
    def __init__(self):
        super(Lstm_16_32_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_64_1_1(nn.Module):
    def __init__(self):
        super(Lstm_16_64_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_64_2_1(nn.Module):
    def __init__(self):
        super(Lstm_16_64_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_64_5_1(nn.Module):
    def __init__(self):
        super(Lstm_16_64_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_64_10_1(nn.Module):
    def __init__(self):
        super(Lstm_16_64_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_64_20_1(nn.Module):
    def __init__(self):
        super(Lstm_16_64_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_128_1_1(nn.Module):
    def __init__(self):
        super(Lstm_16_128_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_128_2_1(nn.Module):
    def __init__(self):
        super(Lstm_16_128_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_128_5_1(nn.Module):
    def __init__(self):
        super(Lstm_16_128_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_128_10_1(nn.Module):
    def __init__(self):
        super(Lstm_16_128_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_128_20_1(nn.Module):
    def __init__(self):
        super(Lstm_16_128_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_256_1_1(nn.Module):
    def __init__(self):
        super(Lstm_16_256_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_256_2_1(nn.Module):
    def __init__(self):
        super(Lstm_16_256_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_256_5_1(nn.Module):
    def __init__(self):
        super(Lstm_16_256_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_256_10_1(nn.Module):
    def __init__(self):
        super(Lstm_16_256_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_16_256_20_1(nn.Module):
    def __init__(self):
        super(Lstm_16_256_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


# 32

class Lstm_32_16_1_1(nn.Module):
    def __init__(self):
        super(Lstm_32_16_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_16_2_1(nn.Module):
    def __init__(self):
        super(Lstm_32_16_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_16_5_1(nn.Module):
    def __init__(self):
        super(Lstm_32_16_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_16_10_1(nn.Module):
    def __init__(self):
        super(Lstm_32_16_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_16_20_1(nn.Module):
    def __init__(self):
        super(Lstm_32_16_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_32_1_1(nn.Module):
    def __init__(self):
        super(Lstm_32_32_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_32_2_1(nn.Module):
    def __init__(self):
        super(Lstm_32_32_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_32_5_1(nn.Module):
    def __init__(self):
        super(Lstm_32_32_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_32_10_1(nn.Module):
    def __init__(self):
        super(Lstm_32_32_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_32_20_1(nn.Module):
    def __init__(self):
        super(Lstm_32_32_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_64_1_1(nn.Module):
    def __init__(self):
        super(Lstm_32_64_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_64_2_1(nn.Module):
    def __init__(self):
        super(Lstm_32_64_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_64_5_1(nn.Module):
    def __init__(self):
        super(Lstm_32_64_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_64_10_1(nn.Module):
    def __init__(self):
        super(Lstm_32_64_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_64_20_1(nn.Module):
    def __init__(self):
        super(Lstm_32_64_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_128_1_1(nn.Module):
    def __init__(self):
        super(Lstm_32_128_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_128_2_1(nn.Module):
    def __init__(self):
        super(Lstm_32_128_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_128_5_1(nn.Module):
    def __init__(self):
        super(Lstm_32_128_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_128_10_1(nn.Module):
    def __init__(self):
        super(Lstm_32_128_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_128_20_1(nn.Module):
    def __init__(self):
        super(Lstm_32_128_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_256_1_1(nn.Module):
    def __init__(self):
        super(Lstm_32_256_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_256_2_1(nn.Module):
    def __init__(self):
        super(Lstm_32_256_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_256_5_1(nn.Module):
    def __init__(self):
        super(Lstm_32_256_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_256_10_1(nn.Module):
    def __init__(self):
        super(Lstm_32_256_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_32_256_20_1(nn.Module):
    def __init__(self):
        super(Lstm_32_256_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


# 64

class Lstm_64_16_1_1(nn.Module):
    def __init__(self):
        super(Lstm_64_16_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_16_2_1(nn.Module):
    def __init__(self):
        super(Lstm_64_16_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_16_5_1(nn.Module):
    def __init__(self):
        super(Lstm_64_16_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_16_10_1(nn.Module):
    def __init__(self):
        super(Lstm_64_16_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_16_20_1(nn.Module):
    def __init__(self):
        super(Lstm_64_16_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_32_1_1(nn.Module):
    def __init__(self):
        super(Lstm_64_32_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_32_2_1(nn.Module):
    def __init__(self):
        super(Lstm_64_32_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_32_5_1(nn.Module):
    def __init__(self):
        super(Lstm_64_32_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_32_10_1(nn.Module):
    def __init__(self):
        super(Lstm_64_32_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_32_20_1(nn.Module):
    def __init__(self):
        super(Lstm_64_32_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_64_1_1(nn.Module):
    def __init__(self):
        super(Lstm_64_64_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_64_2_1(nn.Module):
    def __init__(self):
        super(Lstm_64_64_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_64_5_1(nn.Module):
    def __init__(self):
        super(Lstm_64_64_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_64_10_1(nn.Module):
    def __init__(self):
        super(Lstm_64_64_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_64_20_1(nn.Module):
    def __init__(self):
        super(Lstm_64_64_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_128_1_1(nn.Module):
    def __init__(self):
        super(Lstm_64_128_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_128_2_1(nn.Module):
    def __init__(self):
        super(Lstm_64_128_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_128_5_1(nn.Module):
    def __init__(self):
        super(Lstm_64_128_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_128_10_1(nn.Module):
    def __init__(self):
        super(Lstm_64_128_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_128_20_1(nn.Module):
    def __init__(self):
        super(Lstm_64_128_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_256_1_1(nn.Module):
    def __init__(self):
        super(Lstm_64_256_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_256_2_1(nn.Module):
    def __init__(self):
        super(Lstm_64_256_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_256_5_1(nn.Module):
    def __init__(self):
        super(Lstm_64_256_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_256_10_1(nn.Module):
    def __init__(self):
        super(Lstm_64_256_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_64_256_20_1(nn.Module):
    def __init__(self):
        super(Lstm_64_256_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


# 128


class Lstm_128_16_1_1(nn.Module):
    def __init__(self):
        super(Lstm_128_16_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_16_2_1(nn.Module):
    def __init__(self):
        super(Lstm_128_16_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_16_5_1(nn.Module):
    def __init__(self):
        super(Lstm_128_16_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_16_10_1(nn.Module):
    def __init__(self):
        super(Lstm_128_16_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_16_20_1(nn.Module):
    def __init__(self):
        super(Lstm_128_16_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_32_1_1(nn.Module):
    def __init__(self):
        super(Lstm_128_32_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_32_2_1(nn.Module):
    def __init__(self):
        super(Lstm_128_32_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_32_5_1(nn.Module):
    def __init__(self):
        super(Lstm_128_32_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_32_10_1(nn.Module):
    def __init__(self):
        super(Lstm_128_32_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_32_20_1(nn.Module):
    def __init__(self):
        super(Lstm_128_32_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_64_1_1(nn.Module):
    def __init__(self):
        super(Lstm_128_64_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_64_2_1(nn.Module):
    def __init__(self):
        super(Lstm_128_64_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_64_5_1(nn.Module):
    def __init__(self):
        super(Lstm_128_64_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_64_10_1(nn.Module):
    def __init__(self):
        super(Lstm_128_64_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_64_20_1(nn.Module):
    def __init__(self):
        super(Lstm_128_64_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_128_1_1(nn.Module):
    def __init__(self):
        super(Lstm_128_128_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_128_2_1(nn.Module):
    def __init__(self):
        super(Lstm_128_128_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_128_5_1(nn.Module):
    def __init__(self):
        super(Lstm_128_128_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_128_10_1(nn.Module):
    def __init__(self):
        super(Lstm_128_128_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_128_20_1(nn.Module):
    def __init__(self):
        super(Lstm_128_128_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_256_1_1(nn.Module):
    def __init__(self):
        super(Lstm_128_256_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_256_2_1(nn.Module):
    def __init__(self):
        super(Lstm_128_256_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_256_5_1(nn.Module):
    def __init__(self):
        super(Lstm_128_256_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_256_10_1(nn.Module):
    def __init__(self):
        super(Lstm_128_256_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_128_256_20_1(nn.Module):
    def __init__(self):
        super(Lstm_128_256_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


# 256

class Lstm_256_16_1_1(nn.Module):
    def __init__(self):
        super(Lstm_256_16_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_16_2_1(nn.Module):
    def __init__(self):
        super(Lstm_256_16_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_16_5_1(nn.Module):
    def __init__(self):
        super(Lstm_256_16_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_16_10_1(nn.Module):
    def __init__(self):
        super(Lstm_256_16_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_16_20_1(nn.Module):
    def __init__(self):
        super(Lstm_256_16_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_32_1_1(nn.Module):
    def __init__(self):
        super(Lstm_256_32_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_32_2_1(nn.Module):
    def __init__(self):
        super(Lstm_256_32_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_32_5_1(nn.Module):
    def __init__(self):
        super(Lstm_256_32_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_32_10_1(nn.Module):
    def __init__(self):
        super(Lstm_256_32_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_32_20_1(nn.Module):
    def __init__(self):
        super(Lstm_256_32_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_64_1_1(nn.Module):
    def __init__(self):
        super(Lstm_256_64_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_64_2_1(nn.Module):
    def __init__(self):
        super(Lstm_256_64_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_64_5_1(nn.Module):
    def __init__(self):
        super(Lstm_256_64_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_64_10_1(nn.Module):
    def __init__(self):
        super(Lstm_256_64_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_64_20_1(nn.Module):
    def __init__(self):
        super(Lstm_256_64_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_128_1_1(nn.Module):
    def __init__(self):
        super(Lstm_256_128_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_128_2_1(nn.Module):
    def __init__(self):
        super(Lstm_256_128_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_128_5_1(nn.Module):
    def __init__(self):
        super(Lstm_256_128_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_128_10_1(nn.Module):
    def __init__(self):
        super(Lstm_256_128_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_128_20_1(nn.Module):
    def __init__(self):
        super(Lstm_256_128_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_256_1_1(nn.Module):
    def __init__(self):
        super(Lstm_256_256_1_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_256_2_1(nn.Module):
    def __init__(self):
        super(Lstm_256_256_2_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_256_5_1(nn.Module):
    def __init__(self):
        super(Lstm_256_256_5_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_256_10_1(nn.Module):
    def __init__(self):
        super(Lstm_256_256_10_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


class Lstm_256_256_20_1(nn.Module):
    def __init__(self):
        super(Lstm_256_256_20_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)

        return self.sigmoid(out_l1)


# 2 linear output layers

class Lstm_16_16_1_2(nn.Module):
    def __init__(self):
        super(Lstm_16_16_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_16_2_2(nn.Module):
    def __init__(self):
        super(Lstm_16_16_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_16_5_2(nn.Module):
    def __init__(self):
        super(Lstm_16_16_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_16_10_2(nn.Module):
    def __init__(self):
        super(Lstm_16_16_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_16_20_2(nn.Module):
    def __init__(self):
        super(Lstm_16_16_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_32_1_2(nn.Module):
    def __init__(self):
        super(Lstm_16_32_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_32_2_2(nn.Module):
    def __init__(self):
        super(Lstm_16_32_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_32_5_2(nn.Module):
    def __init__(self):
        super(Lstm_16_32_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_32_10_2(nn.Module):
    def __init__(self):
        super(Lstm_16_32_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_32_20_2(nn.Module):
    def __init__(self):
        super(Lstm_16_32_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_64_1_2(nn.Module):
    def __init__(self):
        super(Lstm_16_64_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_64_2_2(nn.Module):
    def __init__(self):
        super(Lstm_16_64_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_64_5_2(nn.Module):
    def __init__(self):
        super(Lstm_16_64_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_64_10_2(nn.Module):
    def __init__(self):
        super(Lstm_16_64_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_64_20_2(nn.Module):
    def __init__(self):
        super(Lstm_16_64_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_128_1_2(nn.Module):
    def __init__(self):
        super(Lstm_16_128_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_128_2_2(nn.Module):
    def __init__(self):
        super(Lstm_16_128_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_128_5_2(nn.Module):
    def __init__(self):
        super(Lstm_16_128_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_128_10_2(nn.Module):
    def __init__(self):
        super(Lstm_16_128_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_128_20_2(nn.Module):
    def __init__(self):
        super(Lstm_16_128_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_256_1_2(nn.Module):
    def __init__(self):
        super(Lstm_16_256_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_256_2_2(nn.Module):
    def __init__(self):
        super(Lstm_16_256_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_256_5_2(nn.Module):
    def __init__(self):
        super(Lstm_16_256_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_256_10_2(nn.Module):
    def __init__(self):
        super(Lstm_16_256_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_16_256_20_2(nn.Module):
    def __init__(self):
        super(Lstm_16_256_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 2, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(4, 8, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(8, 16, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)

        self.lstm = nn.LSTM(16, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


# 32

class Lstm_32_16_1_2(nn.Module):
    def __init__(self):
        super(Lstm_32_16_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_16_2_2(nn.Module):
    def __init__(self):
        super(Lstm_32_16_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_16_5_2(nn.Module):
    def __init__(self):
        super(Lstm_32_16_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_16_10_2(nn.Module):
    def __init__(self):
        super(Lstm_32_16_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_16_20_2(nn.Module):
    def __init__(self):
        super(Lstm_32_16_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_32_1_2(nn.Module):
    def __init__(self):
        super(Lstm_32_32_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_32_2_2(nn.Module):
    def __init__(self):
        super(Lstm_32_32_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_32_5_2(nn.Module):
    def __init__(self):
        super(Lstm_32_32_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_32_10_2(nn.Module):
    def __init__(self):
        super(Lstm_32_32_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_32_20_2(nn.Module):
    def __init__(self):
        super(Lstm_32_32_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_64_1_2(nn.Module):
    def __init__(self):
        super(Lstm_32_64_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_64_2_2(nn.Module):
    def __init__(self):
        super(Lstm_32_64_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_64_5_2(nn.Module):
    def __init__(self):
        super(Lstm_32_64_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_64_10_2(nn.Module):
    def __init__(self):
        super(Lstm_32_64_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_64_20_2(nn.Module):
    def __init__(self):
        super(Lstm_32_64_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_128_1_2(nn.Module):
    def __init__(self):
        super(Lstm_32_128_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_128_2_2(nn.Module):
    def __init__(self):
        super(Lstm_32_128_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_128_5_2(nn.Module):
    def __init__(self):
        super(Lstm_32_128_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_128_10_2(nn.Module):
    def __init__(self):
        super(Lstm_32_128_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_128_20_2(nn.Module):
    def __init__(self):
        super(Lstm_32_128_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_256_1_2(nn.Module):
    def __init__(self):
        super(Lstm_32_256_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_256_2_2(nn.Module):
    def __init__(self):
        super(Lstm_32_256_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_256_5_2(nn.Module):
    def __init__(self):
        super(Lstm_32_256_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_256_10_2(nn.Module):
    def __init__(self):
        super(Lstm_32_256_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_32_256_20_2(nn.Module):
    def __init__(self):
        super(Lstm_32_256_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 8, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(32, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


# 64

class Lstm_64_16_1_2(nn.Module):
    def __init__(self):
        super(Lstm_64_16_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_16_2_2(nn.Module):
    def __init__(self):
        super(Lstm_64_16_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_16_5_2(nn.Module):
    def __init__(self):
        super(Lstm_64_16_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_16_10_2(nn.Module):
    def __init__(self):
        super(Lstm_64_16_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_16_20_2(nn.Module):
    def __init__(self):
        super(Lstm_64_16_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_32_1_2(nn.Module):
    def __init__(self):
        super(Lstm_64_32_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_32_2_2(nn.Module):
    def __init__(self):
        super(Lstm_64_32_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_32_5_2(nn.Module):
    def __init__(self):
        super(Lstm_64_32_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_32_10_2(nn.Module):
    def __init__(self):
        super(Lstm_64_32_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_32_20_2(nn.Module):
    def __init__(self):
        super(Lstm_64_32_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_64_1_2(nn.Module):
    def __init__(self):
        super(Lstm_64_64_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_64_2_2(nn.Module):
    def __init__(self):
        super(Lstm_64_64_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_64_5_2(nn.Module):
    def __init__(self):
        super(Lstm_64_64_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_64_10_2(nn.Module):
    def __init__(self):
        super(Lstm_64_64_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_64_20_2(nn.Module):
    def __init__(self):
        super(Lstm_64_64_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_128_1_2(nn.Module):
    def __init__(self):
        super(Lstm_64_128_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_128_2_2(nn.Module):
    def __init__(self):
        super(Lstm_64_128_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_128_5_2(nn.Module):
    def __init__(self):
        super(Lstm_64_128_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_128_10_2(nn.Module):
    def __init__(self):
        super(Lstm_64_128_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_128_20_2(nn.Module):
    def __init__(self):
        super(Lstm_64_128_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_256_1_2(nn.Module):
    def __init__(self):
        super(Lstm_64_256_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_256_2_2(nn.Module):
    def __init__(self):
        super(Lstm_64_256_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_256_5_2(nn.Module):
    def __init__(self):
        super(Lstm_64_256_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_256_10_2(nn.Module):
    def __init__(self):
        super(Lstm_64_256_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_64_256_20_2(nn.Module):
    def __init__(self):
        super(Lstm_64_256_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 4, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(4, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


# 128

class Lstm_128_16_1_2(nn.Module):
    def __init__(self):
        super(Lstm_128_16_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_16_2_2(nn.Module):
    def __init__(self):
        super(Lstm_128_16_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_16_5_2(nn.Module):
    def __init__(self):
        super(Lstm_128_16_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_16_10_2(nn.Module):
    def __init__(self):
        super(Lstm_128_16_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_16_20_2(nn.Module):
    def __init__(self):
        super(Lstm_128_16_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_32_1_2(nn.Module):
    def __init__(self):
        super(Lstm_128_32_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_32_2_2(nn.Module):
    def __init__(self):
        super(Lstm_128_32_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_32_5_2(nn.Module):
    def __init__(self):
        super(Lstm_128_32_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_32_10_2(nn.Module):
    def __init__(self):
        super(Lstm_128_32_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_32_20_2(nn.Module):
    def __init__(self):
        super(Lstm_128_32_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_64_1_2(nn.Module):
    def __init__(self):
        super(Lstm_128_64_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_64_2_2(nn.Module):
    def __init__(self):
        super(Lstm_128_64_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_64_5_2(nn.Module):
    def __init__(self):
        super(Lstm_128_64_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_64_10_2(nn.Module):
    def __init__(self):
        super(Lstm_128_64_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_64_20_2(nn.Module):
    def __init__(self):
        super(Lstm_128_64_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_128_1_2(nn.Module):
    def __init__(self):
        super(Lstm_128_128_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_128_2_2(nn.Module):
    def __init__(self):
        super(Lstm_128_128_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_128_5_2(nn.Module):
    def __init__(self):
        super(Lstm_128_128_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_128_10_2(nn.Module):
    def __init__(self):
        super(Lstm_128_128_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_128_20_2(nn.Module):
    def __init__(self):
        super(Lstm_128_128_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_256_1_2(nn.Module):
    def __init__(self):
        super(Lstm_128_256_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_256_2_2(nn.Module):
    def __init__(self):
        super(Lstm_128_256_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_256_5_2(nn.Module):
    def __init__(self):
        super(Lstm_128_256_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_256_10_2(nn.Module):
    def __init__(self):
        super(Lstm_128_256_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_128_256_20_2(nn.Module):
    def __init__(self):
        super(Lstm_128_256_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(64, 128, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(128, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


# 256

class Lstm_256_16_1_2(nn.Module):
    def __init__(self):
        super(Lstm_256_16_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 1, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_16_2_2(nn.Module):
    def __init__(self):
        super(Lstm_256_16_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 2, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_16_5_2(nn.Module):
    def __init__(self):
        super(Lstm_256_16_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 5, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_16_10_2(nn.Module):
    def __init__(self):
        super(Lstm_256_16_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 10, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_16_20_2(nn.Module):
    def __init__(self):
        super(Lstm_256_16_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 16, 20, batch_first=True)

        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_32_1_2(nn.Module):
    def __init__(self):
        super(Lstm_256_32_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 1, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_32_2_2(nn.Module):
    def __init__(self):
        super(Lstm_256_32_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 2, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_32_5_2(nn.Module):
    def __init__(self):
        super(Lstm_256_32_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 5, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_32_10_2(nn.Module):
    def __init__(self):
        super(Lstm_256_32_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 10, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_32_20_2(nn.Module):
    def __init__(self):
        super(Lstm_256_32_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 32, 20, batch_first=True)

        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_64_1_2(nn.Module):
    def __init__(self):
        super(Lstm_256_64_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 1, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_64_2_2(nn.Module):
    def __init__(self):
        super(Lstm_256_64_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 2, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_64_5_2(nn.Module):
    def __init__(self):
        super(Lstm_256_64_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 5, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_64_10_2(nn.Module):
    def __init__(self):
        super(Lstm_256_64_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 10, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_64_20_2(nn.Module):
    def __init__(self):
        super(Lstm_256_64_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 64, 20, batch_first=True)

        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_128_1_2(nn.Module):
    def __init__(self):
        super(Lstm_256_128_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 1, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_128_2_2(nn.Module):
    def __init__(self):
        super(Lstm_256_128_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 2, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_128_5_2(nn.Module):
    def __init__(self):
        super(Lstm_256_128_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 5, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_128_10_2(nn.Module):
    def __init__(self):
        super(Lstm_256_128_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 10, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_128_20_2(nn.Module):
    def __init__(self):
        super(Lstm_256_128_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 128, 20, batch_first=True)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_256_1_2(nn.Module):
    def __init__(self):
        super(Lstm_256_256_1_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 1, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_256_2_2(nn.Module):
    def __init__(self):
        super(Lstm_256_256_2_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 2, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_256_5_2(nn.Module):
    def __init__(self):
        super(Lstm_256_256_5_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_256_10_2(nn.Module):
    def __init__(self):
        super(Lstm_256_256_10_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 10, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


class Lstm_256_256_20_2(nn.Module):
    def __init__(self):
        super(Lstm_256_256_20_2, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 20, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):
        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = F.relu(self.l2(out_l1))

        return self.sigmoid(out_l2)


# MODELO ORIGINAL


class CNNLSTMANN(nn.Module):
    def __init__(self):
        super(CNNLSTMANN, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, stride=2)

        self.p1 = nn.MaxPool1d(3)
        self.p2 = nn.MaxPool1d(5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, 5, batch_first=True)

        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, wave):

        batch_size = wave.shape[0]
        wave = wave.view(batch_size, 1, 60, 100)

        # Pasar cada una por una CNN pequeña
        out_convs = []

        for i in range(60):
            trozo = wave[:, :, i, :]
            trozo = self.bn1(F.relu(self.conv1(trozo)))
            trozo = self.bn2(F.relu(self.conv2(trozo)))
            trozo = self.p1(trozo)
            trozo = self.bn3(F.relu(self.conv3(trozo)))
            trozo = self.bn4(F.relu(self.conv4(trozo)))
            trozo = self.p2(trozo)
            out_convs.append(trozo)

        # Concatenar las salidas
        out_convs = torch.cat(out_convs, dim=2)

        # Cambiar la forma para pasar por lstm
        out_convs = out_convs.permute(0, 2, 1)

        # Pasar por lstm
        out_lstm, _ = self.lstm(out_convs)

        # Ultimo estado
        out_lstm = out_lstm[:, -1, :]
        out_lstm = out_lstm.squeeze()

        out_l1 = self.l1(out_lstm)
        out_l2 = self.l2(out_l1)
        out_l3 = self.l3(out_l2)

        return self.sigmoid(out_l3)
