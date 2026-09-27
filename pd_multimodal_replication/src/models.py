"""
Networks of Section II / Algorithm 1.

  SensorLSTM   X_sensor (T x 6)          -> Z_sensor in R^32  (stacked LSTM, Eq. 1)
  Spiral2DCNN  X_spiral (1 x 64 x 64)    -> Z_spiral in R^64  (Eq. 2-3, Fig. 4)
  Video3DCNN   X_video  (1 x 16 x 64 x 64)-> Z_video  in R^64 (Eq. 4, Fig. 2)
  HybridHead   Z_fused in R^160 (Eq. 5)  -> BAM -> {GRU branch, LSTNet branch}
               -> concat(BAM, GRU, LSTNet) -> Dropout(ReLU(W z + b)) (Eq. 6)
               -> Softmax (Eq. 7)                                   (Section II-G)

Layer widths are not given in the paper; they were chosen so the unimodal
parameter counts land near Table V (LSTM 0.12M, 2-D CNN 0.48M, 3-D CNN 3.2M).
"""
import torch
import torch.nn as nn


class SensorLSTM(nn.Module):
    def __init__(self, in_ch=6, h1=128, h2=64, out=32, readout="mean"):
        super().__init__()
        # readout="last" is the paper's Z_sensor = Dense(h_T). With the paper's
        # 10 epochs / batch 8 it collapsed to chance on 2 of 3 seeds in our
        # tests; mean-pooling the 2nd-layer states over time is stable.
        self.readout = readout
        self.lstm1 = nn.LSTM(in_ch, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.fc = nn.Linear(h2, out)
        self.out_dim = out

    def forward(self, x):                 # x: B x T x 6
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        h = h[:, -1] if self.readout == "last" else h.mean(1)
        return torch.relu(self.fc(h))            # Z_sensor in R^32


class Spiral2DCNN(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 96))   # 64 -> 8
        self.fc = nn.Linear(96 * 8 * 8, out)
        self.out_dim = out

    def forward(self, x):                 # x: B x 1 x 64 x 64
        return torch.relu(self.fc(self.features(x).flatten(1)))


class Video3DCNN(nn.Module):
    def __init__(self, out=64, hidden=192):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),   # 16x32x32
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),         # 8x16x16
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),         # 4x8x8
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 8 * 8, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.out_dim = out

    def forward(self, x):                 # x: B x 1 x 16 x 64 x 64
        return torch.relu(self.fc(self.features(x).flatten(1)))


class UnimodalClassifier(nn.Module):
    """Encoder + softmax head, used to pre-train each subnetwork independently
    ("Subnetworks were trained independently and later fused", Sec. II-B) and
    as the unimodal deep baselines of Table IV."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(encoder.out_dim, 2))

    def forward(self, x):
        return self.head(self.encoder(x))


# --------------------------------------------------------------------------- fusion head
class BAM1D(nn.Module):
    """Bottleneck Attention Module (Park et al., BMVC 2018) for C x L maps:
    M(F) = sigmoid(Mc(F) + Ms(F)),  F' = F + F * M(F)."""

    def __init__(self, c, r=16, dil=4):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, c // r),
                                     nn.BatchNorm1d(c // r), nn.ReLU(), nn.Linear(c // r, c))
        self.spatial = nn.Sequential(
            nn.Conv1d(c, c // r, 1), nn.BatchNorm1d(c // r), nn.ReLU(),
            nn.Conv1d(c // r, c // r, 3, padding=dil, dilation=dil), nn.BatchNorm1d(c // r), nn.ReLU(),
            nn.Conv1d(c // r, c // r, 3, padding=dil, dilation=dil), nn.BatchNorm1d(c // r), nn.ReLU(),
            nn.Conv1d(c // r, 1, 1))

    def forward(self, f):                 # f: B x C x L
        m = torch.sigmoid(self.channel(f)[:, :, None] + self.spatial(f))
        return f + f * m


class LSTNet(nn.Module):
    """LSTNet (Lai et al., SIGIR 2018): Conv1d -> GRU + skip-GRU, plus a linear
    autoregressive highway. Returns a feature vector instead of a forecast."""

    def __init__(self, c_in, conv_ch=64, k=6, hid=64, skip_hid=16, skip=8, hw=16, out=64):
        super().__init__()
        self.conv = nn.Conv1d(c_in, conv_ch, k)
        self.gru = nn.GRU(conv_ch, hid, batch_first=True)
        self.skip, self.skip_gru = skip, nn.GRU(conv_ch, skip_hid, batch_first=True)
        self.hw, self.highway = hw, nn.Linear(hw * c_in, out)
        self.fc = nn.Linear(hid + skip * skip_hid, out)
        self.out_dim = out

    def forward(self, x):                 # x: B x C x L
        c = torch.relu(self.conv(x))      # B x K x L'
        B, K, L = c.shape
        _, h = self.gru(c.transpose(1, 2))
        r = h[-1]
        n = L // self.skip                # recurrent-skip over periods of length `skip`
        s = c[:, :, L - n * self.skip:].reshape(B, K, n, self.skip).permute(0, 3, 2, 1)
        _, hs = self.skip_gru(s.reshape(B * self.skip, n, K))
        r = torch.cat([r, hs[-1].reshape(B, -1)], 1)
        return torch.relu(self.fc(r) + self.highway(x[:, :, -self.hw:].flatten(1)))


class HybridGRULSTNetBAM(nn.Module):
    """Section II-G. Z_fused (160) is treated as a length-160 sequence, lifted
    to C channels, refined by BAM and fed to parallel GRU and LSTNet branches."""

    def __init__(self, in_len=160, c=64, gru_hid=128, d=256, p_drop=0.5):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv1d(1, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU())
        self.bam = BAM1D(c)
        self.gru = nn.GRU(c, gru_hid, batch_first=True, bidirectional=True)
        self.lstnet = LSTNet(c)
        cat = c + 2 * gru_hid + self.lstnet.out_dim
        self.fc = nn.Sequential(nn.Linear(cat, d), nn.ReLU(), nn.Dropout(p_drop))   # Eq. 6
        self.out = nn.Linear(d, 2)                                                  # Eq. 7 (logits)

    def forward(self, z):                 # z: B x 160
        f = self.bam(self.lift(z[:, None]))
        _, h = self.gru(f.transpose(1, 2))
        g = torch.cat([h[-2], h[-1]], 1)
        parts = [f.mean(2), g, self.lstnet(f)]
        return self.out(self.fc(torch.cat(parts, 1)))


class MultimodalNet(nn.Module):
    """Full model of Algorithm 1: three encoders -> Concat (R^160) -> hybrid head."""

    def __init__(self, sensor=None, spiral=None, video=None):
        super().__init__()
        self.sensor = sensor or SensorLSTM()
        self.spiral = spiral or Spiral2DCNN()
        self.video = video or Video3DCNN()
        dim = self.sensor.out_dim + self.spiral.out_dim + self.video.out_dim
        assert dim == 160
        self.head = HybridGRULSTNetBAM(dim)

    def forward(self, xs, xi, xv):
        z = torch.cat([self.sensor(xs), self.spiral(xi), self.video(xv)], 1)   # Eq. 5
        return self.head(z)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
