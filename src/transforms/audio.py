import torch
from torch import nn
from torch.nn import functional as F


class LogPowerSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=320,
        hop_length=160,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.register_buffer(
            "window",
            torch.blackman_window(win_length),
        )

    def forward(self, waveform):
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        power_spectrum = spectrum.abs().pow(2)

        log_power_spectrum = torch.log(
            power_spectrum.clamp_min(
                torch.finfo(power_spectrum.dtype).eps
            )
        )

        return log_power_spectrum


class TrimPadFrames(nn.Module):
    def __init__(self, num_frames=750):
        super().__init__()

        self.num_frames = num_frames

    def forward(self, features):
        current_num_frames = features.shape[-1]

        if current_num_frames < self.num_frames:
            padding = self.num_frames - current_num_frames
            features = F.pad(features, (0, padding))

        elif current_num_frames > self.num_frames:
            max_start = current_num_frames - self.num_frames

            start = torch.randint(
                max_start + 1,
                size=(1,),
            ).item()

            features = features[
                ...,
                start : start + self.num_frames,
            ]

        return features