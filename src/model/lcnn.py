from torch import nn

from src.model.mfm import MFM


class LCNN(nn.Module):
    def __init__(
        self,
        n_classes=2,
        dropout=0.75,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(48),

            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(64),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MFM(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 16 * 46,
                out_features=160,
            ),
            MFM(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(80),
            nn.Linear(
                in_features=80,
                out_features=n_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data_object, **batch):
        features = self.cnn(data_object)
        logits = self.classifier(features)

        return {"logits": logits}

    def __str__(self):
        all_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += (
            f"\nTrainable parameters: {trainable_parameters}"
        )

        return result_info