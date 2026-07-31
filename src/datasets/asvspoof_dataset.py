from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):
    def __init__(
        self,
        audio_dir,
        protocol_path,
        *args,
        **kwargs,
    ):
        index = self._create_index(audio_dir, protocol_path)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, audio_dir, protocol_path):
        audio_dir = Path(audio_dir)
        protocol_path = Path(protocol_path)

        index = []

        with protocol_path.open("r") as protocol:
            for line in protocol:
                _, utterance_id, _, _, label = line.strip().split()

                audio_path = audio_dir / f"{utterance_id}.flac"
                target = 1 if label == "bonafide" else 0

                index.append(
                    {
                        "path": str(audio_path),
                        "label": target,
                    }
                )

        return index

    def load_object(self, path):
        waveform, _ = torchaudio.load(path)

        return waveform