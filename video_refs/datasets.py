from typing import Tuple

import torchvision
from torch import Tensor


class KineticsWithVideoId(torchvision.datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx


class Hmdb51WithVideoId(torchvision.datasets.HMDB51):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, int]:
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, class_index, video_idx