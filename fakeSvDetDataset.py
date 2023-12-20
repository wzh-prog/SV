import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor

from  torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
import random

class FKSV(VisionDataset):
    def __init__(
        self,
        root: str, # video
        jsonFile: str,
        frames_per_clip: int,
        step_between_clips: int = 1,

        frame_rate: Optional[int] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC",
    ) -> None:
        super().__init__(root)

        events, events_to_idx = find_enents(jsonFile)

        self.samples = get_samples(
            jsonFile,
            events_to_idx
        )

        # self.samples = self.samples[:50]

        video_paths = [os.path.join(self.root, path) for (path, _, _) in self.samples]
        print("total videos: ", len(video_paths))
        video_clips = VideoClips(
            video_paths,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )
        # we bookkeep the full version of video clips because we want to be able
        # to return the metadata of full version rather than the subset version of
        # video clips
        self.full_video_clips = video_clips
        self.train = train
        # self.indices = random.sample(list(a for a in range(len(self.samples))), 5)
        self.indices = list(a for a in range(len(self.samples)))
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
        self.events = events

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, _av_info, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, ann_, event_idx = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        # return video, audio, _av_info, ann_, event_idx
        return video,  ann_, event_idx

import json
def find_enents(jsonFile: str):
    # Opening JSON file
    f = open(jsonFile)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    events = set()
    for item_ in data:
        if item_["keywords"] not in events:
            events.add(item_["keywords"])
    events = list(events)
    events_to_idx = {cls_name: i for i, cls_name in enumerate(events)}
    '''
    '上海浦东车管所暂停为特斯拉车上牌': 0,
    '下雨天过横道线不可踏上斑马线的白漆面': 1,
    '湖南长沙国金中心起火': 2,
    '成都集资修地铁': 3,
    '橡皮蛋、假鸡蛋、人造蛋': 4,
    '齐齐哈尔发洪水': 5,
    '''
    return events, events_to_idx

def get_samples(
    jsonFile: str,
    events_to_idx: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, int]]:
    # Opening JSON file
    f = open(jsonFile)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    instances = []
    
    ann = {"假": 0, "真": 1, "辟谣": 2}
    # ann = {1: 0, 2: 1, 0: 2}

    for item_ in data:
        videoPath = item_["video_id"] + ".mp4"
        ann_ = ann[item_["annotation"]]
        event_idx = events_to_idx[item_["keywords"]]
        instances.append((videoPath, ann_, event_idx))

    return instances

