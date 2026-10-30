"""Pinned ImageNet MobileNetV3 encoder; original RGB only, never target text."""
from pathlib import Path
import hashlib
import numpy as np


class FrozenVideoEncoder:
    def __init__(self,device='cpu',cache_dir=None):
        import torch
        from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
        self.torch=torch;self.device=device
        if cache_dir:
            Path(cache_dir).mkdir(parents=True,exist_ok=True)
            torch.hub.set_dir(str(Path(cache_dir)))
        torch.set_num_threads(min(torch.get_num_threads(),4))
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.transforms=weights.transforms()
        self.model=mobilenet_v3_small(weights=weights).features.eval().to(device)
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.weight_url=weights.url
        path=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
        self.weight_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()

    def video(self,path,*,frames=16):
        import cv2
        from PIL import Image
        if not 2<=frames<=256:raise ValueError('frames must be between 2 and 256')
        cap=cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():raise ValueError(f'Cannot decode video {Path(path).name}')
            fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps<=0 or count<2:raise ValueError('Video needs valid frame count and FPS')
            positions=np.linspace(0,count-1,min(frames,count)).round().astype(int)
            tensors=[]
            for index in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
                if not ok:raise ValueError(f'Video sampling failed at frame {index}')
                tensors.append(self.transforms(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))))
            torch=self.torch
            outputs=[]
            with torch.inference_mode():
                for start in range(0,len(tensors),16):
                    batch=torch.stack(tensors[start:start+16]).to(self.device)
                    out=torch.nn.functional.adaptive_avg_pool2d(self.model(batch),1).flatten(1)
                    outputs.append(out.cpu().numpy())
            values=np.concatenate(outputs)
            feature=np.concatenate([values.mean(0),values.std(0)]).astype(np.float32)
            return feature,{'duration_sec':count/fps,'sampled_frames':len(tensors),'frame_indices':positions.tolist(),
                            'encoder':'mobilenet_v3_small.features','weights':self.weight_url,'weights_sha256':self.weight_sha256,
                            'preprocessing':str(self.transforms),'pooling':'temporal_mean_and_std_of_spatial_average_features'}
        finally:cap.release()
