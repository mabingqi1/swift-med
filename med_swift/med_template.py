import torch
from typing import List, Literal

import monai.transforms as mtf
import numpy as np
import SimpleITK as sitk

from swift.llm import register_template
from swift.llm.template.template_meta import TemplateMeta
from swift.llm.template.template.internvl import InternvlTemplate
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context

from med_swift.med_data.io import read_zst
from med_swift.med_data.transforms import ResizeOnly

class MedIntern3_5VLTemplate(InternvlTemplate):
    """ Intern3.5vl Template
    """
    def _get_system(self, inputs: StdTemplateInputs) -> str:
        """忽略 system，返回空字符串"""
        return ''
    
    def replace_tag(self, media_type: Literal['image', 'video'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}

        # TODO set to config file
        pixel_mean = 40
        pixel_std = 40
        resize_size = (-1, 336, 336)
        spatial_size = (64, 336, 336)
        pad_values = -2000
        
        # transform
        transform_train = mtf.Compose([
            ResizeOnly(resize_size, mode='trilinear', align_corners=True, lazy=True),
            mtf.ResizeWithPadOrCrop(spatial_size, constant_values=pad_values, lazy=True),
            mtf.ScaleIntensityRange(a_min=pixel_mean-pixel_std, a_max=pixel_mean+pixel_std, b_min=0, b_max=1, clip=True),
        ])
        transform_eval = transform_train
        transform = transform_train if self.is_training else transform_eval
        
        # read
        if media_type == 'video':
            video_path = inputs.videos[index]
        else:
            video_path = inputs.images[index]
            
        if video_path.endswith('.zst'):
            video = read_zst(video_path)
        elif video_path.endswith('.npy'):
            video = np.load(video_path)
        else:
            image = sitk.ReadImage(video_path)
            video = sitk.GetArrayFromImage(image)
        
        with torch.no_grad():
            video = torch.from_numpy(np.ascontiguousarray(video.transpose(2, 1, 0)))[None]
            video = transform(video)[0].clamp_(0, 1).mul_(255).to(torch.uint8)
            video = video[:, None]  # (D, 1, H, W)

        from PIL import Image
        frame_tokens: List[Context] = []

        for frame in video:
            frame = frame.repeat(3, 1, 1) if frame.shape[0] == 1 else frame
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            inputs.images.append(Image.fromarray(frame_np, mode='RGB'))
            frame_tokens.extend(['<img>', [-100], '</img>\n'])

        del video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if media_type == 'video':
            tokens = ['<video>'] + frame_tokens + ['</video>']

        return tokens

register_template(
    TemplateMeta(
        template_type="med-InternVL3_5",
        template_cls=MedIntern3_5VLTemplate,
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=None,
        suffix=['<|im_end|>'],
    )
)