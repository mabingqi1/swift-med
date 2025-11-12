import torch

import monai.transforms as mtf
from monai.transforms.spatial.array import (
    Sequence, InterpolateMode, DtypeLike, 
    MetaTensor, get_equivalent_dtype, resize
)


class ResizeOnly(mtf.Resize):
    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        mode: str = InterpolateMode.AREA,
        align_corners: bool | None = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            spatial_size,
            mode=mode, align_corners=align_corners,
            anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma,
            dtype=dtype, lazy=lazy
        )
        
    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        align_corners: bool | None = None,
        anti_aliasing: bool | None = None,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma

        input_ndim = img.ndim - 1  # spatial ndim

        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = []
        for isize, s_size in zip(img_size, self.spatial_size):
            if s_size <= 0:
                sp_size.append(isize)
            else:
                sp_size.append(s_size)

        _mode = self.mode if mode is None else mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        lazy_ = self.lazy if lazy is None else lazy
        return resize(  # type: ignore
            img,
            tuple(int(_s) for _s in sp_size),
            _mode,
            _align_corners,
            _dtype,
            input_ndim,
            anti_aliasing,
            anti_aliasing_sigma,
            lazy_,
            self.get_transform_info(),
        )