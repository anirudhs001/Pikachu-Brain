import os
import shutil
import pdb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)

import torch

from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# Added for customized Processor.
import math
import numpy as np
from typing import Dict
from transformers.image_utils import PILImageResampling, ChannelDimension
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    get_resize_output_image_size,
    resize,
)
from typing import List, Optional, Tuple, Union

## ferret constants
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
## ferret consts end

DEFAULT_REGION_FEA_TOKEN = "<region_fea>"


class CLIPImageProcessor_GIT(CLIPImageProcessor):
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=True, height_width_order=True)
        # Hack(haoxuan): Bypass the shortest_edge detection. We hope to get a {"height": size[0], "width": size[1]}, where w=h.
        # if "shortest_edge" not in size:
        #     raise ValueError(f"The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}")
        # output_size = get_resize_output_image_size(image, size=size["shortest_edge"], default_to_square=True)
        output_size = get_resize_output_image_size(
            image, size=(size["height"], size["width"]), default_to_square=True
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            **kwargs,
        )


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, vision_tower_path=None):
        self.image_processor = CLIPImageProcessor_GIT.from_pretrained(
            self.vision_tower_name
        )
        if vision_tower_path is not None:
            self.vision_tower, loading_info = CLIPVisionModel.from_pretrained(
                vision_tower_path, output_loading_info=True
            )
            print("loading_info:", loading_info)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")


def rand_sample(x, max_len):
    if x.shape[0] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
    return x[rand_idx, :]


def rand_sample_repeat(x, max_len):
    if x.shape[0] < max_len:
        indices = torch.randint(0, x.shape[0], (max_len - x.shape[0],))
        # pdb.set_trace()
        return torch.cat((x, x[indices]), dim=0)
    elif x.shape[0] == max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
        return x[rand_idx, :]


def point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float(), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class ConvReLULN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvReLULN1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            self.act,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # (B, C, N) -> (B, C_1, N)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class GeoRegionSampler(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_init_point,
        num_sub_point,
        num_neighbor,
        pooler_mode="mean",
    ):
        super(GeoRegionSampler, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_init_point = num_init_point
        self.num_sub_point = num_sub_point
        self.num_neighbor = num_neighbor

        self.diff_projector_list = nn.ModuleList()
        self.agg_projector_list = nn.ModuleList()
        self.pooler_list = nn.ModuleList()

        for ii in range(len(num_sub_point)):
            self.diff_projector_list.append(
                nn.Linear(self.input_dim + 2, self.input_dim + 2)
            )
            self.agg_projector_list.append(
                ConvReLULN1D(
                    in_channels=2 * (self.input_dim + 2),
                    out_channels=self.input_dim,
                )
            )
            if pooler_mode == "mean":
                self.pooler_list.append(nn.AvgPool1d(kernel_size=num_neighbor[ii]))
            elif pooler_mode == "max":
                self.pooler_list.append(nn.AdaptiveMaxPool1d(output_size=1))
            else:
                raise NotImplementedError(f"{self.pooler_mode} is not supported.")

        self.flatten_projector = nn.Linear(
            self.input_dim * num_sub_point[-1], self.input_dim
        )
        self.dim_projector = nn.Linear(self.input_dim, self.output_dim)

        self.norm_init_weights()

    #  self.dtype = torch.float32
    def norm_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

    def forward(self, feature_map, region_masks, original_dtype, return_dtype):
        assert len(feature_map) == len(region_masks)

        all_points = []
        all_points_fea = []
        all_points_img_ids = []
        # Sample points and their features
        for img_idx, (region_feature_map_i, region_masks_list_i) in enumerate(
            zip(feature_map, region_masks)
        ):
            if len(region_masks_list_i) != 0:
                # (w, h)
                ori_image_wh = torch.tensor(
                    [region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]],
                    device=region_masks_list_i[0].device,
                )[
                    None,
                ]
                # list of elements of shape [num_sample_point, 2]
                # pdb.set_trace()
                cur_non_zero_pos = [
                    rand_sample_repeat(
                        (m.nonzero() / ori_image_wh), self.num_init_point
                    )
                    for m in region_masks_list_i
                ]
                # list -> [num_mask, num_sample_point, 2]
                cur_non_zero_pos = torch.stack(cur_non_zero_pos)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                dup_region_feature_map_i = region_feature_map_i.reshape(
                    h, w, c
                ).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(
                    cur_non_zero_pos.shape[0], 1, 1, 1
                )
                # [num_mask, C, H, W] x [num_mask, num_sample_point, 2] -> [num_mask, C, num_sample_point] -> [num_mask, num_sample_point, C]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(
                    original_dtype
                )
                region_feature_i = point_sample(
                    dup_region_feature_map_i_ori_type,
                    cur_non_zero_pos.flip(dims=(2,)).type(original_dtype),
                    return_dtype,
                    align_corners=True,
                )
                # region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                region_feature_i = region_feature_i.transpose(-2, -1)

                cur_img_ids = [img_idx] * len(cur_non_zero_pos)
                # save to global list
                all_points.append(cur_non_zero_pos)
                all_points_fea.append(region_feature_i)
                all_points_img_ids.extend(cur_img_ids)

        # pdb.set_trace()
        # No region found, return list of None.
        if len(all_points) == 0:
            return [None] * len(region_masks)

        all_points = torch.cat(all_points, dim=0).to(
            return_dtype
        )  # [B*num_mask, num_sample_point, 2]
        all_points_fea = torch.cat(
            all_points_fea, dim=0
        )  # [B*num_mask, num_sample_point, C]
        all_points_img_ids = torch.tensor(
            all_points_img_ids, device=all_points_fea.device
        )
        # pdb.set_trace()
        assert all_points_fea.shape[:-1] == all_points_fea.shape[:-1]

        # Processing.
        for stage_i in range(len(self.num_sub_point)):
            cur_num_sub_point = self.num_sub_point[stage_i]
            cur_num_neighbor = self.num_neighbor[stage_i]

            all_points = all_points.contiguous()  # xy [btach, points, xy]
            fps_idx = farthest_point_sample(all_points, cur_num_sub_point).long()

            new_points = index_points(all_points, fps_idx)  # [B, npoint, 2]
            new_points_fea = index_points(all_points_fea, fps_idx)  # [B, npoint, d]

            idx = knn_point(cur_num_neighbor, all_points, new_points)
            grouped_points = index_points(all_points, idx)  # [B, npoint, k, 2]
            grouped_points_fea = index_points(all_points_fea, idx)  # [B, npoint, k, d]

            # pdb.set_trace()
            local_points_fea = torch.cat(
                [grouped_points_fea, grouped_points], dim=-1
            )  # [B, npoint, k, d+2]
            anchor_points_fea = torch.cat(
                [new_points_fea, new_points], dim=-1
            ).unsqueeze(-2)
            diff_points_fea = local_points_fea - anchor_points_fea

            diff_points_fea = self.diff_projector_list[stage_i](diff_points_fea)
            gather_points_fea = torch.cat(
                [diff_points_fea, anchor_points_fea.repeat(1, 1, cur_num_neighbor, 1)],
                dim=-1,
            )  # [B, npoint, k, 2(d+2)]

            # pdb.set_trace()
            b, n, s, d = gather_points_fea.size()
            gather_points_fea = gather_points_fea.permute(
                0, 1, 3, 2
            )  # [B, npoint, 2(d+2), k]
            gather_points_fea = gather_points_fea.reshape(
                -1, d, s
            )  # [B*npoint, 2(d+2), k]
            gather_points_fea = self.agg_projector_list[stage_i](
                gather_points_fea
            )  # [B*npoint, d, k]
            # pdb.set_trace()
            batch_size, new_dim, _ = gather_points_fea.size()
            gather_points_fea = self.pooler_list[stage_i](gather_points_fea).view(
                batch_size, new_dim
            )  # [B*npoint, d]
            # gather_points_fea = F.adaptive_max_pool1d(gather_points_fea, 1).view(batch_size, -1) # [B*npoint, d]
            # pdb.set_trace()
            gather_points_fea = gather_points_fea.reshape(b, n, -1)  # [B, npoint, d]
            # pdb.set_trace()

            all_points = new_points
            all_points_fea = gather_points_fea

        # pdb.set_trace()
        x = all_points_fea.flatten(1, -1)  # [B, npoint x d]
        x = self.flatten_projector(x)
        all_region_fea = self.dim_projector(x)  # [B, d]

        output_region_fea = []
        for img_idx in range(len(region_masks)):
            cur_mask = all_points_img_ids == img_idx
            # pdb.set_trace()
            if not cur_mask.any():
                output_region_fea.append(None)
            else:
                output_region_fea.append(all_region_fea[cur_mask])

        # pdb.set_trace()
        return output_region_fea


class FERRETMetaModel:
    def __init__(self, config):
        super(FERRETMetaModel, self).__init__(config)
        self.max_sample_point = 512

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        if hasattr(config, "region_fea_adapter"):
            self.region_fea_adapter = nn.Linear(
                config.mm_hidden_size, config.hidden_size
            )

        if hasattr(config, "region_geo_sampler"):
            # pdb.set_trace()
            self.region_geo_sampler = GeoRegionSampler(
                input_dim=config.mm_hidden_size,
                output_dim=config.hidden_size,
                num_init_point=self.max_sample_point,
                num_sub_point=[128, 32],
                num_neighbor=[24, 24],
                pooler_mode=config.sampler_pooler_mode,
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(
        self,
        model_args,
        fsdp=None,
        add_region_feature=False,
        region_geo_sampler=False,
        sampler_pooler_mode="mean",
    ):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if add_region_feature:
            if region_geo_sampler:
                self.config.region_geo_sampler = True
                self.config.sampler_pooler_mode = sampler_pooler_mode
                # pdb.set_trace()
                if not hasattr(self, "region_geo_sampler"):
                    self.region_geo_sampler = GeoRegionSampler(
                        input_dim=self.config.mm_hidden_size,
                        output_dim=self.config.hidden_size,
                        num_init_point=self.max_sample_point,
                        num_sub_point=[128, 32],
                        num_neighbor=[24, 24],
                        pooler_mode=sampler_pooler_mode,
                    )
            else:
                self.config.region_fea_adapter = True
                if not hasattr(self, "region_fea_adapter"):
                    self.region_fea_adapter = nn.Linear(
                        self.config.mm_hidden_size, self.config.hidden_size
                    )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class FERRETMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, region_flag=False, region_geo_sampler=False):
        image_features = self.get_model().get_vision_tower()(images)
        projected_image_features = self.get_model().mm_projector(image_features)

        if region_flag:
            if region_geo_sampler:
                new_region_feature_map = image_features
            else:
                new_region_feature_map = self.get_model().region_fea_adapter(
                    image_features
                )
        else:
            new_region_feature_map = None

        return image_features, projected_image_features, new_region_feature_map

    def extract_region_feature(
        self, region_feature_map, region_masks, original_dtype, return_dtype
    ):
        all_region_features = []
        assert len(region_feature_map) == len(region_masks)
        for region_feature_map_i, region_masks_list_i in zip(
            region_feature_map, region_masks
        ):
            if len(region_masks_list_i) == 0:
                all_region_features.append(None)
            else:
                # (w, h)
                ori_image_wh = torch.tensor(
                    [region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]],
                    device=region_masks_list_i[0].device,
                )[
                    None,
                ]
                # list of elements of shape [num_sample_point, 2]
                non_zero_pos = [
                    rand_sample(
                        (m.nonzero() / ori_image_wh), self.get_model().max_sample_point
                    )
                    for m in region_masks_list_i
                ]
                # [num_mask, num_sample_point(padded), 2]
                non_zero_pos = nn.utils.rnn.pad_sequence(
                    non_zero_pos, padding_value=-1, batch_first=True
                )
                non_zero_pos_mask = ~(non_zero_pos.sum(dim=-1) < 0)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                dup_region_feature_map_i = region_feature_map_i.reshape(
                    h, w, c
                ).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(
                    non_zero_pos.shape[0], 1, 1, 1
                )
                # [num_mask, C, H, W] x [num_mask, num_sample_point(padded), 2] -> [num_mask, C, num_sample_point(padded)]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(
                    original_dtype
                )
                # pdb.set_trace()
                region_feature_i = point_sample(
                    dup_region_feature_map_i_ori_type,
                    non_zero_pos.flip(dims=(2,)).type(original_dtype),
                    return_dtype,
                    align_corners=True,
                )
                region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                # [num_mask, C]
                region_feature_i = torch.stack(
                    [
                        x[m].mean(dim=0)
                        for x, m in zip(
                            region_feature_i.transpose(1, 2), non_zero_pos_mask
                        )
                    ]
                ).nan_to_num()
                all_region_features.append(region_feature_i)

        return all_region_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, region_masks
    ):
        if region_masks is not None:
            region_flag = True
        else:
            region_flag = False
        region_geo_sampler = region_flag and getattr(
            self.config, "region_geo_sampler", False
        )

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            assert region_flag == False
            concat_images = torch.cat([image for image in images], dim=0)
            raw_image_features, image_features, region_feature_map = self.encode_images(
                concat_images, region_flag, region_geo_sampler
            )
            # image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            raw_image_features, image_features, region_feature_map = self.encode_images(
                images, region_flag, region_geo_sampler
            )

        if region_flag:
            if region_geo_sampler:
                # pdb.set_trace()
                region_features = self.get_model().region_geo_sampler(
                    region_feature_map,
                    region_masks,
                    original_dtype=raw_image_features.dtype,
                    return_dtype=image_features.dtype,
                )
            else:
                region_features = self.extract_region_feature(
                    region_feature_map,
                    region_masks,
                    original_dtype=raw_image_features.dtype,
                    return_dtype=image_features.dtype,
                )
            assert len(region_features) == len(input_ids)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if region_flag:
                    assert (
                        cur_input_ids[:image_token_start]
                        == self.config.im_region_fea_token
                    ).sum() == 0
                # If not use start-end token, pt ckpt saved only has mm projector.
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    text_input_embeds = (
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                else:
                    text_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                if labels is not None:
                    cur_new_labels.append(cur_labels)

                # Add region feature into text feature embeddings.
                assert batch_idx + 1 == cur_image_idx
                if region_flag and region_features[batch_idx] is not None:
                    region_embs = torch.zeros_like(text_input_embeds)
                    region_replace_mask = (
                        cur_input_ids == self.config.im_region_fea_token
                    )
                    # pdb.set_trace()
                    region_embs[region_replace_mask] = region_features[batch_idx].to(
                        text_input_embeds.dtype
                    )
                    text_input_embeds = (
                        text_input_embeds
                        * (~region_replace_mask).to(text_input_embeds.dtype)[:, None]
                        + region_embs
                    )
                    # print('region_embs[..., 0].nonzero()', region_embs[..., 0].nonzero())
                    # raise NotImplementedError()
                    # pdb.set_trace()
                else:
                    if hasattr(self.config, "im_region_fea_token"):
                        assert (
                            cur_input_ids == self.config.im_region_fea_token
                        ).sum() == 0

                cur_new_input_embeds.append(text_input_embeds)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(
        self, model_args, tokenizer, add_region_feature=False
    ):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if add_region_feature:
            num_region_fea_tokens = tokenizer.add_tokens(
                [DEFAULT_REGION_FEA_TOKEN], special_tokens=True
            )
            self.config.im_region_fea_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_REGION_FEA_TOKEN]
            )[0]
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if add_region_feature:
                num_new_tokens = num_new_tokens + num_region_fea_tokens

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                if add_region_feature:
                    num_new_tokens = num_new_tokens - num_region_fea_tokens
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


class FERRETConfig(LlamaConfig):
    model_type = "ferret"


class FERRETLlamaModel(FERRETMetaModel, LlamaModel):
    config_class = FERRETConfig

    def __init__(self, config: LlamaConfig):
        super(FERRETLlamaModel, self).__init__(config)


class FERRETLlamaForCausalLM(LlamaForCausalLM, FERRETMetaForCausalLM):
    config_class = FERRETConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = FERRETLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        region_masks: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            region_masks=region_masks,
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


# AutoConfig.register("ferret", FERRETConfig)
# AutoModelForCausalLM.register(FERRETConfig, FERRETLlamaForCausalLM)


def load_pretrained_model(
    model_path, model_name, load_8bit=False, load_4bit=False, device_map="cpu"
):
    kwargs = {"device_map": "mps"}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = FERRETLlamaForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )
    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_im_region_fea_token = getattr(model.config, "im_region_fea_token", None)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_im_region_fea_token is not None:
        tokenizer.add_tokens([DEFAULT_REGION_FEA_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower_path = os.path.join(model_path, "vision_tower")
    if not vision_tower.is_loaded or os.path.exists(vision_tower_path):
        if os.path.exists(vision_tower_path):
            print(f"Start Loading vision tower from {vision_tower_path}")
            vision_tower.load_model(vision_tower_path=vision_tower_path)
            print(f"Finish Loading vision tower from {vision_tower_path}")
        else:
            vision_tower.load_model()

    vision_tower.to(device="mps", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model = model.to(device_map)
    return tokenizer, model, image_processor, context_len


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

if __name__ == "__main__":
    tokenizer, model, image_processer, context_len = load_pretrained_model(
        model_path="/Users/anirudhsingh/MISC/playground/ml-ferret/ferret/model/ferret-7b-v1-3",
        model_name="ferret-7b-v1-3",
    )
    print(tokenizer)
    print(model)
    print(image_processer)
    print(context_len)
