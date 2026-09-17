import math
import torch
import torch.nn as nn
import timm
from dataclasses import dataclass
from typing import Optional, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from utils.cross_bottleneck_attn import CrossBottleneckAttn

class MiVOLOConfig(PretrainedConfig):
    model_type = "mivolo"

    def __init__(
        self,
        model_name: str = "volo_d1_384",
        num_classes: int = 3,
        in_chans: int = 3,
        with_persons_model: bool = False,
        only_age: bool = False,
        num_classes_gender: int = 2,
        min_age: float = 1.0,
        max_age: float = 95.0,
        avg_age: float = 48.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.with_persons_model = with_persons_model
        self.only_age = only_age
        self.num_classes_gender = num_classes_gender
        self.min_age = min_age
        self.max_age = max_age
        self.avg_age = avg_age
        self.initializer_range = initializer_range


@dataclass
class MiVOLOOutput(ModelOutput):
    head_outputs: Optional[torch.FloatTensor] = None
    age_output: Optional[torch.FloatTensor] = None
    raw_age_output: Optional[torch.FloatTensor] = None
    gender_probs: Optional[torch.FloatTensor] = None
    gender_class_idx: Optional[torch.FloatTensor] = None
    raw_gender_output: Optional[torch.FloatTensor] = None

class VOLOPatchEmbed(nn.Module):

    def __init__(
        self,
        img_size=384,
        stem_stride=2,
        patch_size=8,
        in_chans=6,
        hidden_dim=64,
        embed_dim=192,
    ):
        super().__init__()

        assert in_chans == 6

        self.with_persons_model = True

        self.conv1 = self.create_stem(
            stem_stride,
            3,
            hidden_dim,
        )

        self.conv2 = self.create_stem(
            stem_stride,
            3,
            hidden_dim,
        )

        projection_kernel = patch_size // stem_stride

        self.proj1 = nn.Conv2d(
            hidden_dim,
            embed_dim,
            kernel_size=projection_kernel,
            stride=projection_kernel,
        )

        self.proj2 = nn.Conv2d(
            hidden_dim,
            embed_dim,
            kernel_size=projection_kernel,
            stride=projection_kernel,
        )

        # 384 -> stem stride 2 -> 192
        # 192 -> projection stride 4 -> 48
        #
        # Therefore CrossBottleneckAttn receives:
        # [B, 384, 48, 48]
        #
        # This matches the checkpoint's spatial dimensions.
        self.map = CrossBottleneckAttn(
            embed_dim,
            dim_out=embed_dim,
            num_heads=1,
            feat_size=(48, 48),
        )

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim ** 2

    def create_stem(
        self,
        stem_stride,
        in_chans,
        hidden_dim,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_chans,
                hidden_dim,
                kernel_size=7,
                stride=stem_stride,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        if x.shape[1] != 6:
            raise ValueError(
                f"MiVOLO expects 6 channels "
                f"(face + body), got {x.shape}"
            )

        # Face stream
        face = x[:, :3]

        # Body stream
        body = x[:, 3:]

        face = self.conv1(face)
        face = self.proj1(face)

        body = self.conv2(body)
        body = self.proj2(body)

        # [B,192,48,48] + [B,192,48,48]
        # -> [B,384,48,48]
        x = torch.cat(
            [face, body],
            dim=1,
        )

        # Cross attention
        x = self.map(x)

        # IMPORTANT:
        # Return BCHW because timm VOLO expects BCHW here.
        return x


class MiVOLOPreTrainedModel(PreTrainedModel):
    config_class = MiVOLOConfig
    base_model_prefix = "mivolo"


class MiVOLOForImageClassification(MiVOLOPreTrainedModel):

    def __init__(self, config: MiVOLOConfig) -> None:
        super().__init__(config)
        self.config = config

        timm_model_name = config.model_name

        if timm_model_name.startswith("mivolo_"):
            timm_model_name = timm_model_name.replace("mivolo_", "volo_")

        # Create TIMM model first
        self.model = timm.create_model(
            model_name=timm_model_name,
            num_classes=config.num_classes,
            in_chans=config.in_chans,
            pretrained=False,
        )

        self.model.patch_embed = VOLOPatchEmbed(
            img_size=384,
            stem_stride=2,
            patch_size=8,
            in_chans=6,
            hidden_dim=64,
            embed_dim=192,
        )

        self.post_init()

    def forward(
        self,
        faces_input=None,
        body_input=None,
        concat_input=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if concat_input is None:
            if faces_input is None and body_input is None:
                raise ValueError("Provide at least faces_input or body_input.")
            model_input = faces_input if faces_input is not None else body_input
        else:
            model_input = concat_input

        output = self.model(model_input)
        print("Model output shape:", output.shape)
        print("Raw output:", output.detach().cpu())
        print("Gender logits:",output[:, :2].detach().cpu())
        print("Age raw:",output[:, 2:].detach().cpu())
        
        # Gender
        raw_gender_output = output[:, : self.config.num_classes_gender]
        gender_output = raw_gender_output.softmax(dim=-1)
        gender_probs, gender_class_idx = gender_output.topk(1, dim=-1)

        # Age
        raw_age_output = output[:, self.config.num_classes_gender :]

        if raw_age_output.shape[-1] == 1:
            print("\n========== AGE CONFIG ==========")
            print("min_age =", self.config.min_age)
            print("max_age =", self.config.max_age)
            print("avg_age =", self.config.avg_age)
            print("num_classes =", self.config.num_classes)
            print("num_classes_gender =", self.config.num_classes_gender)
            print("in_chans =", self.config.in_chans)
            print("================================\n")
            age_output = (
                raw_age_output * (self.config.max_age - self.config.min_age) / 2.0
                + self.config.avg_age
            )
        else:
            age_probs = raw_age_output.softmax(dim=-1)
            age_bins = torch.arange(
                age_probs.shape[-1], device=age_probs.device, dtype=age_probs.dtype
            )
            age_output = (age_probs * age_bins).sum(dim=-1, keepdim=True)
            age_output = (
                age_output
                * (self.config.max_age - self.config.min_age)
                / (age_probs.shape[-1] - 1)
                + self.config.min_age
            )

        if not return_dict:
            return (
                output,
                age_output,
                raw_age_output,
                gender_probs,
                gender_class_idx,
                raw_gender_output,
            )

        return MiVOLOOutput(
            head_outputs=output,
            age_output=age_output,
            raw_age_output=raw_age_output,
            gender_probs=gender_probs,
            gender_class_idx=gender_class_idx,
            raw_gender_output=raw_gender_output,
        )