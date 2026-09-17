import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from safetensors.torch import load_file
from transformers import ViTImageProcessor
# from transformers.modeling_outputs import ModelOutput

from .cross_bottleneck_attn import CrossBottleneckAttn

from dataclasses import dataclass

from transformers.utils import ModelOutput


@dataclass
class MiVOLOOutput(ModelOutput):
    logits: torch.Tensor
    age: Optional[torch.Tensor] = None
    gender: Optional[torch.Tensor] = None

# ============================================================
# CONFIG
# ============================================================

class MiVOLOConfig:

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model_name = getattr(
            self,
            "model_name",
            "mivolo_d5_512",
        )

        self.num_classes = getattr(
            self,
            "num_classes",
            3,
        )

        self.num_classes_gender = getattr(
            self,
            "num_classes_gender",
            2,
        )

        self.min_age = getattr(
            self,
            "min_age",
            0,
        )

        self.max_age = getattr(
            self,
            "max_age",
            122,
        )

        self.avg_age = getattr(
            self,
            "avg_age",
            61.0,
        )

        self.input_size = getattr(
            self,
            "input_size",
            384,
        )

        self.use_persons = getattr(
            self,
            "use_persons",
            True,
        )

        self.use_person_crops = getattr(
            self,
            "use_person_crops",
            True,
        )

        self.use_face_crops = getattr(
            self,
            "use_face_crops",
            True,
        )

        self.with_persons_model = getattr(
            self,
            "with_persons_model",
            True,
        )

        self.only_age = getattr(
            self,
            "only_age",
            False,
        )


# ============================================================
# OUTPUT
# ============================================================

# class MiVOLOOutput(ModelOutput):

#     logits: Optional[torch.Tensor] = None
#     age: Optional[torch.Tensor] = None
#     gender: Optional[torch.Tensor] = None
#     gender_confidence: Optional[torch.Tensor] = None


# ============================================================
# PATCH EMBED
# ============================================================

class VOLOPatchEmbed(nn.Module):

    def __init__(
        self,
        img_size=384,
        stem_conv=True,
        stem_stride=2,
        patch_size=8,
        in_chans=6,
        hidden_dim=64,
        embed_dim=192,
    ):
        super().__init__()

        assert patch_size in [4, 8, 16]
        assert in_chans in [3, 6]

        self.with_persons_model = in_chans == 6
        self.use_cross_attn = True

        if stem_conv:

            if not self.with_persons_model:

                self.conv = self.create_stem(
                    stem_stride,
                    in_chans,
                    hidden_dim,
                )

            else:

                # Keep the same interface as MiVOLO.
                self.conv = True

            if self.with_persons_model:

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

        else:

            self.conv = None

        # ----------------------------------------------------
        # Face + person
        # ----------------------------------------------------

        if self.with_persons_model:

            projection_stride = (
                patch_size // stem_stride
            )

            self.proj1 = nn.Conv2d(
                hidden_dim,
                embed_dim,
                kernel_size=projection_stride,
                stride=projection_stride,
            )

            self.proj2 = nn.Conv2d(
                hidden_dim,
                embed_dim,
                kernel_size=projection_stride,
                stride=projection_stride,
            )

            stem_out_shape = (
                self.get_output_size_module(
                    (img_size, img_size),
                    self.conv1,
                )
            )

            self.proj_output_size = (
                self.get_output_size(
                    stem_out_shape,
                    self.proj1,
                )
            )

            # Keep your existing local CrossBottleneckAttn
            # signature.
            self.map = CrossBottleneckAttn(
                embed_dim,
                dim_out=embed_dim,
                num_heads=1,
                feat_size=self.proj_output_size,
            )

        else:

            projection_stride = (
                patch_size // stem_stride
            )

            self.proj = nn.Conv2d(
                hidden_dim,
                embed_dim,
                kernel_size=projection_stride,
                stride=projection_stride,
            )

            self.patch_dim = (
                img_size // patch_size
            )

            self.num_patches = (
                self.patch_dim ** 2
            )

    # ========================================================
    # STEM
    # ========================================================

    @staticmethod
    def create_stem(
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

    # ========================================================
    # OUTPUT SIZE
    # ========================================================

    @staticmethod
    def get_output_size(
        input_shape,
        conv_layer,
    ):

        padding = conv_layer.padding
        dilation = conv_layer.dilation
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride

        return [
            (
                (
                    input_shape[i]
                    + 2 * padding[i]
                    - dilation[i]
                    * (kernel_size[i] - 1)
                    - 1
                )
                // stride[i]
            )
            + 1
            for i in range(2)
        ]

    @staticmethod
    def get_output_size_module(
        input_size,
        stem,
    ):

        output_size = list(input_size)

        for module in stem:

            if isinstance(
                module,
                nn.Conv2d,
            ):

                output_size = [
                    (
                        (
                            output_size[i]
                            + 2 * module.padding[i]
                            - module.dilation[i]
                            * (module.kernel_size[i] - 1)
                            - 1
                        )
                        // module.stride[i]
                    )
                    + 1
                    for i in range(2)
                ]

        return output_size

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        if x.ndim != 4:

            raise ValueError(
                f"Expected BCHW input, got {tuple(x.shape)}"
            )

        if self.with_persons_model:

            if x.shape[1] != 6:

                raise ValueError(
                    "Expected 6-channel input "
                    "(3 face + 3 person), "
                    f"got {x.shape[1]}"
                )

            face = x[:, :3]
            person = x[:, 3:]

            face = self.conv1(face)
            face = self.proj1(face)

            person = self.conv2(person)
            person = self.proj2(person)

            # 192 + 192 = 384
            x = torch.cat(
                [face, person],
                dim=1,
            )

            x = self.map(x)

            return x

        x = self.conv(x)
        x = self.proj(x)

        return x


# ============================================================
# MODEL
# ============================================================

class MiVOLOForImageClassification(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config

        import timm

        # IMPORTANT:
        #
        # The checkpoint contains:
        #
        #   mivolo.model.patch_embed...
        #   mivolo.model.network...
        #   mivolo.model.head...
        #   mivolo.model.aux_head...
        #   mivolo.model.post_network...
        #
        # After removing the "mivolo." prefix, the keys become:
        #
        #   model.patch_embed...
        #   model.network...
        #   model.head...
        #
        # Therefore the actual backbone MUST be called "model".
        self.model = timm.create_model(
            "volo_d1_384",
            pretrained=False,
            num_classes=3,
            in_chans=6,
        )

        # Replace standard VOLO patch embedding with
        # MiVOLO face/person patch embedding.
        self.model.patch_embed = VOLOPatchEmbed(
            img_size=384,
            stem_conv=True,
            stem_stride=2,
            patch_size=8,
            in_chans=6,
            hidden_dim=64,
            embed_dim=192,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        pixel_values=None,
        faces_input=None,
        body_input=None,
        **kwargs,
    ):

        if pixel_values is None:

            if faces_input is None:
                raise ValueError(
                    "faces_input or pixel_values "
                    "must be provided."
                )

            if body_input is None:
                pixel_values = faces_input

            else:
                pixel_values = torch.cat(
                    [
                        faces_input,
                        body_input,
                    ],
                    dim=1,
                )

        # Run the complete MiVOLO/VOLO model.
        logits = self.model(
            pixel_values
        )

        if isinstance(
            logits,
            (tuple, list),
        ):
            logits = logits[0]

        if logits.ndim != 2:
            raise RuntimeError(
                "Unexpected MiVOLO output shape: "
                f"{tuple(logits.shape)}"
            )

        if logits.shape[1] != 3:
            raise RuntimeError(
                "Expected 3 MiVOLO outputs "
                "(male, female, age), "
                f"got {logits.shape[1]}"
            )

        return MiVOLOOutput(
            logits=logits
        )



# ============================================================
# PREDICTOR
# ============================================================

class MiVOLOPredictor:
    def __init__(
        self,
        config_path,
        weights_path,
        device="cpu",
    ):
        print("\nCreating MiVOLO model...")
        print(f"Loading checkpoint:\n{weights_path}")

        self.device = torch.device(device)

        # ---------------------------------------------------------
        # Load configuration
        # ---------------------------------------------------------
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        self.config = MiVOLOConfig(**config_dict)

        # Age decoding parameters
        self.min_age = float(self.config.min_age)
        self.max_age = float(self.config.max_age)
        self.avg_age = float(self.config.avg_age)

        print(
            f"Age config: "
            f"min={self.min_age}, "
            f"max={self.max_age}, "
            f"avg={self.avg_age}"
        )

        # ---------------------------------------------------------
        # Create model
        # ---------------------------------------------------------
        self.model = MiVOLOForImageClassification(
            self.config
        )

        # ---------------------------------------------------------
        # Load safetensors checkpoint
        # ---------------------------------------------------------
        from safetensors.torch import load_file

        checkpoint = load_file(
            weights_path,
            device="cpu",
        )

        print(f"Checkpoint tensors : {len(checkpoint)}")

        # ---------------------------------------------------------
        # Normalize checkpoint keys
        #
        # Checkpoint:
        #
        #   mivolo.model.patch_embed...
        #
        # Model:   model.patch_embed...
        # ---------------------------------------------------------
        normalized_checkpoint = {}

        for key, value in checkpoint.items():

            new_key = key

            if new_key.startswith("mivolo."):
                new_key = new_key[len("mivolo."):]

            normalized_checkpoint[new_key] = value

        # ---------------------------------------------------------
        # Check compatibility BEFORE loading
        # ---------------------------------------------------------
        model_state = self.model.state_dict()

        compatible = {}
        skipped = []

        for key, value in normalized_checkpoint.items():

            if key not in model_state:
                skipped.append(key)
                continue

            if model_state[key].shape != value.shape:
                skipped.append(
                    f"{key}: "
                    f"checkpoint={tuple(value.shape)} "
                    f"model={tuple(model_state[key].shape)}"
                )
                continue

            compatible[key] = value

        missing = [
            key
            for key in model_state.keys()
            if key not in compatible
        ]

        unexpected = [
            key
            for key in normalized_checkpoint.keys()
            if key not in compatible
        ]

        print(f"Compatible tensors : {len(compatible)}")
        print(f"Skipped tensors    : {len(skipped)}")
        print(f"Missing model keys : {len(missing)}")
        print(f"Unexpected keys    : {len(unexpected)}")

        # ---------------------------------------------------------
        # NEVER silently run with a partially loaded model
        # ---------------------------------------------------------
        if missing or unexpected or skipped:
            print("\nCheckpoint/model mismatch detected.")

            if missing:
                print("\nMissing keys:")
                for key in missing[:20]: print("  ", key)

            if unexpected:
                print("\nUnexpected keys:")
                for key in unexpected[:20]: print("  ", key)

            if skipped:
                print("\nSkipped keys:")
                for key in skipped[:20]: print("  ", key)

            raise RuntimeError( "MiVOLO checkpoint does not exactly match the model architecture." )

        # ---------------------------------------------------------
        # Load
        # ---------------------------------------------------------
        self.model.load_state_dict( compatible, strict=True )
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # Image preprocessing
        # ---------------------------------------------------------
        self.processor = ViTImageProcessor(
            do_resize=True,
            size={
                "height": int(self.config.input_size),
                "width": int(self.config.input_size),
            },
            do_center_crop=False,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=[ 0.485, 0.456, 0.406 ],
            image_std=[ 0.229, 0.224, 0.225 ],
        )

        print("MiVOLO loaded.")


    # ========================================================
    # IMAGE -> RGB NUMPY
    # ========================================================

    @staticmethod
    def _to_rgb_numpy(image):

        # ----------------------------------------------------
        # PIL
        # ----------------------------------------------------

        if isinstance(
            image,
            Image.Image,
        ):

            image = image.convert(
                "RGB"
            )

            return np.asarray(
                image,
                dtype=np.uint8,
            )

        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        if isinstance(
            image,
            np.ndarray,
        ):

            if image.ndim != 3:

                raise ValueError(
                    "Expected HWC image, "
                    f"got {image.shape}"
                )

            # Assume NumPy images coming from
            # OpenCV are BGR.
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        raise TypeError(
            "Expected PIL.Image.Image or "
            f"numpy.ndarray, got {type(image)}"
        )

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    def _prepare_image(
        self,
        image,
    ):

        image_rgb = (
            self._to_rgb_numpy(
                image
            )
        )

        processed = self.processor(
            images=image_rgb,
            return_tensors="pt",
        )

        return processed[
            "pixel_values"
        ].to(
            self.device
        )

    # ========================================================
    # PREDICT
    # ========================================================
    @torch.no_grad()
    def predict(self, face_image, person_image=None):
        """
        MiVOLO inference.

        MiVOLO output layout:

            logits[:, 0] = male
            logits[:, 1] = female
            logits[:, 2] = normalized age

        Age decoding uses the values from config.json:

            age = raw_age * (max_age - min_age) + avg_age
        """

        # =========================================================
        # PREPARE FACE IMAGE
        # =========================================================
        face_tensor = self._prepare_image(face_image)

        # =========================================================
        # PREPARE PERSON IMAGE
        # =========================================================
        if person_image is not None:
            person_tensor = self._prepare_image(person_image)
        else:
            # For a face-only fallback, use the face crop
            # for both MiVOLO input streams.
            person_tensor = face_tensor.clone()

        # =========================================================
        # MiVOLO INPUT
        #
        # First 3 channels  = face
        # Last 3 channels   = person
        # =========================================================
        pixel_values = torch.cat(
            [face_tensor, person_tensor],
            dim=1,
        )

        pixel_values = pixel_values.to(self.device)

        # =========================================================
        # INFERENCE
        # =========================================================
        output = self.model(
            pixel_values=pixel_values
        )

        logits = output.logits

        # Make absolutely sure we're working with [B, 3]
        if logits.ndim != 2 or logits.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected MiVOLO output shape: {tuple(logits.shape)}. "
                f"Expected [batch, 3]."
            )

        # =========================================================
        # IMPORTANT:
        # MiVOLO output:
        # logits[:, 0] = male
        # logits[:, 1] = female
        # logits[:, 2] = normalized age
        # =========================================================
        male_logit = logits[:, 0]
        female_logit = logits[:, 1]
        raw_age = logits[:, 2]

        # =========================================================
        # GENDER
        # =========================================================        #

        gender_logits = logits[:, 0:2]
        raw_age = logits[:, 2]

        # -------------------------
        # Gender
        # -------------------------
        gender_probs = torch.softmax(
            gender_logits,
            dim=1,
        )

        gender_id = torch.argmax(
            gender_probs,
            dim=1,
        )

        gender_confidence = torch.max(
            gender_probs,
            dim=1,
        ).values

        gender_names = {
            0: "male",
            1: "female",
        }

        gender = gender_names[
            int(gender_id[0].item())
        ]

        # -------------------------
        # Age
        # -------------------------
        age = (
            raw_age * 122.0
            + 61.0
        )

        age = torch.clamp(
            age,
            min=0.0,
            max=122.0,
        )

        age_value = float(
            age[0].item()
        )

        result = {
            "age": round(age_value, 2),
            "gender": gender,
            "gender_confidence": float(
                gender_confidence[0].item()
            ),
            "raw_logits": [
                float(x)
                for x in logits[0]
                .detach()
                .cpu()
                .tolist()
            ],
            "raw_age": float(
                raw_age[0].item()
            ),
        }

        return result



    # ========================================================
    # BATCH
    # ========================================================

    @torch.no_grad()
    def predict_batch(
        self,
        face_images,
        person_images=None,
    ):

        if not face_images:

            return []

        face_tensors = [
            self._prepare_image(
                image
            )
            for image in face_images
        ]

        face_tensor = torch.cat(
            face_tensors,
            dim=0,
        )

        if person_images is None:

            person_tensor = (
                face_tensor.clone()
            )

        else:

            person_tensors = [
                self._prepare_image(
                    image
                )
                for image in person_images
            ]

            person_tensor = torch.cat(
                person_tensors,
                dim=0,
            )

        pixel_values = torch.cat(
            [
                face_tensor,
                person_tensor,
            ],
            dim=1,
        )

        output = self.model(
            pixel_values=pixel_values
        )

        logits = output.logits

        raw_age = logits[:, 0]

        age = (
            raw_age
            * (
                self.config.max_age
                - self.config.min_age
            )
            / 2.0
            + self.config.avg_age
        )

        age = torch.clamp(
            age,
            min=self.config.min_age,
            max=self.config.max_age,
        )

        gender_logits = logits[
            :,
            1:3,
        ]

        gender_prob = torch.softmax(
            gender_logits,
            dim=1,
        )

        gender_ids = torch.argmax(
            gender_prob,
            dim=1,
        )

        results = []

        for i in range(
            len(face_images)
        ):

            gender_id = int(
                gender_ids[i]
            )

            gender = (
                "male"
                if gender_id == 0
                else "female"
            )

            results.append(
                {
                    "age": float(
                        age[i].cpu()
                    ),

                    "gender": gender,

                    "gender_confidence": float(
                        gender_prob[
                            i,
                            gender_id,
                        ].cpu()
                    ),

                    "raw_logits": [
                        float(x)
                        for x in logits[
                            i
                        ].cpu()
                    ],
                }
            )

        return results
