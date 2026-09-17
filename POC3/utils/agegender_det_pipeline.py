import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    ViTImageProcessor,
    AutoConfig,
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
)
from safetensors.torch import load_file

from utils.mivolo_model import MiVOLOConfig, MiVOLOForImageClassification

CONFIG_MAPPING.register("mivolo", MiVOLOConfig)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.register(
    MiVOLOConfig, MiVOLOForImageClassification
)

from insightface.app import FaceAnalysis



class MiVOLOPredictor:

    def __init__(
        self,
        config_path="/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/config.json",
        safetensors_path="/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/model.safetensors",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", self.device)

        self.processor = ViTImageProcessor(
            size={"height": 384, "width": 384},
            do_resize=True,
            do_rescale=True,
            rescale_factor=1 / 255.0,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        # ---------------------------------------------------------
        # CONFIG
        # ---------------------------------------------------------

        config = AutoConfig.from_pretrained(
            config_path,
            local_files_only=True,
        )
        print("\n========== MIVOLO CONFIG ==========")
        print("model_name       =", config.model_name)
        print("input_size       =", getattr(config, "input_size", None))
        print("in_chans         =", config.in_chans)
        print("num_classes      =", config.num_classes)
        print("num_classes_gender =", config.num_classes_gender)
        print("min_age          =", config.min_age)
        print("max_age          =", config.max_age)
        print("avg_age          =", config.avg_age)
        print("with_persons_model =", config.with_persons_model)
        print("use_persons      =", getattr(config, "use_persons", None))
        print("use_person_crops =", getattr(config, "use_person_crops", None))
        print("use_face_crops   =", getattr(config, "use_face_crops", None))
        print("===================================\n")

        # Our custom patch embedding receives:
        # face RGB (3) + person RGB (3) = 6 channels
        config.in_chans = 6
        config.num_classes = 3
        config.num_classes_gender = 2

        config.model_name = "volo_d1_384"

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------
        self.model = MiVOLOForImageClassification(config)
        
        # print("\n========== MODEL PATCH EMBED ==========")
        # for name, param in self.model.model.patch_embed.named_parameters():
        #     print(name, tuple(param.shape))
        # print("========================================\n")
        # ---------------------------------------------------------
        # LOAD SAFETENSORS
        # ---------------------------------------------------------

        raw_state_dict = load_file(
            safetensors_path,
            device="cpu",
        )
        # print("\n========== HEAD SHAPES ==========")
        # for name, value in raw_state_dict.items():
        #     if any(x in name.lower() for x in [
        #         "head",
        #         "fc",
        #         "classifier",
        #         "age",
        #         "gender",
        #     ]):
        #         print(name, tuple(value.shape))
        # print("=================================\n")

        # print("\n========== PATCH EMBED CHECKPOINT KEYS ==========")
        # for key, value in raw_state_dict.items():
        #     if "patch_embed" in key:
        #         print(f"{key:70s} {tuple(value.shape)}")
        # print("=================================================\n")

        model_state_dict = self.model.state_dict()

        cleaned_state_dict = {}

        print("\n========== CHECKPOINT LOADING ==========")

        for key, value in raw_state_dict.items():

            new_key = key

            # Remove mivolo. prefix if present
            if new_key.startswith("mivolo."):
                new_key = new_key.replace(
                    "mivolo.",
                    "",
                    1,
                )

            # Model parameters live under self.model
            if not new_key.startswith("model."):
                new_key = f"model.{new_key}"

            # Unknown key
            if new_key not in model_state_dict:
                print(f"[SKIP UNKNOWN] {new_key}")
                continue

            # Shape mismatch
            if value.shape != model_state_dict[new_key].shape:
                print(
                    f"[SKIP SHAPE] {new_key}\n"
                    f"    checkpoint: {tuple(value.shape)}\n"
                    f"    model:      {tuple(model_state_dict[new_key].shape)}"
                )
                continue

            cleaned_state_dict[new_key] = value

        print(f"\nCheckpoint tensors: {len(raw_state_dict)}")
        print(f"Compatible tensors: {len(cleaned_state_dict)}")

        # ---------------------------------------------------------
        # LOAD
        # ---------------------------------------------------------

        load_result = self.model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        print("\n========== WEIGHT LOADING ==========")
        print(
            "Missing keys:",
            len(load_result.missing_keys),
        )
        print(
            "Unexpected keys:",
            len(load_result.unexpected_keys),
        )

        if load_result.missing_keys:
            print("\nMissing keys:")
            for key in load_result.missing_keys:
                print("  ", key)

        if load_result.unexpected_keys:
            print("\nUnexpected keys:")
            for key in load_result.unexpected_keys:
                print("  ", key)

        print("====================================\n")

        # ---------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------

        self.model = self.model.to(
            self.device
        ).float()

        self.model.eval()

    @torch.no_grad()
    def predict(self, face_crop, body_crop=None):

        face_tensor = self.processor(
            images=face_crop,
            return_tensors="pt",
        )["pixel_values"]

        face_tensor = face_tensor.to(
            self.device,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # MiVOLO requires:
        # [face RGB + body RGB] = 6 channels
        # --------------------------------------------------
        if body_crop is None:
            print('body_crop is', body_crop)
            body_tensor = face_tensor.clone()
            
        else:
            body_tensor = self.processor(
                images=body_crop,
                return_tensors="pt",
            )["pixel_values"]

            body_tensor = body_tensor.to(
                self.device,
                dtype=torch.float32,
            )
            print(type(body_crop), 'body_crop is not none')

        concat_input = torch.cat(
            [face_tensor, body_tensor],
            dim=1
        )
        
        print("Face tensor:", face_tensor.shape)
        print("Body tensor:", body_tensor.shape)
        print("Concat input:", concat_input.shape)


        outputs = self.model(
            concat_input=concat_input,
            return_dict=True,
        )
        print("Model output shape:", outputs.head_outputs.shape)
        print("Raw output:", outputs.head_outputs.detach().cpu())
        
        
        age = (
            outputs.age_output
            .detach()
            .cpu()
            .flatten()[0]
            .item()
        )

        gender_score = (
            outputs.gender_probs
            .detach()
            .cpu()
            .flatten()[0]
            .item()
        )

        gender_idx = (
            outputs.gender_class_idx
            .detach()
            .cpu()
            .flatten()[0]
            .item()
        )
        gender_id2label = getattr(
            self.model.config,
            "gender_id2label",
            {
                "0": "male",
                "1": "female",
            },
        )

        gender = gender_id2label.get(
            str(int(gender_idx)),
            "unknown",
        ).capitalize()

        # gender = (
        #     "Male"
        #     if int(gender_idx) == 0
        #     else "Female"
        # )

        return {
            "age": round(float(age), 1),
            "gender": gender,
            "gender_confidence": round(
                float(gender_score),
                2,
            ),
        }


# class AgeGenderPipeline:
#     def __init__(self):
#         self.mivolo = MiVOLOPredictor()

#     # def process_image(self, face_crop):
#     #     if isinstance(face_crop, np.ndarray):
#     #         # Pad image boundaries by 20% to approximate MiVOLO's target aspect margin
#     #         h, w, _ = face_crop.shape
#     #         pad_h, pad_w = int(h * 0.2), int(w * 0.2)
#     #         face_crop = cv2.copyMakeBorder(
#     #             face_crop, pad_h, pad_h, pad_w, pad_w, 
#     #             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
#     #         )
#     #         face_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
#     #     elif not isinstance(face_crop, Image.Image):
#     #         raise TypeError("face_crop must be an OpenCV ndarray or PIL Image")

#     #     pred = self.mivolo.predict(face_crop=face_crop)
#     #     return [pred]

#     def process_image(self, face_crop):
#         if isinstance(face_crop, np.ndarray):
#             face_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
#         elif not isinstance(face_crop, Image.Image):
#             raise TypeError("face_crop must be an OpenCV ndarray or PIL Image")

#         pred = self.mivolo.predict(face_crop=face_crop)
#         return [pred]



class AgeGenderPipeline:

    def __init__(self):

        # ---------------------------------------------------------
        # Buffalo face detector
        # ---------------------------------------------------------
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

        # ---------------------------------------------------------
        # MiVOLO
        # ---------------------------------------------------------
        self.mivolo = MiVOLOPredictor()

    # -------------------------------------------------------------
    # Create larger context crop around Buffalo face bbox
    # -------------------------------------------------------------
    def make_context_crop(
        self,
        image,
        bbox,
        scale=2.5,
    ):
        x1, y1, x2, y2 = map(int,bbox)
        h, w = image.shape[:2]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bw = x2 - x1
        bh = y2 - y1

        new_w = bw * scale
        new_h = bh * scale

        nx1 = max(0, int(cx - new_w / 2))
        ny1 = max(0, int(cy - new_h / 2))
        nx2 = min(w, int(cx + new_w / 2))
        ny2 = min(h, int(cy + new_h / 2))

        return image[ny1:ny2, nx1:nx2]

    # -------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------
    def process_image(self, original_image):

        if not isinstance(original_image, np.ndarray):
            raise TypeError(
                "process_image expects an OpenCV "
                "numpy ndarray."
            )

        # ---------------------------------------------------------
        # Buffalo detection
        # ---------------------------------------------------------
        faces = self.face_app.get(original_image)

        print("Buffalo detected faces:",len(faces))

        if len(faces) == 0:
            return []

        results = []

        # ---------------------------------------------------------
        # Process every detected face
        # ---------------------------------------------------------
        for i, face in enumerate(faces):

            # Buffalo bbox
            bbox = face.bbox.astype(int)

            print(f"\n========== FACE {i} ==========")
            print("Buffalo bbox:",bbox)

            x1, y1, x2, y2 = bbox
            h, w = original_image.shape[:2]
            # Safety clipping
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            # -----------------------------------------------------
            # Tight face crop
            # -----------------------------------------------------
            face_crop = original_image[y1:y2, x1:x2]
            imgsave = f'/home/webclues-nidhi/mYpY/RnD/Age_detectoion/POC3/output/{time.time()}.jpg'
            cv2.imwrite(imgsave, face_crop)
            print('saved-------------------', imgsave)
            if face_crop.size == 0:
                print("Invalid face crop")
                continue

            # -----------------------------------------------------
            # Larger context crop
            # -----------------------------------------------------
            context_crop = self.make_context_crop(
                original_image,
                bbox,
                scale=4.5,
                )

            if context_crop.size == 0:
                print("Invalid context crop")
                continue

            print("Face crop:",face_crop.shape)
            print("Context crop:",context_crop.shape)

            # -----------------------------------------------------
            # Convert BGR -> RGB -> PIL
            # -----------------------------------------------------
            face_pil = Image.fromarray(
                cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB,
                )
            )

            context_pil = Image.fromarray(
                cv2.cvtColor(
                    context_crop,
                    cv2.COLOR_BGR2RGB,
                )
            )

            # -----------------------------------------------------
            # MiVOLO
            # -----------------------------------------------------
            pred = self.mivolo.predict(
                face_crop=face_pil,
                body_crop=context_pil,
            )

            results.append(pred)

        return results