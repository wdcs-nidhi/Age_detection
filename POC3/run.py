import cv2
from utils.agegender_det_pipeline import AgeGenderPipeline

if __name__ == "__main__":

    pipeline = AgeGenderPipeline()

    image = cv2.imread(
        "/home/webclues-nidhi/mYpY/Data/imgs/age/divy.png"
    )   
    if image is None:
        raise FileNotFoundError(
            "Image not found at path."
        )
    results = pipeline.process_image(image)
   
    print(
        "Inference Results:",
        results
    )


# from safetensors.torch import load_file

# path = "/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/model.safetensors"

# state = load_file(path, device="cpu")

# print("\n========== ALL CHECKPOINT KEYS ==========")

# for k, v in state.items():
#     print(f"{k:90s} {tuple(v.shape)}")

# print("========================================")




# from safetensors.torch import load_file
# from transformers import AutoConfig
# from utils.mivolo_model import MiVOLOForImageClassification

# # 1. Instantiate model
# config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
# config.model_name = "volo_d1_384"
# config.in_chans = 3
# model = MiVOLOForImageClassification(config)

# # 2. Load raw keys from safetensors
# safetensors_path: str = "/home/aiml/projects/spotem/ai_models/v3/A0042/mivolo/model.safetensors"
# raw_state_dict = load_file(safetensors_path, device="cpu")

# # 3. Print samples side by side
# model_keys = set(model.state_dict().keys())
# ckpt_keys = list(raw_state_dict.keys())

# print("--- Checkpoint Keys Sample (First 5) ---")
# for k in ckpt_keys[:5]:
#     print(k)

# print("\n--- Model Architecture Keys Sample (First 5) ---")
# for k in list(model_keys)[:5]:
#     print(k)


# from huggingface_hub import hf_hub_download

# # Downloads only the raw file path on disk without initializing the transformers model
# checkpoint_path = hf_hub_download(
#     repo_id="iitolstykh/mivolo_v2", 
#     filename="model.safetensors",  # or "pytorch_model.bin" depending on revision
#     local_dir="./weights"
# )

# print(f"Downloaded weights file directly to: {checkpoint_path}")