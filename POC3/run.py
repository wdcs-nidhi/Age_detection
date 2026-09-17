import cv2
from utils.agegender_det_pipeline import ( AgeGenderPipeline, )


from huggingface_hub import hf_hub_download

# 1. Download the weights file
checkpoint_path = hf_hub_download(
    repo_id="iitolstykh/mivolo_v2", 
    filename="model.safetensors",
    local_dir="./weights"
)
print(f"Downloaded weights file directly to: {checkpoint_path}")

# 2. Download the config.json file to the same directory
config_path = hf_hub_download(
    repo_id="iitolstykh/mivolo_v2", 
    filename="config.json",
    local_dir="./weights"
)
print(f"Downloaded config file directly to: {config_path}")



# if __name__ == "__main__":

#     pipeline = AgeGenderPipeline(
#         face_model="buffalo_l",
#         person_model="yolov8n.pt",
#         mivolo_config=("/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/config.json"),
#         mivolo_weights=("/home/webclues-nidhi/mYpY/RnD/Age_detectoion/models/model.safetensors"),
#         device="cpu",
#     )
#     image_path = ("/home/webclues-nidhi/mYpY/Data/imgs/age/divy.png")
#     output_path = ("/home/webclues-nidhi/mYpY/RnD/Age_detectoion/output_debug.jpg")

#     image = cv2.imread(image_path)

#     if image is None:
#         raise FileNotFoundError( f"Image not found: {image_path}" )

#     output = pipeline.process_image( image, draw=True, )

#     print("\nInference Results:")
#     for result in output["results"]:
#         print( result )
#     cv2.imwrite( output_path, output["image"], )
#     print( "\nSaved:", output_path, )

