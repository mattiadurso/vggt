import os
from vggt.wrapper import VGGTWrapper

vggt = VGGTWrapper(cuda_id=4)

base_path = "/data/mdurso"

dataset = "imc"
scenes = os.listdir(f"{base_path}/{dataset}")

for scene in scenes:
    input = f"{base_path}/{dataset}/{scene}/set_100/images"
    output = f"{base_path}/results/vggt/{dataset}/{scene}/sparse"
    os.makedirs(output, exist_ok=True)

    # reconstruction
    rec = vggt.forward(input, output, max_images=150, use_ba=True)


dataset = "eth3d"
scenes = os.listdir(f"{base_path}/{dataset}")

for scene in scenes:
    input = f"{base_path}/eth3d/{scene}/images_by_k"
    output = f"{base_path}/results/vggt/eth3d/{scene}/sparse"
    os.makedirs(output, exist_ok=True)

    # reconstruction
    rec = vggt.forward(input, output, max_images=150, use_ba=True)
