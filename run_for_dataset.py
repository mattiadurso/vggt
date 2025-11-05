import os
from vggt.wrapper import VGGTWrapper


paths = {
    "imc_local": {
        "base_path": "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_2D/imc/data/phototourism",
        "images_path": "set_100/images",
        "output_path": "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_3D/results",
    },
    "imc_orochi": {
        "base_path": "/data/mdurso/imc",
        "images_path": "set_100/images",
        "output_path": "/data/mdurso/imc/results",
    },
    "eth3d_local": {
        "base_path": "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_3D/eth3d",
        "images_path": "images_by_k",
        "output_path": "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_3D/results",
    },
    "eth3d_orochi": {
        "base_path": "/data/mdurso/eth3d",
        "images_path": "images_by_k",
        "output_path": "/data/mdurso/eth3d/results",
    },
    "synth_local": {
        "base_path": "/home/mattia/HDD_Fast/Synth_poses",
        "images_path": "images",
        "output_path": "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_3D/results",
    },
}

use_ba = False
_ba = "_ba" if use_ba else ""

dataset = "imc_local"
base_path = paths[dataset]["base_path"]
images_path = paths[dataset]["images_path"]
output_path = paths[dataset]["output_path"]
scenes = sorted(os.listdir(f"{base_path}"))

if not use_ba:
    vggt = VGGTWrapper(cuda_id=0)

for scene in scenes:
    input = f"{base_path}/{scene}/{images_path}"
    output = f"{output_path}/vggt{_ba}/{dataset.split('_')[0]}/{scene}/sparse"
    if use_ba:
        vggt = VGGTWrapper(cuda_id=0, oom_safe=use_ba)

    _ = vggt.forward(input, output, max_images=-1, use_ba=use_ba)


# ##### MapAnything ######
# dataset = "imc"
# base_path = paths[dataset]["base_path"]
# images_path = paths[dataset]["images_path"]
# output_path = paths[dataset]["output_path"]
# scenes = sorted(os.listdir(f"{base_path}"))

# for scene in scenes:
#     os.system(
#         f"python scripts/demo_colmap.py \
#             --scene_dir {base_path}/{scene} \
#             --images_dir {base_path}/{scene}/{images_path} \
#             --out_dir {output_path}/map_anything/{dataset}/{scene} \
#     "
#     )
