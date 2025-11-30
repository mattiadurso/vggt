import os
import glob
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
    "7scenes_local": {
        "base_path": "/home/mattia/Desktop/datasets/7scenes/benchmark_colmap",
        "images_path": "images_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "mipnerf360_local": {
        "base_path": "/home/mattia/Desktop/datasets/mipnerf360",
        "images_path": "images_4_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
    "terrasky3D_local": {
        "base_path": "/home/mattia/Desktop/datasets/terrasky3D",
        "images_path": "images_150",
        "output_path": "/home/mattia/Desktop/Repos/batchsfm/benchmark",
    },
}

use_ba = True
cuda_id = 0

for dataset in ["terrasky3D_local"]:
    _ba = "_ba" if use_ba else ""
    base_path = paths[dataset]["base_path"]
    images_path = paths[dataset]["images_path"]
    output_path = (
        "/home/mattia/Desktop/Repos/batchsfm/benchmark"  # paths[dataset]["output_path"]
    )
    scenes = sorted(os.listdir(f"{base_path}"))

    if not use_ba:
        vggt = VGGTWrapper(cuda_id=cuda_id)

    for scene in scenes:
        input_path = f"{base_path}/{scene}/{images_path}"
        print(f"Processing {dataset} - {scene}...")

        num_images = len(
            glob.glob(f"{input_path}/*.*")
            + glob.glob(f"{input_path}/*/*.*", recursive=True)
        )
        output = f"{output_path}/vggt{_ba}/{dataset.split('_')[0]}/{scene}/sparse"

        if use_ba:
            vggt = VGGTWrapper(cuda_id=cuda_id, oom_safe=use_ba)

        _ = vggt.forward(
            input_path,
            output,
            max_images=-1,
            use_ba=use_ba,
            query_frame_num=16,  # dont change this anyonmore
            shared_camera=False,
        )
