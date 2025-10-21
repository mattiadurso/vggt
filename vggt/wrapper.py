import os
import gc
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple
import copy
import pycolmap
import time
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)


class VGGTWrapper:
    """Wrapper class for VGGT model to perform 3D reconstruction from images."""

    def __init__(
        self,
        cuda_id: int = 0,
        seed: int = 42,
        model_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    ):
        """
        Initialize the VGGT wrapper.

        Args:
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
            seed: Random seed for reproducibility.
            model_url: URL to download the VGGT model weights.
        """
        self.seed = seed
        self._set_seed(seed)

        # Setup device and dtype
        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        )

        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Configure CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load model
        self.model = self._load_model(model_url)

        # Fixed resolutions
        self.vggt_fixed_resolution = 518
        self.img_load_resolution = 1024

        print(f"VGGTWrapper initialized on {self.device} with dtype {self.dtype}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _load_model(self, model_url: str) -> VGGT:
        """Load the VGGT model."""
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
        model.eval()
        model = model.to(self.device)
        print("VGGT model loaded successfully")
        return model

    def _find_images(self, images_path: str) -> List[str]:
        """
        Find all images in the given path, including subdirectories.

        Args:
            images_path: Path to directory containing images.

        Returns:
            List of image file paths.
        """
        valid_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        image_paths = []

        for ext in valid_extensions:
            # Search in root, one level deep, and two levels deep
            image_paths.extend(glob.glob(os.path.join(images_path, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(images_path, "*", f"*.{ext}")))
            image_paths.extend(
                glob.glob(os.path.join(images_path, "*", "*", f"*.{ext}"))
            )

        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {images_path}")

        print(f"Found {len(image_paths)} images in {images_path}")
        return image_paths

    def _sample_images(
        self, image_paths: List[str], max_images: Optional[int] = None
    ) -> List[str]:
        """
        Randomly sample images if needed.

        Args:
            image_paths: List of all image paths.
            max_images: Maximum number of images to use. None means use all.

        Returns:
            Sampled list of image paths.
        """
        if max_images is not None and len(image_paths) > max_images:
            sampled_paths = random.sample(image_paths, max_images)
            sampled_paths = sorted(sampled_paths)  # Keep sorted order
            print(f"Randomly sampled {max_images} images from {len(image_paths)}")
            return sampled_paths
        return image_paths

    def _run_vggt(
        self, images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run VGGT model to estimate cameras and depth.

        Args:
            images: Batch of images [B, 3, H, W].

        Returns:
            Tuple of (extrinsic, intrinsic, depth_map, depth_conf).
        """
        assert len(images.shape) == 4 and images.shape[1] == 3

        # Resize to VGGT resolution
        images_resized = F.interpolate(
            images,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                images_batch = images_resized[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)

            # Predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_batch.shape[-2:]
            )

            # Predict depth
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images_batch, ps_idx
            )

        # Convert to numpy
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()

        return extrinsic, intrinsic, depth_map, depth_conf

    @torch.no_grad()
    def _reconstruct_with_ba(
        self,
        images: torch.Tensor,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        points_3d: np.ndarray,
        max_query_pts: int,
        query_frame_num: int,
        vis_thresh: float,
        max_reproj_error: float,
        shared_camera: bool,
        camera_type: str,
        fine_tracking: bool,
    ):
        """Reconstruct with bundle adjustment."""
        image_size = np.array(images.shape[-2:])
        scale = self.img_load_resolution / self.vggt_fixed_resolution

        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = (
                predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=max_query_pts,
                    query_frame_num=query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=fine_tracking,
                )
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = self.img_load_resolution

        return reconstruction, reconstruction_resolution

    def _reconstruct_without_ba(
        self,
        images: torch.Tensor,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        depth_conf: np.ndarray,
        points_3d: np.ndarray,
        conf_thres_value: float,
        max_points_for_colmap: int,
    ):
        """Reconstruct without bundle adjustment."""
        image_size = np.array([self.vggt_fixed_resolution, self.vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        # Get RGB values
        points_rgb = F.interpolate(
            images,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # Create coordinate grid
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        # Filter by confidence
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )

        return reconstruction, self.vggt_fixed_resolution

    def _rescale_reconstruction(
        self,
        reconstruction,
        base_image_paths: List[str],
        original_coords: np.ndarray,
        img_size: int,
        shift_point2d: bool,
        shared_camera: bool,
    ):
        """Rescale and rename reconstruction to match original images."""
        rescale_camera = True

        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = base_image_paths[pyimageid - 1]

            if rescale_camera:
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = original_coords[pyimageid - 1, -2:]
                resize_ratio = max(real_image_size) / img_size
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp

                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

            if shift_point2d:
                top_left = original_coords[pyimageid - 1, :2]
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                rescale_camera = False

        return reconstruction

    def triangulate_with_colmap(
        self,
        images_path: str,
        input_sparse_path: str,
        output_sparse_path: str,
        clear_points: bool = True,
        refine_intrinsics: bool = False,
        use_ba: bool = False,
        verbose: bool = False,
    ):
        """
        Triangulate 3D points using COLMAP's feature extraction, matching, and triangulation.
        This is a bit unelegant since we have to call COLMAP multiple times for each image to
        extract features with the correct camera parameters.

        This method:
        1. Loads the reconstruction from input_sparse_path
        2. Extracts SIFT features for each image using the camera parameters from reconstruction
        3. Matches features across all image pairs
        4. Triangulates 3D points from the matches
        5. Optionally runs bundle adjustment

        Args:
            images_path: Path to directory containing images.
            input_sparse_path: Path to input COLMAP sparse reconstruction (with cameras and poses).
            output_sparse_path: Path where to save the triangulated reconstruction.
            clear_points: Whether to clear existing 3D points before triangulation.
            refine_intrinsics: Whether to refine camera intrinsics during BA.
            ba_iterations: Number of bundle adjustment iterations (0 = skip BA).
        """
        verbose = "" if verbose else " > /dev/null 2>&1"
        # add sparse to input and output paths if not present
        if not input_sparse_path.endswith("sparse"):
            input_sparse_path = os.path.join(input_sparse_path, "sparse")
        if not output_sparse_path.endswith("sparse"):
            output_sparse_path = os.path.join(output_sparse_path, "sparse")
        output_sparse_path = (
            output_sparse_path + "_ba" if use_ba else output_sparse_path
        )
        print(f"Triangulating points with COLMAP...")
        print(f"  Images path: {images_path}")
        print(f"  Input sparse: {input_sparse_path}")
        print(f"  Output sparse: {output_sparse_path}")

        # Load reconstruction to get camera parameters
        reconstruction = pycolmap.Reconstruction(input_sparse_path)

        # Create working directory for COLMAP database
        work_dir = Path(output_sparse_path).parent / "colmap_work"
        os.makedirs(work_dir, exist_ok=True)
        database_path = work_dir / "database.db"

        if database_path.exists():
            print("Using existing COLMAP database...")
        else:
            # Create temporary directory to move images one at a time
            temp_path = work_dir / "temp"
            os.makedirs(temp_path, exist_ok=True)

            # Move all images to temp directory first
            print("Moving images to temporary directory...")
            os.system(f"cp -r {images_path}/* {temp_path}")

            # Process each image individually with COLMAP feature extraction
            print("Extracting features for each image...")
            for pyimage in tqdm(
                sorted(reconstruction.images.values(), key=lambda x: x.image_id),
                desc="Extracting features",
            ):
                image_name = pyimage.name
                camera = reconstruction.cameras[pyimage.camera_id]

                # Move image back temporarily
                temp_image_path = f"{temp_path}/{image_name}"
                final_image_path = work_dir / "frames" / image_name

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(final_image_path), exist_ok=True)
                os.system(f"mv {temp_image_path} {final_image_path}")

                # Get camera parameters
                camera_model = camera.model.name
                params = [str(p) for p in camera.params]

                # Run COLMAP feature extraction for this image
                cmd = (
                    f"colmap feature_extractor "
                    f"--database_path {database_path} "
                    f"--image_path {work_dir / 'frames'} "
                    f"--ImageReader.camera_model {camera_model} "
                    f'--ImageReader.camera_params "{",".join(params)}"'
                )
                os.system(cmd + verbose)
                # os.system(cmd + verbose)

            # Match features
            print("Matching features...")
            match_cmd = f"colmap exhaustive_matcher --database_path {database_path}"
            os.system(match_cmd + verbose)
            # os.system(match_cmd + verbose)

        os.makedirs(output_sparse_path, exist_ok=True)
        triangulate_cmd = (
            f"colmap point_triangulator "
            f"--input_path {input_sparse_path} "
            f"--database_path {database_path} "
            f"--image_path {work_dir / 'frames'} "
            f"--output_path {output_sparse_path} "
            f"--clear_points {1 if clear_points else 0} "
            f"--refine_intrinsics {1 if refine_intrinsics else 0} "
        )

        triangulate_cmd += (
            "--Mapper.ba_refine_focal_length 0 "
            "--Mapper.ba_refine_extra_params 0 "
            "--Mapper.ba_refine_principal_point 0 "
            "--Mapper.ba_local_max_num_iterations 1 "
            "--Mapper.ba_global_max_num_iterations 1 "
            "--Mapper.ba_global_function_tolerance 1.0 "
            "--Mapper.ba_local_function_tolerance 1.0 "
        )

        print("Triangulating points...")
        os.system(triangulate_cmd + verbose)

        if use_ba:
            print("Running bundle adjustment...")
            ba_cmd = (
                f"colmap bundle_adjuster "
                f"--input_path {output_sparse_path} "
                f"--output_path {output_sparse_path} "
                f"--BundleAdjustment.max_num_iterations 1024 "
                f"--BundleAdjustment.refine_focal_length 1 "
                f"--BundleAdjustment.refine_principal_point 1 "
                f"--BundleAdjustment.refine_extra_params 1 "
            )
            os.system(ba_cmd)

        # Remove temporary directory
        # os.system(f"rm -rf {work_dir}")

        print(f"Triangulation complete. Results saved to {output_sparse_path}\n")

        # Load and return the triangulated reconstruction in text format
        triangulated_reconstruction = pycolmap.Reconstruction(output_sparse_path)
        triangulated_reconstruction.write_text(output_sparse_path)

        return triangulated_reconstruction

    def forward(
        self,
        images_path: str,
        output_path: str,
        max_images: int = 150,
        use_ba: bool = False,
        one_camera_per_folder: bool = False,
        # BA-specific parameters
        max_reproj_error: float = 8.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.2,
        query_frame_num: int = 8,
        max_query_pts: int = 4096,
        fine_tracking: bool = True,
        # Non-BA parameters
        conf_thres_value: float = 5.0,
        max_points_for_colmap: int = 100_000,
    ):
        """
        Run VGGT reconstruction on images and save results.

        Args:
            images_path: Path to directory containing images.
            output_path: Path where to save COLMAP reconstruction (text format).
            max_images: Maximum number of images to process (randomly sampled if exceeded).
            use_ba: Whether to use bundle adjustment.
            one_camera_per_folder: Whether to assume one camera per folder.
            max_reproj_error: Maximum reprojection error for BA.
            shared_camera: Whether to use shared camera for all images.
            camera_type: Camera type for reconstruction.
            vis_thresh: Visibility threshold for tracks.
            query_frame_num: Number of frames to query for tracking.
            max_query_pts: Maximum number of query points.
            fine_tracking: Use fine tracking (slower but more accurate).
            conf_thres_value: Confidence threshold for depth filtering (without BA).
            max_points_for_colmap: Maximum 3D points for COLMAP (without BA).
        """
        if self.model is None:
            self.__init__()

        if not output_path.endswith("sparse"):
            output_path = os.path.join(output_path, "sparse")
        os.makedirs(output_path, exist_ok=True)
        timings = {}
        t_total_start = time.time()

        # Find and sample images
        t_start = time.time()
        image_paths = self._find_images(images_path)
        image_paths = self._sample_images(image_paths, max_images)
        timings["find_and_sample_images"] = time.time() - t_start

        # Get base paths
        base_image_paths = [os.path.relpath(path, images_path) for path in image_paths]

        # Load and preprocess images
        print(f"Loading {len(image_paths)} images...")
        t_start = time.time()
        images, original_coords = load_and_preprocess_images_square(
            image_paths, self.img_load_resolution
        )
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)
        timings["load_and_preprocess"] = time.time() - t_start

        # Run VGGT
        print("Running VGGT model...")
        t_start = time.time()
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        timings["run_vggt"] = time.time() - t_start

        # Reconstruct with or without BA
        if use_ba:
            # del self.model  # free memory
            # gc.collect()
            # torch.cuda.empty_cache()
            # self.model = None

            print("Running reconstruction with Bundle Adjustment...")
            t_start = time.time()
            reconstruction, recon_resolution = self._reconstruct_with_ba(
                images,
                extrinsic,
                intrinsic,
                depth_map,
                depth_conf,
                points_3d,
                max_query_pts,
                query_frame_num,
                vis_thresh,
                max_reproj_error,
                shared_camera,
                camera_type,
                fine_tracking,
            )
            timings["reconstruction_with_ba"] = time.time() - t_start
        else:
            print("Running reconstruction without Bundle Adjustment...")
            t_start = time.time()
            reconstruction, recon_resolution = self._reconstruct_without_ba(
                images,
                extrinsic,
                intrinsic,
                depth_conf,
                points_3d,
                conf_thres_value,
                max_points_for_colmap,
            )
            timings["reconstruction_without_ba"] = time.time() - t_start

        # Rescale reconstruction to original resolution
        t_start = time.time()
        reconstruction = self._rescale_reconstruction(
            reconstruction,
            base_image_paths,
            original_coords.cpu().numpy(),
            recon_resolution,
            shift_point2d=use_ba,
            shared_camera=shared_camera if use_ba else False,
        )
        timings["rescale_reconstruction"] = time.time() - t_start

        # Save reconstruction
        t_start = time.time()
        os.makedirs(output_path, exist_ok=True)
        reconstruction.write_text(output_path)
        timings["save_reconstruction"] = time.time() - t_start

        timings["total"] = time.time() - t_total_start

        # Print timing summary
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        print(
            f"Find and sample images:      {timings['find_and_sample_images']:>8.2f}s"
        )
        print(f"Load and preprocess images:  {timings['load_and_preprocess']:>8.2f}s")
        print(f"Run VGGT model:              {timings['run_vggt']:>8.2f}s")
        if use_ba:
            print(
                f"Reconstruction (with BA):    {timings['reconstruction_with_ba']:>8.2f}s"
            )
        else:
            print(
                f"Reconstruction (without BA): {timings['reconstruction_without_ba']:>8.2f}s"
            )
        print(
            f"Rescale reconstruction:      {timings['rescale_reconstruction']:>8.2f}s"
        )
        print(f"Save reconstruction:         {timings['save_reconstruction']:>8.2f}s")
        print("-" * 60)
        print(f"TOTAL TIME:                  {timings['total']:>8.2f}s")
        print("=" * 60 + "\n")

        print(f"Reconstruction saved to {output_path}")

        # save timings to a text file
        with open(os.path.join(output_path, "timings.txt"), "w") as f:
            for key, value in timings.items():
                f.write(f"{key}: {value:.4f} s\n")

        del (
            images,
            original_coords,
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            points_3d,
        )
        torch.cuda.empty_cache()

        return reconstruction


if __name__ == "__main__":
    import argparse

    # add dataset scene, and use-ba as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eth3d")
    parser.add_argument("--scene", type=str, default="electro")
    parser.add_argument("--max-images", type=int, default=150)
    parser.add_argument("--use-ba", action="store_true")
    parser.add_argument("--cuda-id", type=int, default=0)
    args = parser.parse_args()

    from vggt.wrapper import VGGTWrapper

    vggt = VGGTWrapper(cuda_id=args.cuda_id)

    # setting paths
    base_path = "/data/mdurso"
    if os.path.exists(base_path):
        if args.dataset == "eth3d":
            input = f"{base_path}/eth3d/{args.scene}/images_by_k"
            output = f"{base_path}/results/vggt/eth3d/{args.scene}/sparse"

        elif args.dataset == "imc":
            input = f"{base_path}/imc/{args.scene}/set_100/images"
            output = f"{base_path}/results/vggt/imc/{args.scene}/sparse"
        os.makedirs(output, exist_ok=True)

    else:
        base_path = "/home/mattia/Desktop/Repos"
        input = f"{base_path}/wrapper_factory/benchmarks_3D/eth3d/electro/images_by_k"
        output = f"{base_path}/vggt/wrapper_output/sparse"

    # reconstruction
    rec = vggt.forward(input, output, max_images=150, use_ba=True)

    # with VGGT default BA, only n images can be optimized.
    # TODO: use pycolmap BA to optimize this

    # triangulation with colmap (just adding more points to the reconstruction)
    # vggt.triangulate_with_colmap(input, output, output)
