""" Example for stable-diffusion to generate a picture from a text ."""

import argparse
import logging
import math
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from neural_compressor import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer
from optimum.intel.neural_compressor.utils import load_quantized_model
from pytorch_fid import fid_score
from datasets import load_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = ""


logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Example of a post-training quantization script.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="The input of the model, like: 'a photo of an astronaut riding a horse on mars'.",
    )
    parser.add_argument(
        "--calibration_text",
        type=str,
        default="Womens Princess Little Deer Native American Costume",
        help="The calibration data of the model, like: 'Womens Princess Little Deer Native American Costume'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_results",
        help="The path to save model and quantization configures.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt, defaults to 1",
    )
    parser.add_argument(
        "--apply_quantization",
        action="store_true",
        help="Whether or not to apply quantization.",
    )
    parser.add_argument(
        "--quantization_approach",
        type=str,
        default="static",
        help="Quantization approach. Supported approach are static, dynamic and aware_training.",
    )
    parser.add_argument(
        "--tolerance_criterion",
        type=float,
        default=0.01,
        help="Performance tolerance when optimizing the model.",
    )
    parser.add_argument(
        "--verify_loading",
        action="store_true",
        help="Whether or not to verify the loading of the quantized model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether or not to benchmark.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Whether or not to benchmark with quantized model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="random seed",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed: local_rank")
    parser.add_argument(
        "--base_images", type=str, default="base_images", help="Path to training images for FID input."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def benchmark(pipeline, generator):
    warmup = 2
    total = 5
    total_time = 0
    with torch.no_grad():
        for i in range(total):
            prompt = "a photo of an astronaut riding a horse on mars"
            start2 = time.time()
            images = pipeline(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
            end2 = time.time()
            images[0].save(os.path.join("tmp_int8_images", "fp8.png"))
            if i >= warmup:
                total_time += end2 - start2
            print("Total inference latency: ", str(end2 - start2) + "s")
    print("Average latency: ", (total_time) / (total - warmup), "s")


def main():
    # Passing the --help flag to this script.

    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Parameters {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load pretrained model and generate a pipeline
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_name_or_path)
    generator = torch.Generator("cpu").manual_seed(args.seed)

    if args.benchmark:
        if not args.int8:
            print("====fp32 inference====")
            benchmark(pipeline, generator)

    name = "unet"

    def calibration_func(model):
        calib_num = 5
        setattr(pipeline, name, model)
        with torch.no_grad():
            for i in range(calib_num):
                pipeline(
                    args.calibration_text,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    generator=generator,
                    num_images_per_prompt=args.num_images_per_prompt,
                )

    def eval_func(model):
        setattr(pipeline, name, model)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        with torch.no_grad():
            new_images = pipeline(
                args.input_text,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
            ).images
            if os.path.isfile(os.path.join(tmp_int8_images, "int8.png")):
                os.remove(os.path.join(tmp_int8_images, "int8.png"))
            grid = image_grid(new_images, rows=_rows, cols=args.num_images_per_prompt // _rows)
            grid.save(os.path.join(tmp_int8_images, "int8.png"))
            fid = fid_score.calculate_fid_given_paths((args.base_images, tmp_int8_images), 1, "cpu", 2048, 8)
            return fid

    # if not args.apply_quantization:q
    #     raise ValueError("No optimization activated.")

    supported_approach = {"static", "dynamic"}
    if args.quantization_approach not in supported_approach:
        raise ValueError(
            f"Unknown quantization approach. Supported approach are {supported_approach}."
            f"{args.quantization_approach} was given."
        )

    if args.apply_quantization:
        quantization_config = PostTrainingQuantConfig(approach=args.quantization_approach, calibration_sampling_size=10)
        quantizer = INCQuantizer.from_pretrained(pipeline.unet, calibration_fn=calibration_func)

        dataset = None
        if args.quantization_approach == "static":
            dataset = load_dataset("laion/laion400m")
        print(quantization_config.max_trials)

        quantizer.quantize(
            quantization_config=quantization_config,
            save_directory=args.output_dir,
            calibration_dataset=dataset,
            remove_unused_columns=False,
        )

    if args.apply_quantization and args.verify_loading:
        loaded_model = load_quantized_model(args.output_dir, model=getattr(pipeline, "unet"))
        result_optimized_model = eval_func(quantizer._quantized_model)
        result_loaded_model = eval_func(loaded_model)
        if result_loaded_model != result_optimized_model:
            logger.error("The quantized model was not successfully loaded.")
        else:
            logger.info(f"The quantized model was successfully loaded.")

    if args.benchmark and args.int8:
        print("====int8 inference====")
        loaded_model = load_quantized_model(args.output_dir, model=getattr(pipeline, "unet"))
        loaded_model.eval()
        setattr(pipeline, "unet", loaded_model)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        benchmark(pipeline, generator)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
