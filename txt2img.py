import argparse, os
import numpy as np
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
import custom_pipelines
from time import perf_counter
from optimum.intel.neural_compressor.utils import load_quantized_model


LAUNCHERS = [
    "pytorch",
    "onnx",
    "openvino_onnx",
    "openvino"
]

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="a long road through fog to the sun",
        help="the prompt to render"
    )

    parser.add_argument(
        "-l",
        "--launcher",
        type=str,
        required=False,
        choices=LAUNCHERS,
        default="pytorch",
        help="Optional. Name of using backend for runtime. Available backends = {LAUNCHERS}. Default is 'PyTorch'"
    )

    parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        required=False,
        default="weights",
        help="Optional. Path to source of model weights"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help='Optional. Name of the output file(s) to save.',
        default="outputs/"
    )

    parser.add_argument(
        "--use_quantized",
        action="store_true",
        help='Optional. Whether or not use quantized models.',
    )
    return parser


class BaseLauncher:
    __provider__ = "base"
    def __init__(self, models_dir) -> None:
        self.model_id = os.path.join(models_dir, self.__provider__, "stable-diffusion-v1-4")

class PyTorchLauncher(BaseLauncher):
    __provider__ = "pytorch"

    def __init__(self, models_dir, use_quantized=False) -> None:
        super().__init__(models_dir)
        device = "cpu"
        self.pipe = custom_pipelines.PytorchPipeline(self.model_id, use_quantized=False)
        # loaded_model = load_quantized_model("quantization_output", model=getattr(self.pipe, "unet"))
        # loaded_model.eval()
        # setattr(self.pipe, "unet", loaded_model)
        self.pipe = self.pipe.to(device)

    def __call__(self, prompt: str):
        return self.pipe(prompt).images[0]


class OnnxLauncher(BaseLauncher):
    __provider__ = "onnx"

    def __init__(self, models_dir, use_quantized=False) -> None:
        super().__init__(models_dir)
        self.pipe = OnnxStableDiffusionPipeline.from_pretrained(self.model_id)
        if use_quantized:
            self.pipe.unet = OnnxRuntimeModel.from_pretrained(os.path.join(self.model_id, "unet_quantized"))
        else:
            self.pipe.unet = OnnxRuntimeModel.from_pretrained(os.path.join(self.model_id, "unet"))

    def __call__(self, prompt: str):
        return self.pipe(prompt).images[0]


class OpenVINOToOnnxLauncher(BaseLauncher):
    __provider__ = "onnx"

    def __init__(self, models_dir) -> None:
        super().__init__(models_dir)
        onnx_pipe =  OnnxStableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        provider="CPUExecutionProvider",
                    )
        self.openvino_pipe = custom_pipelines.OpenVINOStableDiffusionPipeline.from_onnx_pipeline(onnx_pipe)

    def __call__(self, prompt: str):
        return self.openvino_pipe(prompt).images[0]

class OpenVINOLauncher(BaseLauncher):
    __provider__ = "openvino"

    def __init__(self, models_dir) -> None:
        super().__init__(models_dir)
        self.openvino_pipe = custom_pipelines.OpenVINOPipeline(self.model_id)

    def __call__(self, prompt: str):
        return self.openvino_pipe(prompt).images[0]

def get_pipeline(options):
    if options.launcher == "pytorch":
        return PyTorchLauncher(options.models_dir, options.use_quantized)
    if options.launcher == "onnx":
        return OnnxLauncher(options.models_dir, options.use_quantized)
    if options.launcher == "openvino_onnx":
        return OpenVINOToOnnxLauncher(options.models_dir)
    if options.launcher == "openvino":
        return OpenVINOLauncher(options.models_dir)

def main():
    opt = build_parser().parse_args()
    launcher = get_pipeline(opt)
    input_prompt = opt.prompt

    os.makedirs(opt.output, exist_ok=True)
    print("Start inference on {}\n".format(opt.launcher))
    start_time = perf_counter()
    result = launcher(input_prompt)
    finish_time = perf_counter()
    output_count = len(os.listdir(opt.output))
    result.save(os.path.join(opt.output, f'result_{output_count:03}.png'))
    print(f"Your samples are ready, all inference take {finish_time - start_time} sec and result waiting for you here: {opt.output} \n \nEnjoy.")


if __name__ == "__main__":
    main()
