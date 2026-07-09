import torch
import torch.nn as nn
from utils.general import printable_params, rprint_architecture

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors

@printable_params
@rprint_architecture
class PI0Wrapper(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super(PI0Wrapper, self).__init__()
        self.config = config
        model_id = config.get("model_id", "lerobot/pi0_libero_finetuned")
        print(f"Loading PI0 model from {model_id}")
        self.model = PI0Policy.from_pretrained(model_id)

        self.tracing_layers = [f"transformer.h.{i}" for i in range(self.model.config.n_layer)]

        # TODO: implement processing
        preprocess, postprocess = make_pre_post_processors(
            self.model.config,
            model_id,
            # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )


        # Robot and environment configuration
        # Camera keys must match the name and resolutions of the ones used for training!
        # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
        camera_config = {
            "base_0_rgb": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
            "right_wrist_0_rgb": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        }

        # more under: https://github.com/huggingface/lerobot/blob/main/examples/tutorial/pi0/using_pi0_example.py
