import os

import monodepth2.networks
import numpy as np
import PIL.Image as pil
import torch
from layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist
from torchvision import transforms

# Create depth frames for each frame from a video.


def create_depth_frames(video_byte_array, model_name="mono+stereo_640x192"):
    """Function to predict for a video."""
    assert (
        model_name is not None
    ), "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = monodepth2.networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = monodepth2.networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    num_frames, original_height, original_width, _ = video_byte_array.shape
    disp_map = np.zeros((num_frames, original_height, original_width))

    # Go through each frame and predict the depth map
    for i in range(num_frames):
        input_image = pil.fromarray(np.uint8(video_byte_array[i])).convert("RGB")
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False
        )

        # Saving numpy file
        # Save the resized disp instead
        scaled_disp, _ = disp_to_depth(disp_resized.squeeze(), 0.1, 100)
        disp_map[i] = scaled_disp.cpu().detach().numpy()
    return disp_map
