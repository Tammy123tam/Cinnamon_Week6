# !pip install -U ultralytics
import torch
import cv2
from pathlib import Path
import os
import numpy as np
import torch.nn as nn
import argparse


def load_model(device):
    if device not in ['cuda', "cpu"]:
        raise Exception("Invalid device")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)
    list_modules = []
    for name, param in model.named_parameters():
        list_modules.append(name)
    for name in list_modules:
        arr = name.split(".")[:-1]
        layer = model
        for i in arr:
            layer = getattr(layer, i)
            layer.register_forward_hook(getActivation(layer))
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    return model


def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        outputs[name] = output
    return hook


def load_image(image):
    if image is None or not Path.exists(Path(image)):
        raise FileNotFoundError("Can't find image path")
    img = cv2.imread(image)[:, :, ::-1]
    img = cv2.resize(img, (1280, 1280))
    return img


def normalize(image):
    imin = np.min(image)
    imax = np.max(image)

    # Check if imin and imax are the same or very close to each other
    if np.isclose(imin, imax, atol=1e-6):
        return image.astype(np.uint8)

    a = 255.0 / (imax - imin)
    b = 255.0 - a * imax

    new_img = np.clip((a * image + b), 0, 255).astype(np.uint8)
    return new_img


def forward(img, model, device):
    with torch.cuda.amp.autocast():
        if device == "cuda":
            outputs = model.forward(torch.reshape(
                torch.tensor(img), (1, 3, 1280, 1280)).half().cuda())
        elif device == "cpu":
            outputs = model.forward(torch.reshape(
                torch.tensor(img), (1, 3, 1280, 1280)).half())
        else:
            raise Exception("Invalid device")
        return outputs


def extract(model, output_dir, limit: int = 20):
    path = Path(".") / output_dir
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    range_i = len(list(outputs.keys()))
    list_modules = []
    for name, param in model.named_parameters():
        list_modules.append(name)
    list_modules = list_modules[::2]
    layers = {}
    for name in list_modules:
        layer = name.split('.')[3:-1]
        if layers.get(layer[0]) is None:
            layers[layer[0]] = [layer[1]]
        else:
            layers[layer[0]].append(layer[1])
    i = 0
    for k in layers.keys():
        for x, v in enumerate(layers[k]):
            if i == range_i:
                break
            # Create folder for each layer
            path = Path(".") / output_dir / f"{k}_{v}_{x}"
            if not path.is_dir():
                path.mkdir(parents=True, exist_ok=True)

            # Log the process
            if torch.is_tensor(outputs[list(outputs.keys())[i]]):
                num_channels = min(
                    list(outputs[list(outputs.keys())[i]].shape)[1], limit)
                for j in range(num_channels):
                    im = normalize(np.float32(
                        outputs[list(outputs.keys())[i]][0, j].cpu().detach().numpy()))
                    cv2.imwrite(Path(path / f"{j}.png").as_posix(), im)
            i += 1

    print("Extraction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image extraction file")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="cat.png",
        help="input image",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="output folder",
    )
    parser.add_argument('-gpu', action='store_true',
                        default=False, help="enable gpu")
    args = parser.parse_args()
    if not args.gpu:
        raise EnvironmentError(
            "GPU must be enabled, since some functions in Yolo is not work with Cpu")
    outputs = {}
    args.gpu = 'cuda'
    model = load_model(device=args.gpu)
    image = load_image(image=args. input_dir, )
    _ = forward(img=image, model=model, device=args.gpu)
    if args.output_dir is None:
        args.output_dir = args.input_dir.split(".")[-2]
    extract(model, output_dir=args.output_dir, limit=20)
