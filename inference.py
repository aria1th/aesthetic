import torch
from transformers import CLIPModel, CLIPProcessor
from aesthetic import Classifier
import argparse
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import json
import os
import glob
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

aesthetic_path = 'aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
DEBUG = False # disables threading

def convert_image(image_path):
    """
    Converts an image to RGB format using OpenCV.
    If OpenCV is not installed, uses PIL.
    """
    if cv2 is not None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        return convert_image(image_path)

def image_embeddings_file(rgb_image, model, processor, device=0):
    inputs = processor(images=rgb_image, return_tensors='pt')['pixel_values']
    inputs = inputs.to(f'cuda:{device}')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

def inference(image_path, device=0):
    clipmodel = CLIPModel.from_pretrained(clip_name).to(f'cuda:{device}').eval()
    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    image = convert_image(image_path)
    image_embeds = image_embeddings_file(image, clipmodel, clipprocessor, device)
    aes_model = Classifier(512, 256, 1)
    aes_model.load_state_dict(torch.load(aesthetic_path))
    aes_model.to(f'cuda:{device}').eval()
    prediction = aes_model(torch.from_numpy(image_embeds).float().to(f'cuda:{device}'))
    #print(f'Prediction: {prediction.item()}')
    return prediction.item()

def inference_thread_job(images, clip_model, clip_processor, aes_model, device, results_container, image_paths, desc="", pbar=None):
    pbar.set_description(desc)
    #pbar = tqdm(total=len(images), desc=f'Inference on device {device} minibatch {minibatch_desc}')
    for image, image_path in zip(images, image_paths):
        image_embeds = image_embeddings_file(image, clip_model, clip_processor, device)
        prediction = aes_model(torch.from_numpy(image_embeds).float().to(f'cuda:{device}'))
        results_container.append((image_path, prediction.item()))
        pbar.update(1)

def inference_batch(image_paths, device=0, minibatch_size=32, result_container:List=None, pbar=None) -> Tuple[str, float]:
    clipmodel = CLIPModel.from_pretrained(clip_name).to(f'cuda:{device}').eval()
    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    aes_model = Classifier(512, 256, 1)
    aes_model.load_state_dict(torch.load(aesthetic_path))
    aes_model.to(f'cuda:{device}').eval()
    results_container_thread = result_container
    print(f"Handling {len(image_paths)} images on device {device}")
    image_batches = [image_paths[i:i + minibatch_size] for i in range(0, len(image_paths), minibatch_size)]
    print(f"Number of batches: {len(image_batches)}")
    # assert total number of images is equal to the number of images in the image_paths list
    assert sum([len(i) for i in image_batches]) == len(image_paths), "Number of images in batches does not equal number of images in image_paths list"
    with ThreadPoolExecutor(max_workers=1) as executor:
        # with single-thread ordered execution, make jobs be e
        _i = 0
        for batches in image_batches:
            images = [convert_image(i) for i in batches]
            ##pbar = tqdm(total=len(images), desc=f'Inference on device {device} minibatch {minibatch_desc}')
            
            desc = f'Inference on device {device} minibatch {_i}/{len(image_batches)}'
            if not DEBUG:
                executor.submit(inference_thread_job, images, clipmodel, clipprocessor, aes_model, device, results_container_thread, batches, desc, pbar)
            else:
                inference_thread_job(images, clipmodel, clipprocessor, aes_model, device, results_container_thread, batches, desc, pbar)
            _i += 1

    # wait for executor to finish
    executor.shutdown(wait=True)
    # save to json file
    #with open(filepath, 'w', encoding='utf-8') as fp:
    #    json.dump(dict(results_container_thread), fp)

def inference_multi_gpu(image_paths, devices:List[int],minibatch_size=32, result_path='aesthetic.json'):
    """
    image_paths: list of image paths or glob pattern, or directory, file containing image paths
    devices: list of devices to run inference on
    minibatch_size: minibatch size per device
    result_path: path to save results
    """
    devices_len = len(devices)
    # handle image_paths
    if isinstance(image_paths, list):
    # split image paths into equal parts
        pass
    elif isinstance(image_paths, str):
        # check if existing directory
        if os.path.isdir(image_paths):
            image_paths = glob.glob(f'{image_paths}/*')
        # check if existing file
        elif os.path.isfile(image_paths):
            with open(image_paths) as f:
                image_paths = f.readlines()
        # check if glob pattern
        elif '*' in image_paths:
            image_paths = glob.glob(image_paths)
        else:
            raise ValueError(f'Invalid image_paths: {image_paths}')
    # remove .txt files from image_paths
    image_paths = [i for i in image_paths if not i.endswith('.txt')]
    per_device = len(image_paths) // devices_len
    image_batches = [image_paths[i:i + per_device] for i in range(0, len(image_paths), per_device)]
    print(f'Number of images: {len(image_paths)}, {[len(i) for i in image_batches]}')
    # assert total number of images is equal to the number of images in the image_paths list
    assert sum([len(i) for i in image_batches]) == len(image_paths), "Number of images in batches does not equal number of images in image_paths list"
    result_container = []
    futures = []
    for i, device in enumerate(devices):
        pbar = tqdm(total=len(image_batches[i]), desc=f'Inference on device {device}')
        if not DEBUG:
            futures.append(ThreadPoolExecutor(max_workers=1).submit(inference_batch, image_batches[i], device, minibatch_size, result_container, pbar))
        else:
            inference_batch(image_batches[i], device, minibatch_size, result_container, pbar)
    # wait for all futures to finish
    for future in futures:
        future.result()
    # save to json file
    with open(result_path, 'w', encoding='utf-8') as fp:
        json.dump(dict(result_container), fp)
    print(f'Saved results to {result_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, required=True, help='Path to images.')
    parser.add_argument('--devices', type=str, required=True, help='List of devices to run inference on. To use multiple instance in single device, use comma separated values. (e.g. 0,0,1,2,3))')
    # default 32
    parser.add_argument('--minibatch-size', type=int, help='Minibatch size per device.', default=32)
    parser.add_argument('--result-path', type=str, help='Path to save results.', default='aesthetic.json')
    args = parser.parse_args()
    devices = [int(i) for i in args.devices.split(',')]
    start_time = time.time()
    inference_multi_gpu(args.images_path, devices, args.minibatch_size, args.result_path)
    end_time = time.time()
    print(f'Time taken: {end_time - start_time:2f} seconds')
    # python inference.py --images-path /data0/subset-1000 --devices 0,0,1,2,3 --minibatch-size 32 --result-path aesthetic.json
