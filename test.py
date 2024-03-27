import torch
import sys

import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=1, type=int,  help='model_type')
    parser.add_argument('--input', default=256, type=int,  help='input size')

    args = parser.parse_args()

    #os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    if args.model == 1:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    if args.model == 2:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    model.eval()

    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(args.input),
        transforms.CenterCrop(args.input),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

    ## Enable RT-Swap component.
    ## Due to the IPC connection issue, rt-swap should be called before pushing any data to GPU.
    ## RT-Swap will automatically sends the model to GPU.
    model.rt_swap(priority=1, period= 500, model_type=args.model)
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        for i in range(10):
            print("Inference %d" % i, file=sys.stderr)
            output = model(input_batch)
           
    output = output.to('cpu')
