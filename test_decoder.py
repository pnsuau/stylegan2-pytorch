import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import visdom

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
def norm_range(t, range_2):
    if range_2 is not None:
        norm_ip(t, range_2[0], range_2[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
        
def generate(args, g_ema, device, mean_latent):
    vis = visdom.Visdom(port='1235')

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)
           #print(sample_z)

           sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
           vis.image(sample[0])
           #print(sample)
           #print(sample.mean())
           #print(sample.min())
           #print(sample.max())
           range_2=(-1, 1)
           tensor = sample[0].clone()  # avoid modifying tensor in-place
           scale_each = False
           if scale_each is True:
               for t in tensor:  # loop over mini-batch dimension
                   norm_range(t, range_2)
           else:
               norm_range(tensor, range_2)
    vis.image(tensor)
    #print(tensor)
    #print(tensor.mean())
    #print(tensor.min())
    #print(tensor.max())


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
