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
    #vis = visdom.Visdom(port='1235')

    with torch.no_grad():
        g_ema.eval()

        results={}
        
        for i in tqdm(range(args.nb_z)):
           sample_z = torch.randn(1, args.latent, device=device)

           sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

           #vis.image(sample[0])
           range_2=(-1, 1)
           tensor_z = sample[0]#.clone()  # avoid modifying tensor in-place
           norm_range(tensor_z, range_2)
           #vis.image(tensor_z)

           one_z = torch.ones(sample_z.size()).to(device)


           minus = torch.zeros(args.nb_eps)

           plus = torch.zeros(args.nb_eps)

           #sample_z_minus = sample_z.clone().to(device) -args.eps * one_z
           sample_z_minus = sample_z.to(device) -args.eps * one_z

           #sample_z_plus = sample_z.clone().to(device) + args.eps * one_z
           sample_z_plus = sample_z.to(device) + args.eps * one_z

           
           for k in range(args.nb_eps):
               #print(sample_z_minus[0,423])
               sample_minus, _ = g_ema([sample_z_minus], truncation=args.truncation, truncation_latent=mean_latent)

               sample_plus, _ = g_ema([sample_z_plus], truncation=args.truncation, truncation_latent=mean_latent)

               #vis.image(sample_minus[0])
               range_2=(-1, 1)
               tensor_minus = sample_minus[0]  # avoid modifying tensor in-place
               norm_range(tensor_minus, range_2)
               #vis.image(tensor_minus)

               #vis.image(sample_plus[0])
               range_2=(-1, 1)
               tensor_plus = sample_plus[0]  # avoid modifying tensor in-place
               norm_range(tensor_plus, range_2)
               #vis.image(tensor_plus)

               criterion = torch.nn.MSELoss()

               minus[k]=criterion(tensor_z,tensor_minus)
               #print(tensor_z.shape,tensor_minus.shape)
               
               plus[k]=criterion(tensor_z,tensor_plus)

               sample_z_minus = sample_z_minus -args.eps * one_z
               
               sample_z_plus = sample_z_minus + args.eps * one_z

           current_results = {'minus':minus,'plus':plus}
           results[str(i)] = current_results


           
        torch.save(results,'/data1/pnsuau/cartier/cropped_uncentered_512_zoom/test_smoothness/ring/results_2.pt')
           
    

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--nb_z', type=int, default=20)
    parser.add_argument('--nb_eps', type=int, default=20)
    parser.add_argument('--eps', type=float, default=0.1)
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
