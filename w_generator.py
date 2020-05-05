import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator

import glob

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--saveroot', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--w_plus', action='store_true')
    #parser.add_argument('files', metavar='FILES', nargs='+')

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    #imgs = []

    imgs_name = []
    
    all_files = glob.glob(args.dataroot + '/*')
    
    for imgfile in all_files:
        if is_image_file(imgfile):
            #img = transform(Image.open(imgfile).convert('RGB'))
            imgs_name.append(imgfile)
        

    nb_img_done = 0

    batch=2

    #img_gen = torch.zeros([len(imgs_name),3,512,512])

    #print(img_gen.shape)
    
    while nb_img_done < len(imgs_name):

        imgs = []

        for k in range(batch):
            img = transform(Image.open(imgs_name[nb_img_done+k]).convert('RGB'))
            #print(imgs)
            imgs.append(img)        
        
        imgs = torch.stack(imgs, 0).to(device)

        g_ema = Generator(args.size, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)
        
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=device.startswith('cuda'),#gpu_ids=[0,1,2,3]
        )

        noises = g_ema.make_noise()

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(2, 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        #print(latent_in.shape)

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]['lr'] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            current_img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = current_img_gen.shape

            if height > 256:
                factor = height // 256

                current_img_gen = current_img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                current_img_gen = current_img_gen.mean([3, 5])

            p_loss = percept(current_img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(current_img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
                    f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'
                )
            )

        #result_file = {}

        current_img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

        #print(current_img_gen.size())
        
        #img_gen[nb_img_done:nb_img_done+batch]=current_img_gen
        
        #filename = os.path.splitext(os.path.basename(args.files[0]))[0] + '.pt'
        #filename = 'z'+ '_'+ str(nb_img_done)  + '.pt'

        #print('filename',filename)

        nb_img_done+=batch
        print('nb_img_done',nb_img_done)

        #current_img_ar = make_image(current_img_gen)

        for i, input_name in enumerate(imgs_name[nb_img_done:nb_img_done+batch]):
            temp = input_name.split('/')[-1]
            #result_file[temp[0:-4]] = latent_in[i]
            #img_name = os.path.splitext(os.path.basename(input_name))[0] + '-project.png'
            #pil_img = Image.fromarray(current_img_ar[i])
            #pil_img.save(args.saveroot + temp)
            #print(latent_in[i].shape)
            torch.save(latent_in[i],args.saveroot + temp[0:-4] +'.pt')










        
    #img_ar = make_image(img_gen)

    #for i, input_name in enumerate(imgs_name):
        #result_file[input_name] = {'img': img_gen[i], 'latent': latent_in[i]}
        #img_name = os.path.splitext(os.path.basename(input_name))[0] + '-project.png'
        #pil_img = Image.fromarray(img_ar[i])
        #pil_img.save(img_name)
    
    
    #torch.save(result_file, filename)


    #temp =torch.load(filename)
    #print(temp)
