
##########################################################################


##########################################################################
import torch
import argparse
import os
import csv
import random
import yaml
import wandb
import monai
import time
import numpy as np
from yaml.loader import SafeLoader
from tabulate import tabulate

import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
from tqdm import tqdm

from src.unet import UNet3D
from src.dataloader import XCT_dataset
from src.utils import load_model, get_views, update_config, save_config, save_ct_slices, to_unNorm, toUnMinMax, save_volume, Structural_Similarity, Peak_Signal_to_Noise_Rate_3D, MAE, MSE

##########################################################################

parser = argparse.ArgumentParser(
    description="OT Reg. for 2D-3D Translation")
parser.add_argument('--config', type=str, default=None,
                   help='Path to config file')
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--save_volume", action="store_true", help="Saves CT volumes")
parser.add_argument('--model_path', type=str,
                    help='Path to model weights')
parser.add_argument('--experiment', type=str,
                    help='name of experiment')
parser.add_argument("--rough_align", action="store_true", help="Apply rought alignment to 2d views")
parser.add_argument("--sample_noise", action="store_true", help="Concatenates noise instead of duplicating views")


args, cfg_args = parser.parse_known_args()

with open(args.config) as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)
    config = update_config(config, cfg_args)

t_val = []
for arg in vars(args):
    t_val.append([arg, getattr(args, arg)])
print('\n')
print(tabulate(t_val, 
               ['input', 'value'],
               tablefmt="psql"))

##########################################################################

dataset = config['data']['dataset']

results_dir = os.path.join(f"{config['training']['outf']}", f"{config['experiment']}")
test_dir = os.path.join(results_dir, dataset, "test")
inference_dir = os.path.join(results_dir, dataset, "inference")
test_file = os.path.join(test_dir, 'metrics_results.csv')

try:
    os.makedirs(results_dir, exist_ok=True)
except OSError:
    pass
try:
    os.makedirs(test_dir, exist_ok=True)
except OSError:
    pass
try:
    os.makedirs(inference_dir, exist_ok=True)
except OSError:
    pass

save_config(os.path.join(results_dir, dataset, 'config.yaml'), config)
    
if config['training']['manualSeed'] is None:
    manualSeed = random.randint(1, 10000)
else: manualSeed = config['training']['manualSeed']
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset ops
num_xrays = config['data']['num_proj']
datadir = config['data']['data_dir']
image_size = config['data']['image_size']
batch_size = config['data']['batch_size']

if dataset in ['chest', 'knee', 'midrc', 'anti-pd-1', 'covid-19-ar', 'lctsc', 'nsclc', 'spie-aapm']:
    valid_dataset = XCT_dataset(data_dir=datadir, train=False, dataset=dataset,
                                xray_scale=image_size,
                                projections=num_xrays,
                                f16=config['training']['amp'],
                                separate_xrays=False,
                                use_synthetic=config['data']['synthetic'])
else:
    raise Exception("Unknown option set for data:dataset")
valid_loader = DataLoader(valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)

device = torch.device("cuda:0" if args.cuda else "cpu")

if config['model'] == 'unet':
    from src.unet import UNet3D
    AE = UNet3D(config['data']['num_ch'], config['data']['num_ch'],
                groupnorm=config['training']['groupnorm'],
                attention=config['training']['attention']).to(device)
elif config['model'] == 'swin-unetr':
    from monai.networks.nets import SwinUNETR
    AE = SwinUNETR(
        img_size=(image_size, image_size, image_size),
        in_channels=config['data']['num_ch'],
        out_channels=config['data']['num_ch'],
        feature_size=config['transformer']['feature_size'],
        use_checkpoint=config['transformer']['grad_ckpt']).to(device)
else:
    raise Exception(f"Unknown model option {config['model']}")
print(f"Using model {config['model']}")

# load pretrained models if given path to AE
model_path = args.model_path
AE = load_model(AE, model_path)
model_name = os.path.basename(model_path)

AE.eval()

ssim_value = 0.0
psnr_value = 0.0
mse_value = 0.0
mae_value = 0.0
fid_value = 0.0
total_time = 0.0

noises_train = torch.randn(len(valid_dataset), image_size, image_size) if args.sample_noise and num_xrays == 1 else None

pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
for it, data in pbar:
    ct = data["ct"].to(device)
    x = data["xrays"].to(device)
    views = get_views(x, data["view_list"], data["idx"], image_size//len(data["view_list"]), noises_train,
                          False, args.rough_align).to(device)

    t1 = time.time()
    ct_hat = AE(views)
    t0 = time.time()
    total_time += t0 - t1

    torch.save(ct_hat[0], os.path.join(inference_dir, f"{data['dir_name'][0]}.pt"))

    imgs = torch.stack([ct, ct_hat], 0).unsqueeze(0)

    # tensor -> numpy array -> unnormalisation
    real_ct, recon_ct = to_unNorm(ct[0], ct_hat[0])
    
    ssim = Structural_Similarity(real_ct, recon_ct, size_average=False, PIXEL_MAX=1.0)
    mse = MSE(real_ct, recon_ct, size_average=False)
    mae = MAE(real_ct, recon_ct, size_average=False)
    
    # to Hounsfield scale
    recon_ct_np = toUnMinMax(recon_ct).astype(np.int32) - 1024
    ct_gt_np = toUnMinMax(real_ct).astype(np.int32) - 1024
    
    psnr = Peak_Signal_to_Noise_Rate_3D(ct_gt_np, recon_ct_np, size_average=False, PIXEL_MAX=4095)
    
    # Saving to .raw for visualisation
    save_volume(real_ct, os.path.join(f"{results_dir}/chest/CT/test", f"{dataset}_real_ct_{it:04}"))
    save_volume(recon_ct, os.path.join(f"{results_dir}/chest/CT/test",
                                       f"{dataset}_recon_ct_{it:04}"))

    save_ct_slices([ct, ct_hat],
                   f"{results_dir}/chest/CT/test/{dataset}_eval_{it}")

    save_ct_slices([ct, ct_hat],
                   f"{results_dir}/chest/CT/test/{dataset}_eval_max_proj_{it}",
                   max_projection=True)
    
    psnr_value += psnr[0]

    ssim_value += ssim[-1][0]

    mae_value += mae[0]

    mse_value += mse[0]

psnr_value /= len(valid_loader)
ssim_value /= len(valid_loader)
mae_value /= len(valid_loader)
mse_value /= len(valid_loader)
total_time /= len(valid_loader)

print(f"psnr: {psnr_value:.4f}")
print(f"ssim: {ssim_value:.4f}")
print(f"mae: {mae_value:.4f}")
print(f"mse: {mse_value:.4f}")
print(f"Inference time: {total_time:.4f}")

with open(test_file, 'w', encoding='UTF8') as csv_stat:
    csv_writer = csv.writer(csv_stat)
    csv_writer.writerow(['WEIGHTS', 'PSNR', 'SSIM', 'MSE', 'MAE', 'Inf. time'])
    csv_writer.writerow([model_name, psnr_value, ssim_value, mse_value, mae_value, total_time])
print("Results written in test/metric_results.csv")
