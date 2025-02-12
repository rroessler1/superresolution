import argparse
import os
import torch
import torchvision

from pathlib import Path
from pytorch_msssim import ssim
from torch.nn import MSELoss, Upsample
from torch.utils.data.dataloader import DataLoader
from srdataset import SREvalDataset
from basicsrmodel import BasicSRModel
from residualmodel import ResidualModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
WRITE_PRED_IMGS = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Output directory to save results')
    args = parser.parse_args()
    return args.path

def eval_losses(y_pred, y):
    mse_loss = MSELoss()
    loss_psnr = -10*torch.log10(mse_loss(y_pred, y))
    loss_ssim = ssim(y_pred, y, data_range=1.0, size_average=False)
    return loss_psnr, loss_ssim


def eval_loop(eval_dataloader, model, output_path):
    size = len(eval_dataloader.dataset)
    loss_psnr = [0]*3
    loss_ssim = [0]*3
    bilinear_func = Upsample(scale_factor=2, mode='bilinear')
    bicubic_func = Upsample(scale_factor=2, mode='bicubic')

    for i, batch in enumerate(eval_dataloader):
        low_res, high_res = batch
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        # get predictions
        high_res_prediction = model(low_res)
        bilinear_pred = bilinear_func(low_res)
        bicubic_pred = bicubic_func(low_res)
        high_res_prediction = torch.clamp(high_res_prediction, min=0, max=1)
        if WRITE_PRED_IMGS:
            high_res_prediction = torch.clamp(high_res_prediction, min=0, max=1)
            torchvision.io.write_png(high_res_prediction[0, ...].mul(255).byte().to('cpu'), os.path.join(output_path, f"hr_image{i}.png"))
            torchvision.io.write_png(bilinear_pred[0, ...].mul(255).byte().to('cpu'), os.path.join(output_path, f"bilinear_image{i}.png"))
            torchvision.io.write_png(bicubic_pred[0, ...].mul(255).byte().to('cpu'), os.path.join(output_path, f"bicubic_image{i}.png"))
            print("wrote images")

        # compute the loss
        for i, pred in enumerate([high_res_prediction, bilinear_pred, bicubic_pred]):
            l_p, l_s = eval_losses(pred, high_res)
            loss_psnr[i] += l_p
            loss_ssim[i] += l_s

    loss_psnr = [loss/size for loss in loss_psnr]
    loss_ssim = [loss/size for loss in loss_ssim]
    return loss_psnr, loss_ssim


output_path = Path(parse_arguments())
eval_dataset = SREvalDataset("./data/eval/")
eval_dataloader = DataLoader(eval_dataset, batch_size=1,
                             shuffle=False, num_workers=1, drop_last=False, pin_memory=False)
model = BasicSRModel(num_blocks=10).to(device)
# model = ResidualModel(num_blocks=10).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

with torch.no_grad():
    loss_psnr, loss_ssim = eval_loop(eval_dataloader, model, output_path)
    print(f"PSNR: {loss_psnr}")
    print(f"SSIM: {loss_ssim}")
    with open(os.path.join(output_path, "results.txt"), 'w') as output_file:
        output_file.write("PSNR:\n")
        output_file.write(f"Ours: {loss_psnr[0]} Bilinear: {loss_psnr[1]} Bicubic: {loss_psnr[2]}\n")
        output_file.write("SSIM:\n")
        output_file.write(f"Ours: {loss_ssim[0]} Bilinear: {loss_ssim[1]} Bicubic: {loss_ssim[2]}\n")
