import math
import torch
from pytorch_msssim import ssim
from torch.nn import MSELoss
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from srdataset import SRDataset
from basicsrmodel import BasicSRModel
from residualmodel import ResidualModel


LEARNING_RATE = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()


def eval_losses(y_pred, y):
    mse_loss = MSELoss()
    loss_psnr = -10*torch.log10(mse_loss(y_pred, y))
    loss_ssim = ssim(y_pred, y, data_range=1.0, size_average=False)
    return loss_psnr, loss_ssim


def train_loop(train_dataloader, validation_dataloader, model, loss_fn, optimizer, number_of_epochs=1000):
    size = len(train_dataloader.dataset)
    lowest_loss = math.inf
    patience = 100
    num_iters_without_improvement = 0

    for epoch in range(number_of_epochs):
        print(f"Epoch: {epoch}")
        loss_sum = 0
        for batch_id, batch in enumerate(train_dataloader):
            low_res, high_res = batch
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            # reset the gradient
            optimizer.zero_grad()
            # forward pass through the model
            high_res_prediction = model(low_res)
            # compute the loss
            loss = loss_fn(high_res_prediction, high_res)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            if batch_id % 20 == 0:
                loss, current = loss.item(), batch_id * 4
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        writer.add_scalar("Loss/train", loss, epoch)
        if loss_sum < lowest_loss:
            print(f"New lowest loss: {loss_sum/size:>7f}, saving model")
            lowest_loss = loss_sum
            torch.save(model.state_dict(), 'model_weights.pth')
            num_iters_without_improvement = 0
        else:
            num_iters_without_improvement += 1
            if num_iters_without_improvement >= patience:
                print(f"Trained {patience} epochs without improvement, early stopping...")
                break

        with torch.no_grad():
            eval_loss = 0
            loss_psnr = 0
            loss_ssim = 0
            val_size = len(validation_dataloader.dataset)

            for batch_id, batch in enumerate(validation_dataloader):
                low_res, high_res = batch
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                high_res_prediction = model(low_res)
                loss = loss_fn(high_res_prediction, high_res)
                eval_loss += loss.item()
                l_p, l_s = eval_losses(high_res_prediction, high_res)
                loss_psnr += l_p
                loss_ssim += l_s
            eval_loss /= val_size
            loss_psnr /= val_size
            loss_ssim /= val_size
            writer.add_scalar("Loss/eval/L1", eval_loss, epoch)
            writer.add_scalar("Loss/eval/psnr", loss_psnr, epoch)
            writer.add_scalar("Loss/eval/ssim", loss_ssim, epoch)



VALIDATION_FRAC = 0.2

dataset = SRDataset("./data/train/")
num_files = len(dataset)
split_idx = int((1-VALIDATION_FRAC) * num_files)
train_indices = list(range(0, split_idx))
validation_indices = list(range(split_idx, num_files))
train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
model = BasicSRModel(num_blocks=10).to(device)
# model = ResidualModel(num_blocks=10).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
loss_fn = torch.nn.L1Loss().to(device)
train_loop(train_dataloader, validation_dataloader, model, loss_fn, optimizer)
