import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
from tqdm import tqdm
import torch

def train(model, device, loader, optimizer, criterion, contrastive_weight=0.2):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch, training=True)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        task_loss = torch.sum(loss_mat) / torch.sum(is_valid)

        contrastive_loss = model.compute_contrastive_loss()

        if torch.isnan(task_loss) or torch.isnan(contrastive_loss):
            print(f"Warning: NaN loss value detected in step {step}.")
            print(f"Loss of mandate: {task_loss.item()}, comparative loss: {contrastive_loss.item()}")
            continue

        loss = task_loss + contrastive_weight * contrastive_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Task Loss: {task_loss.item():.4f}, Contrastive Loss: {contrastive_loss.item():.4f}")


def train_reg(args, model, device, loader, optimizer, contrastive_weight=0.2):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch, training=True)
        y = batch.y.view(pred.shape).to(torch.float64)

        task_loss = torch.sum((pred - y) ** 2) / y.size(0)

        contrastive_loss = model.compute_contrastive_loss()

        loss = task_loss + contrastive_weight * contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Task Loss: {task_loss.item():.4f}, Contrastive Loss: {contrastive_loss.item():.4f}")