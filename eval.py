import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

def eval(args, model, device, loader, criterion):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if np.isnan(y_scores).any() or np.isinf(y_scores).any():
        print("Warning: y_scores contains NaN or infinity values!")
        y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=1.0, neginf=0.0)
        print("NaN and infinity values have been replaced with sensible default values")

    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y ** 2 > 0
    # Loss matrix
    loss_mat = criterion(pred.double(), (y + 1) / 2)
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat) / torch.sum(is_valid)

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            valid_scores = y_scores[is_valid, i]
            valid_true = (y_true[is_valid, i] + 1) / 2

            if np.isnan(valid_scores).any() or np.isinf(valid_scores).any():
                print(f"WARNING: Column {i} contains NaN or infinity values, skip ROC AUC calculation for that column!")
                continue

            try:
                roc_list.append(roc_auc_score(valid_true, valid_scores))
            except Exception as e:
                print(f"Error calculating ROC AUC for column {i}: {e}")
                continue

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    if len(roc_list) == 0:
        print("Warning: ROC AUC calculations failed for all targets!")
        return 0.5, loss

    eval_roc = sum(roc_list) / len(roc_list)

    return eval_roc, loss


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    multi_task = y_true.shape[1] > 1 if len(y_true.shape) > 1 else False

    if not multi_task or len(y_true.shape) == 1:
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()
        mse = mean_squared_error(y_true, y_scores)
        mae = mean_absolute_error(y_true, y_scores)
        rmse = np.sqrt(mse)
        return mse, mae, rmse
    else:
        mse = mean_squared_error(y_true.flatten(), y_scores.flatten())
        mae = mean_absolute_error(y_true.flatten(), y_scores.flatten())
        rmse = np.sqrt(mse)

        task_mse = []
        task_mae = []
        task_rmse = []
        for i in range(y_true.shape[1]):
            task_mse.append(mean_squared_error(y_true[:, i], y_scores[:, i]))
            task_mae.append(mean_absolute_error(y_true[:, i], y_scores[:, i]))
            task_rmse.append(np.sqrt(task_mse[-1]))

        return mse, mae, rmse, task_mse, task_mae, task_rmse