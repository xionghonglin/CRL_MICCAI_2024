from functools import partial
import torch
from tqdm import tqdm
import utils

def eval_psnr(loader, model):
    model.eval()
    metric_fn = utils.calc_psnr
    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        with torch.no_grad():
            imgs = [batch[str(i)][0].cuda() for i in range(1, 8)]
            conds = [batch[str(i)][1].cuda() for i in range(1, 8)]

            preds = [model(imgs[0], conds[0], conds[j])[0] for j in range(1, 7)]

        res = sum(metric_fn(preds[j-1], imgs[j]) for j in range(1, 7)) / 6
        val_res.add(res.item(), imgs[0].shape[0])

    return val_res.item()

