import torch
from tqdm import tqdm
from modules.train import calculate_dice


def evaluate(loader, model, device):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, true_masks in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device)
            true_masks = true_masks.to(device)

            mask_logits, _ = model(images)
            preds = (torch.sigmoid(mask_logits) > 0.5).float()
            
            dice_scores.append(calculate_dice(preds, true_masks))

    return sum(dice_scores) / len(dice_scores)
