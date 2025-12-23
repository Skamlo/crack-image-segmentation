def calculate_dice(preds, targets, smooth=1e-6):
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()
