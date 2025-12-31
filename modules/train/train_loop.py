import torch
from modules.train import train_epoch, evaluate


def train(
        train_loader, test_loader, model, optimizer, criterion,
        device:str="cpu", n_epochs:int=30, save_dir:str="./checkpoints"
    ):
    best_dice = 0.0

    history = {
        "loss": [],
        "val_dice": []
    }

    for epoch in range(n_epochs):
        train_loss, history_loss = train_epoch(train_loader, model, optimizer, criterion, device)
        val_dice = evaluate(test_loader, model, device)
        history["loss"].extend(history_loss)
        history["val_dice"].append(val_dice)
        
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {train_loss:.4f} | Test Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

            torch.save(state_dict, f"{save_dir}/best_model.pth")
            print(">>> Saved Best Model")

    return history_loss
