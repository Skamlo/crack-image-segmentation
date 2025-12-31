from tqdm import tqdm


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    history_loss = []
    
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        _, class_logits = model(images)
        
        loss = criterion(class_logits, labels)
        history_loss.append(loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), history_loss
