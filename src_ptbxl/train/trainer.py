import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from src_ptbxl.config  import BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH
from  src_ptbxl.data.loader import load_and_split_data
from src_ptbxl.models.transformer import ECGTransformerModel

import warnings
warnings.filterwarnings('ignore')

def train_model():
    train_dataset, val_dataset, test_dataset = load_and_split_data()

    # Class distribution summary
    print("\n📊 Class Distributions:")
    print(f"Train: {np.bincount(train_dataset.labels)}")
    print(f"Val:   {np.bincount(val_dataset.labels)}")
    print(f"Test:  {np.bincount(test_dataset.labels)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    model = ECGTransformerModel().cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=torch.tensor([1.0, 2.0, 5.0]).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        for signals, labels in train_loader:
            signals, labels = signals.cuda(), labels.cuda()
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(signals), labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.cuda(), labels.cuda()
                outputs = model(signals)
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f} | Acc: {100*correct/len(val_dataset):.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return model
