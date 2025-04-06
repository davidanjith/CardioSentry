import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.transformer import ECGTransformerModel
from train.trainer import train_model
from data.loader import load_and_split_data
from utilities.gradcam import visualize_gradcam
from utilities.ecg_by_condition import get_ecg_by_condition
from config import MODEL_SAVE_PATH

def main():
    # 1. Load or train models
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading pre-trained models...")
        model = ECGTransformerModel().cuda()
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Training new models...")
        model = train_model()

    # 2. Set models to eval mode but enable gradients for visualization
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    # # 3. Test different ECG conditions
    # conditions = {'Normal': 'NORM', 'Ischemia': 'ISC_', 'MI': 'AMI'}
    #
    # for name, code in conditions.items():
    #     print(f"\n===== Testing {name} Cases =====")
    #     dataset = get_ecg_by_condition(code, max_samples=1)
    #     ecg, true_label = dataset[0]
    #     ecg = ecg.numpy()
    #
    #     # Prediction
    #     with torch.no_grad():
    #         inputs = torch.tensor(ecg).unsqueeze(0).cuda()
    #         outputs = model(inputs)
    #         probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    #         pred_class = outputs.argmax().item()
    #
    #     print(f"True Label: {true_label} ({name})")
    #     print(f"Predicted: {pred_class} (Normal: {probs[0]:.1%}, Ischemia: {probs[1]:.1%}, MI: {probs[2]:.1%})")
    #
    #     # Visualization
    #     fig, cam = visualize_gradcam(model, ecg, true_label)
    #     if fig:
    #         fig.savefig(f'./plots/{name.lower()}_ecg_analysis.png', dpi=300)
    #         plt.close(fig)
    #         print(f"Saved visualization to {name.lower()}_ecg_analysis.png")

    # 4. Test random sample
    print("\n===== Testing Random Sample =====")
    _, _, test_dataset = load_and_split_data()
    random_idx = np.random.randint(len(test_dataset))
    ecg, true_label = test_dataset[random_idx]
    ecg = ecg.numpy()

    with torch.no_grad():
        inputs = torch.tensor(ecg).unsqueeze(0).cuda()
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = outputs.argmax().item()

    print(f"Random Test Sample - True Label: {true_label}")
    print(f"Predicted: {pred_class} (Normal: {probs[0]:.1%}, Ischemia: {probs[1]:.1%}, MI: {probs[2]:.1%})")

    fig, cam = visualize_gradcam(model, ecg, true_label)
    if fig:
        fig.savefig('./plots/random_test_ecg.png', dpi=300)
        plt.close(fig)
        print("Saved visualization to random_test_ecg.png")

if __name__ == "__main__":
    main()