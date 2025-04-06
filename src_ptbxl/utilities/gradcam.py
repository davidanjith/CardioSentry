import torch
import numpy as np
import matplotlib.pyplot as plt
from src_ptbxl.config import TARGET_SAMPLES

def visualize_gradcam(model, ecg_sample, true_label=None, target_layer='transformer.layers.2'):
    """
    Safe GradCAM visualization for ECG predictions
    Args:
        model: Trained ECGTransformerModel (must be in eval mode)
        ecg_sample: Input ECG of shape (12, 3600)
        true_label: Ground truth label
        target_layer: Transformer layer to visualize
    Returns:
        matplotlib figure and attention weights
    """
    try:
        # Store original state
        original_training = model.training
        original_requires_grad = [p.requires_grad for p in model.parameters()]

        # Prepare for GradCAM
        model.eval()  # Ensure dropout/batchnorm are in eval mode
        for param in model.parameters():
            param.requires_grad = True  # Enable gradients for visualization only

        # Prepare input tensor
        input_tensor = torch.tensor(ecg_sample, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        input_tensor.requires_grad_(True)

        # Hook containers
        gradients = []
        activations = []

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Register hooks
        target_module = dict([*model.named_modules()])[target_layer]
        handle_b = target_module.register_backward_hook(backward_hook)
        handle_f = target_module.register_forward_hook(forward_hook)

        # Forward pass
        with torch.enable_grad():  # Temporarily enable gradients
            output = model(input_tensor)
            probas = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
            pred_class = output.argmax().item()

            # Backward pass
            model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0][pred_class] = 1
            (output * one_hot).sum().backward()

        # Process activations
        grads = gradients[-1].mean(dim=1, keepdim=True)  # [1, 12, 1]
        acts = activations[-1]  # [1, 12, 128]

        # Compute CAM
        weights = grads.mean(dim=-1, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        cam = cam.squeeze().cpu().detach().numpy()

        # Visualization
        fig = plt.figure(figsize=(18, 14), dpi=100)
        gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1, 1, 0.5])

        # Title Panel
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis('off')
        diagnosis_map = ["Normal", "Ischemia", "Myocardial Infarction"]
        title_text = f"ECG Attention Analysis | True Label: {diagnosis_map[true_label]}" if true_label is not None else "ECG Attention Analysis"
        ax_title.text(0.5, 0.5, title_text, fontsize=14, ha='center', va='center',
                      bbox=dict(facecolor='whitesmoke', alpha=0.8))

        # Prediction Summary
        ax0 = fig.add_subplot(gs[1])
        ax0.axis('off')
        ax0.text(0.1, 0.5,
                 f"Predicted: {diagnosis_map[pred_class]}\n"
                 f"Confidence: {probas[pred_class]:.1%}\n\n"
                 f"Probability Distribution:\n"
                 f"Normal: {probas[0]:.1%}\n"
                 f"Ischemia: {probas[1]:.1%}\n"
                 f"MI: {probas[2]:.1%}",
                 fontsize=12, bbox=dict(facecolor='whitesmoke', alpha=0.8))

        # Lead II with Attention
        ax1 = fig.add_subplot(gs[2])
        lead_idx = 1
        ax1.plot(ecg_sample[lead_idx], 'b', linewidth=1.5, label='Lead II')
        im = ax1.imshow(np.expand_dims(cam, 0),
                        cmap='jet',
                        aspect='auto',
                        alpha=0.3,
                        extent=[0, TARGET_SAMPLES,
                                ecg_sample[lead_idx].min(),
                                ecg_sample[lead_idx].max()],
                        vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1, label='Attention Intensity')
        ax1.set_title("Lead II with Model Attention")
        ax1.grid(True)

        # All Leads Overview
        ax2 = fig.add_subplot(gs[3])
        for i in range(12):
            offset = i * 1.5
            ax2.plot(ecg_sample[i] + offset, linewidth=0.8)
            ax2.fill_between(np.arange(TARGET_SAMPLES),
                             offset, offset + cam[i] * 1.2,
                             color='red', alpha=0.2)
        ax2.set_yticks(np.arange(12) * 1.5)
        ax2.set_yticklabels([f'L{i + 1} ({cam[i]:.2f})' for i in range(12)])
        ax2.set_title("12-Lead ECG with Attention Weights")
        ax2.grid(True)

        plt.tight_layout()
        return fig, cam

    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return None, None
    finally:
        # Cleanup
        if 'handle_b' in locals(): handle_b.remove()
        if 'handle_f' in locals(): handle_f.remove()
        # Restore original models state
        model.train(original_training)
        for param, req_grad in zip(model.parameters(), original_requires_grad):
            param.requires_grad = req_grad