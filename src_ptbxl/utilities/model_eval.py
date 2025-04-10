from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import classification_report


def evaluate_model(model, test_dataset, num_classes=3):
    model.eval()
    all_labels = []
    all_probs = []

    for ecg, label in test_dataset:
        inputs = torch.tensor(ecg).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        all_labels.append(label)
        all_probs.append(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)

    # Classification metrics
    acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    #per-class precision, recall, F1
    report = classification_report(all_labels, preds, target_names=["Normal", "Ischemia", "MI"], digits=4)
    print(report)

    #Plot ROC Curve
    fpr = dict()
    tpr = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('./plots/eval_roc_curve.png', dpi=300)
    plt.close()

    #Plot Precision-Recall Curve
    for i in range(num_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        plt.plot(recall_vals, precision_vals, label=f'Class {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('./plots/eval_precision_recall.png', dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Ischemia", "MI"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig("./plots/eval_confusion_matrix.png", dpi=300)
    plt.close()
    #Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Normal", "Ischemia", "MI"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Normalized Confusion Matrix")
    plt.savefig("./plots/eval_confusion_matrix_norm.png", dpi=300)
    plt.close()


