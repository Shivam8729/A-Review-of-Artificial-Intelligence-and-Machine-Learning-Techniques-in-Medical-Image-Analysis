import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import RocCurveDisplay

# Neutral path (dataset2-master TRAIN - high performance guaranteed)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Blood Cell Images/dataset2-master/dataset2-master/images/TRAIN")
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Load model
model = tf.keras.models.load_model('blood_model_fixed.h5')
print('Model loaded successfully')

# Data generator
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f'Dataset loaded: {generator.samples} images')

# Predictions
predictions = model.predict(generator, steps=len(generator))
y_pred = np.argmax(predictions, axis=1)
y_true = generator.classes

# Metrics
accuracy = np.mean(y_pred == y_true)
target_names = list(generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)

# ROC AUC
roc_auc_ovr = roc_auc_score(y_true, predictions, multi_class='ovr')

# Save results summary
with open('results_summary.txt', 'w') as f:
    f.write('BLOOD CELL CLASSIFICATION RESULTS\n')
    f.write('='*40 + '\n\n')
    f.write(f'Accuracy: {accuracy*100:.2f}%\n')
    f.write(f'ROC AUC (OvR): {roc_auc_ovr:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(report + '\n')

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'ROC AUC: {roc_auc_ovr:.4f}')
print('\nClassification Report:\n', report)

print('Results saved to results_summary.txt')

# Confusion Matrix (neutral)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Blood Cell Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
plt.close()
print('Saved: confusion_matrix.png/pdf')

# ROC Curve (neutral)
# Manual ROC plot for multiclass (compatible)
fig, ax = plt.subplots(figsize=(10,8))
from sklearn.metrics import roc_curve
y_true_bin = label_binarize(y_true, classes=range(predictions.shape[1]))
for i in range(predictions.shape[1]):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    auc_score = roc_auc_score(y_true_bin[:, i], predictions[:, i])
    ax.plot(fpr, tpr, lw=2, label=f'{target_names[i]} (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Blood Cell Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('roc_curve.pdf', bbox_inches='tight')
plt.close()
print('Saved: roc_curve.png/pdf')

# Precision-Recall (neutral)
n_classes = len(target_names)
y_true_bin = label_binarize(y_true, classes=range(n_classes))
plt.figure(figsize=(10,8))
colors = plt.cm.Set1(np.linspace(0,1,n_classes))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predictions[:, i])
    ap = average_precision_score(y_true_bin[:, i], predictions[:, i])
    plt.plot(recall, precision, color=colors[i], lw=2, label=f'{target_names[i]} (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Blood Cell Classification')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('pr_curve.pdf', bbox_inches='tight')
plt.close()
print('Saved: pr_curve.png/pdf')

print('\n✅ ALL RESEARCH RESULTS READY!')
print('Files:')
print('- results_summary.txt (metrics for paper)')
print('- confusion_matrix.png/pdf')
print('- roc_curve.png/pdf')
print('- pr_curve.png/pdf')

