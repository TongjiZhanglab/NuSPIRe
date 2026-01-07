import torch
import torch.nn as nn
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import ViTForImageClassification
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import LambdaLR
import argparse


def set_seeds(seed_value=42, cuda_deterministic=False):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        

def warmup_lr_lambda(current_epoch: int, warmup_epochs: int):
    if (current_epoch < warmup_epochs):
        return float(current_epoch + 1) / float(max(1, warmup_epochs))
    return 1.0

# set up
parser = argparse.ArgumentParser(description="Setup experiment parameters")
parser.add_argument('--num', type=int, required=True, help='Number of samples per class')
parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
parser.add_argument('--rep', type=int, required=True, help='Number of replicate')
args = parser.parse_args()
num_samples_per_class = args.num
device = args.device
num_repeats = args.rep
    
SEED = 42
DEVICE = torch.device(f"cuda:{device}")
DATA_DIR  = '../lung5_rep1_cancer_nuclear_image_15micron/'
BATCH_SIZE = 300
NUM_EPOCHS = 30
PORJECT_NAME = f'MLP_Frozen_{num_samples_per_class}_lung5_rep1_Classification'
set_seeds(SEED)
folder_name = f'./{PORJECT_NAME}_checkpoint'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"'{folder_name}'has been created.")
else:
    print(f"'{folder_name}' already exists.")

# Dataset
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.21869252622127533], std=[0.1809280514717102])
])

dataset = ImageFolder(DATA_DIR, transform=transform)
labels = [dataset[i][1] for i in range(len(dataset))]

# Define train and test sizes
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Split
train_indices = indices[:train_size]
valid_indices = indices[train_size:train_size + valid_size]
test_indices = indices[train_size + valid_size:]
class_1_train_indices = [i for i in train_indices if labels[i] == 1]
class_2_train_indices = [i for i in train_indices if labels[i] == 2]
class_0_train_indices = [i for i in train_indices if labels[i] == 0]

for repeat in range(num_repeats):
    np.random.shuffle(class_1_train_indices)
    np.random.shuffle(class_2_train_indices)
    np.random.shuffle(class_0_train_indices)

class_1_train_indices = class_1_train_indices[:num_samples_per_class]
class_2_train_indices = class_2_train_indices[:num_samples_per_class]
class_0_train_indices = class_0_train_indices[:num_samples_per_class]

balanced_train_indices = (
    class_1_train_indices +
    class_2_train_indices +
    class_0_train_indices
)
np.random.shuffle(balanced_train_indices)

train_sampler = SubsetRandomSampler(balanced_train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

# print(balanced_train_indices)
# print(valid_indices)
# print(test_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers= 4)
valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers= 4)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers= 4)


# Model
model = ViTForImageClassification.from_pretrained("/mnt/Storage/home/huayuwei/container_workspace/spCS/2.result/0.pretrain_model/V5/epoch69",num_labels=3)
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model.classifier = MLP(input_dim=768, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, output_dim=3)
model.to(DEVICE)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
writer = SummaryWriter(f'./tensorboard/{PORJECT_NAME}')
step1 = 0
step2 = 0
best_val_loss = float('inf')
best_val_f1 = 0
warmup_epochs = 5
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_lr_lambda(epoch, warmup_epochs))


for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
    model.train()
    train_preds, train_labels = [], []
    loss_list = []
    for i, (x, l) in tqdm(enumerate(train_loader), total=len(train_loader)):
        x = x.to(DEVICE)
        l = l.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(x, labels=l)
        
        loss = outputs.loss
        
        _, predicted = torch.max(outputs.logits, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(l.cpu().numpy())
        
        writer.add_scalar("Step/Train Loss", loss.item(), step1)
        loss_list.append(loss.item())
        
        step1 += 1
        loss.backward()
        optimizer.step()

    train_loss = np.mean(loss_list)
    train_accuracy = 100 * (np.array(train_preds) == np.array(train_labels)).mean()
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_precision = precision_score(train_labels, train_preds, average='macro')
    
    model.eval()
    val_preds, val_labels = [], []
    loss_list = []
    with torch.no_grad():
        for i, (x, l) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            x = x.to(DEVICE)
            l = l.to(DEVICE)

            outputs = model(x, labels=l)

            loss = outputs.loss
            
            _, predicted = torch.max(outputs.logits, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(l.cpu().numpy())
            
            writer.add_scalar("Step/Validation Loss", loss.item(), step2)

            loss_list.append(loss.item())
            step2 += 1
    val_loss = np.mean(loss_list)
    val_accuracy = 100 * (np.array(val_preds) == np.array(val_labels)).mean()
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro')
    
    val_labels_bin = label_binarize(val_labels, classes=[0, 1, 2])
    val_preds_bin = label_binarize(val_preds, classes=[0, 1, 2])
    val_auc = roc_auc_score(val_labels_bin, val_preds_bin, average='macro', multi_class='ovr')

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), f'{folder_name}/{PORJECT_NAME}_best_loss_model.pt')
        model.save_pretrained(f'{folder_name}/{PORJECT_NAME}_best_loss_model')
        best_val_loss = val_loss

    # Save the model if the validation F1 score is the best we've seen so far.
    if val_f1 > best_val_f1:
        torch.save(model.state_dict(), f'{folder_name}/{PORJECT_NAME}_best_f1_model.pt')
        model.save_pretrained(f'{folder_name}/{PORJECT_NAME}_best_f1_model')
        best_val_f1 = val_f1

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("Epoch/Lr", lr, epoch)
    writer.add_scalar("Epoch/Validation ROC AUC", val_auc, epoch)
    writer.add_scalars("Epoch/Loss", {'Train Loss': train_loss, 'Validation Loss': val_loss}, epoch)
    writer.add_scalars("Epoch/ACC", {'Train ACC': train_accuracy, 'Validation ACC': val_accuracy}, epoch)
    writer.add_scalars("Epoch/Precision", {'Train Precision': train_precision, 'Validation Precision': val_precision}, epoch)
    writer.add_scalars("Epoch/F1_Score", {'Train F1 Score': train_f1, 'Validation F1 Score': val_f1}, epoch)

    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train ACC: {train_accuracy:.4f}%, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}")
    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation ACC: {val_accuracy:.4f}%, Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation ROC AUC: {val_auc:.4f}")

    scheduler.step()

# Test with best f1 model
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.Grayscale(),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.21869252622127533], std=[0.1809280514717102])
])

dataset = ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
model_path = f'{folder_name}/{PORJECT_NAME}_best_f1_model.pt'
model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
model.eval()
test_preds, test_labels = [], []
test_probs = []

with torch.no_grad():
    for x, l in tqdm(test_loader, total=len(test_loader)):
        x = x.to(DEVICE)
        l = l.to(DEVICE)

        outputs = model(x)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(probabilities, 1)

        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(l.cpu().numpy())
        test_probs.extend(probabilities.cpu().numpy())

test_probs = np.array(test_probs)

df = pd.DataFrame({
    'True Labels': test_labels,
    'Predicted Labels': test_preds
})

for i in range(test_probs.shape[1]):
    df[f'Prob_Class{i}'] = test_probs[:, i]

df.to_csv(f'{PORJECT_NAME}.csv', index=False)
print("Test labels, predictions, and probabilities have been saved")

test_labels_binarized = label_binarize(test_labels, classes=[0, 1, 2])
test_preds_binarized = label_binarize(test_preds, classes=[0, 1, 2])

accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds, average='macro')
precision = precision_score(test_labels, test_preds, average='macro')
recall = recall_score(test_labels, test_preds, average='macro')
rocauc = roc_auc_score(test_labels_binarized, test_preds_binarized, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'ROC AUC: {rocauc:.4f}')

conf_matrix = confusion_matrix(test_labels, test_preds)
print("Confusion Matrix:")
print(conf_matrix)
