
# NuSPIRe: Nuclear Morphology focused Self-supervised Pretrained model for Image Representations



## Model description

NuSPIRe (Nuclear Morphology focused Self-supervised Pretrained model for Image Representations) is a deep learning framework designed to extract nuclear morphological features from DAPI-stained images for guiding field-of-view (FOV) optimization in spatial omics. Trained using self-supervised learning on 15.52 million unlabeled nuclear images from diverse tissues, NuSPIRe leverages pre-existing imaging data to identify biologically informative regions. While primarily developed for FOV selection and layout refinement, it also offers potential for broader morphology-based spatial inference.
![Overview Of NuSPIRe](https://huggingface.co/TongjiZhanglab/NuSPIRe/resolve/main/Images/model_overview.png)

## Training Details

- **Pretraining Dataset**: NuCorpus-15M, a dataset comprising 15.52 million cell nucleus images from both human and mouse tissues, spanning 15 different organs or tissue types.
- **Input**: The model processes DAPI-stained images of cell nuclei, which are commonly used to visualize nuclear structure.
- **Tasks**: NuSPIRe is capable of handling various downstream tasks, including cell type identification, perturbation detection, and predicting gene expression levels.
- **Framework**: The model is implemented in PyTorch and is available for fine-tuning on specific tasks.
- **Pre-training Strategy**: NuSPIRe was trained using a Masked Image Modeling (MIM) approach for self-supervised learning, allowing it to extract meaningful features from nuclear morphology without needing labeled data.
- **Downstream Performance**: The model significantly outperforms traditional methods in few-shot learning tasks and performs robustly even with very small amounts of labeled data.

## Usage

### Representation Extraction

To extract representations using the pre-trained NuSPIRe model, please refer to the code example below:

```python
# Import necessary libraries
import torch
import requests
from PIL import Image
import lightning.pytorch as pl
from transformers import ViTModel
from torchvision import transforms

# Set random seed for reproducibility
pl.seed_everything(0, workers=True)

# Open an example image
url = 'https://huggingface.co/TongjiZhanglab/NuSPIRe/resolve/main/Images/image_aabhacci-1.png'
image = Image.open(requests.get(url, stream=True).raw).convert('L')

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize image to 112x112 pixels
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.21869252622127533], std=[0.1809280514717102])  # Normalize single channel
])

# Apply the transformations and add a batch dimension
image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 112, 112]

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)

# Load the NuSPIRe model
model = ViTModel.from_pretrained("TongjiZhanglab/NuSPIRe")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Disable gradient calculation for faster inference
with torch.no_grad():
    # Perform forward pass through the model
    outputs = model(pixel_values=image_tensor)
    # Extract the pooled output representation
    representation = outputs.pooler_output

# Move the representation to CPU and convert to NumPy array (if further processing is needed)
representation = representation.cpu().numpy()

# Print the output representation
print(representation)

```

### Fine-tuning

NuSPIRe can be fine-tuned on smaller labeled datasets for specific tasks:

```python
# Import necessary libraries
import torch
import requests
from PIL import Image
import lightning.pytorch as pl
from torchvision import transforms
from transformers import ViTForImageClassification

# Set random seed for reproducibility
pl.seed_everything(0, workers=True)

# Open an example image
url = 'https://huggingface.co/TongjiZhanglab/NuSPIRe/resolve/main/Images/image_aabhacci-1.png'
image = Image.open(requests.get(url, stream=True).raw).convert('L')

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize image to 112x112 pixels
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.21869252622127533], std=[0.1809280514717102])  # Normalize single channel
])

# Apply the transformations and add a batch dimension
image_tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 112, 112]

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)

# Load the NuSPIRe model for image classification
model = ViTForImageClassification.from_pretrained("TongjiZhanglab/NuSPIRe", num_labels=2)
model.to(device)
model.train()  # Set the model to training mode

# Prepare the labels tensor (example with label 0)
labels = torch.tensor([0]).to(device)

# Forward pass: Compute outputs and loss
outputs = model(image_tensor, labels=labels)
logits = outputs.logits
loss = outputs.loss

# Backward pass: Compute gradients
loss.backward()

# Print the outputs for verification
print("Logits:", logits)
print("Loss:", loss.item())

```
<!-- 
## Citation

If you use NuSPIRe in your research, please cite the following paper:

Hua, Y., Li, S., & Zhang, Y. (2024). NuSPIRe: Nuclear Morphology focused Self-supervised Pretrained model for Image Representations. 
-->

## License

This project is licensed under the MIT License.
See the LICENSE file for details.
