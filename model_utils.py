import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model(pretrained=True, num_classes=2):
    if pretrained:
        model = models.resnet50(weights="IMAGENET1K_V1") # or v2 ?
    else:
        # Same model, but using random weights
        model = models.resnet50(weights=None)
        
    # Original structure: CNNs - array of values - fc (fully connected layer) - 1000 classes
    in_features = model.fc.in_features # Find number of inputs for original layer
    
    # Change last layer with a 2-class designated one: benign and malignant
    model.fc = nn.Linear(in_features, num_classes)
    
    return model.to(device) # Send the model to GPU /CPU

def get_gradcam(model, image):
    # compute the heatmap on the model's last convolutional layer -> layer 4 in ResNet
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device)) # (H, W, Channels) -> (Batch, Channels, H, W)
        predicted = output.argmax(1).item() # dim 1 -> classes
        probs = torch.softmax(output, dim=1)[0] # probability of the prediction

    # compute the heatmap for the predicted class
    targets = [ClassifierOutputTarget(predicted)] # ignore the other classes
    # send the image, compute gradients (backpropagation), flattening + weights, relu (remove negative values)
    grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device), targets=targets) # just a black and white mask (white is important)


    # for visualization
    img = image.permute(1, 2, 0).numpy() # (Channels, H, W) -> (H, W, Channels)

    # img * std + mean
    img = img * models.ResNet50_Weights.IMAGENET1K_V1.transforms().std + models.ResNet50_Weights.IMAGENET1K_V1.transforms().mean
    img = np.clip(img, 0, 1).astype(np.float32) # limit the values to [0, 1]
    
    # heatmap over the image
    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True) # [0] -> first and only image from the batch (3d object)
    
    return visualization, img, predicted, probs