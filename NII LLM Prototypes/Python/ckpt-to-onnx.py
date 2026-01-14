import torch
import torch.onnx
import torchvision.models as models # Example: using a standard ResNet model
import onnx

# 1. Define or import your specific model architecture
# Replace this with your actual model class definition
model = models.resnet50(pretrained=False)

# 2. Load the checkpoint file
checkpoint_path = 'your_model.ckpt' # Replace with your checkpoint file name
device = torch.device("cpu") # Use "cuda" if GPU is available and desired

# Load the state_dict from the checkpoint
# Note: Checkpoint structures can vary (e.g., some save 'state_dict' within a dict)
try:
    # Common case: checkpoint is the state_dict itself
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
except:
    # Alternative case: state_dict is nested (common in training frameworks)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict']) # Adjust key as necessary

# Set the model to evaluation mode (important for operations like BatchNorm/Dropout)
model.eval()

# 3. Define a dummy input tensor for tracing
# The shape must match the input the model expects (e.g., [batch_size, channels, height, width])
batch_size = 1
input_shape = (3, 224, 224) # Adjust to your model's required input dimensions
dummy_input = torch.randn(batch_size, *input_shape, device=device)

# 4. Export the model to ONNX
output_onnx_path = "model.onnx"
print(f"Starting ONNX export to {output_onnx_path}...")

torch.onnx.export(model,                        # model instance
                  dummy_input,                  # example input
                  output_onnx_path,             # output file name
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=11,             # the ONNX opset version to use
                  do_constant_folding=True,     # optimize constants
                  input_names=['input'],        # name the input node
                  output_names=['output'],      # name the output node
                  dynamic_axes={'input': {0: 'batch_size'}, # dynamic batch size
                                'output': {0: 'batch_size'}})

print("Model export complete.")

# Optional: Verify the ONNX model
try:
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful.")
except onnx.checker.ValidationError as e:
    print(f"ONNX model verification failed: {e}")