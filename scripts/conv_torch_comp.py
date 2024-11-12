import torch
from safetensors.torch import save_file, load_file
import subprocess
#part1
conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
torch.manual_seed(19)
# Parameter = torch.rand(3,3).unsqueeze(0).unsqueeze(0)
Parameter = torch.rand(1,1,3,3)
# Parameter = torch.tensor([ [1., 2., 3.] , [4., 5., 6.], [7., 8., 9.]]).unsqueeze(0).unsqueeze(0)
conv.weight = torch.nn.Parameter(Parameter)

input_image = torch.rand(2048,2048).unsqueeze(0).unsqueeze(0)  
conv.bias = None
output_image = conv(input_image)
print(input_image)
print(conv.weight)
print(output_image)
print("\n"*3)
save_file({"conv" : conv.weight, "input_image" : input_image}, "test.safetensors")
#part2
arg = "test.safetensors"
warns = subprocess.run([r"path to bin file", arg], capture_output=True, text = True)
print(warns)
# part3
tensors = load_file(r"outputcudnn.safetensors")
print(tensors["output_tensor"])
Delta = output_image - tensors["output_tensor"]
print(torch.linalg.det(Delta)[0]==0)
