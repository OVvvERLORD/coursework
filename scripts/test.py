# import torch
# from safetensors.torch import save_file, load_file
# import subprocess
# #part1
# conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
# torch.manual_seed(19)
# # Parameter = torch.rand(3,3).unsqueeze(0).unsqueeze(0)
# Parameter = torch.rand(1,1,3,3)
# # Parameter = torch.tensor([ [1., 2., 3.] , [4., 5., 6.], [7., 8., 9.]]).unsqueeze(0).unsqueeze(0)
# conv.weight = torch.nn.Parameter(Parameter)

# input_image = torch.rand(2048,2048).unsqueeze(0).unsqueeze(0)  
# conv.bias = None
# output_image = conv(input_image)
# print(input_image)
# print(conv.weight)
# print(output_image)
# print("\n"*3)
# save_file({"conv" : conv.weight, "input_image" : input_image}, "test.safetensors")
# #part2
# arg = "test.safetensors"
# warns = subprocess.run([r"path to bin file", arg], capture_output=True, text = True)
# print(warns)
# # part3
# tensors = load_file(r"outputcudnn.safetensors")
# print(tensors["output_tensor"])
# Delta = output_image - tensors["output_tensor"]
# print(torch.linalg.det(Delta)[0]==0)


# import diffusers
# import torch
# test = diffusers.UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
# print(test)




import torch
from safetensors.torch import save_file, load_file
torch.manual_seed(52)
# test_input = torch.rand(3,1280,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp.safetensors")

##layernorm testings
# layernorm = torch.nn.LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
# layernorm.bias = None
# output = layernorm(test_input)
# tensors = load_file(r"C:\study\coursework\src\trash\test_layernorm_rust.safetensors")['output_tensor']
# save_file({"layer_norm_output" : output}, r"C:\study\coursework\src\trash\test_layernorm_python.safetensors")

##groupnorm testings
# test_input = torch.rand(3,1280,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp.safetensors")
# groupnorm = torch.nn.GroupNorm(32, 1280, eps=1e-05, affine=True)
# output = groupnorm(test_input)
# save_file({"group_norm_output" : output}, r"C:\study\coursework\src\trash\test_grnorm_python.safetensors")

##act testings
# silu = torch.nn.SiLU()
# test_input = torch.rand(3,1280,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_act.safetensors")
# save_file({"silu_test" : silu(test_input)}, r"C:\study\coursework\src\trash\test_silu_python.safetensors")

# gelu = torch.nn.GELU()
# test_input = torch.rand(3,1280,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_act.safetensors")
# save_file({"gelu_test" : gelu(test_input)}, r"C:\study\coursework\src\trash\test_gelu_python.safetensors")

##LINEAR
##sym and ubiased
# lin = torch.nn.Linear(in_features=128, out_features=128, bias=False)
# test_input = torch.rand(3,3,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp_linear.safetensors")
# save_file({"lin_w" : lin.weight}, r"C:\study\coursework\src\trash\test_weight_linear.safetensors")
# m = lin(test_input)
# tensors = load_file(r"C:\study\coursework\src\trash\test_linear_rust.safetensors")['output_tensor']
# delta = m - tensors
# print(torch.allclose(tensors, m, 1e-67))
# save_file({"lin_w" : m}, r"C:\study\coursework\src\trash\test_linear_python.safetensors")
##sym and biased
# lin = torch.nn.Linear(in_features=128, out_features=128, bias=True)
# test_input = torch.rand(3,3,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp_bias_linear.safetensors")
# save_file({"lin_w" : lin.weight}, r"C:\study\coursework\src\trash\test_weight_bias_linear.safetensors")
# save_file({"bias_w" : lin.bias}, r"C:\study\coursework\src\trash\test_bias_linear.safetensors")
# m = lin(test_input)
# tensors = load_file(r"C:\study\coursework\src\trash\test_linear_bias_rust.safetensors")['output_tensor']
# delta = m - tensors
# print(torch.allclose(tensors, m, 1e-67))
# save_file({"lin_w" : m}, r"C:\study\coursework\src\trash\test_linear_bias_python.safetensors")
##unsym and unbiased
# lin = torch.nn.Linear(in_features=128, out_features=64, bias=False)
# test_input = torch.rand(3,3,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp_unsym_linear.safetensors")
# save_file({"lin_w" : lin.weight}, r"C:\study\coursework\src\trash\test_weight_unsym_linear.safetensors")
# m = lin(test_input)
# tensors = load_file(r"C:\study\coursework\src\trash\test_linear_unsym_rust.safetensors")['output_tensor']
# delta = m - tensors
# print(torch.allclose(tensors, m, 1e-67))
# save_file({"lin_w" : m}, r"C:\study\coursework\src\trash\test_linear_unsym_python.safetensors")
##unsym and biased
lin = torch.nn.Linear(in_features=128, out_features=64, bias=True)
test_input = torch.rand(3,3,128,128)
save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp_unsym_bias_linear.safetensors")
save_file({"lin_w" : lin.weight}, r"C:\study\coursework\src\trash\test_weight_unsym_bias_linear.safetensors")
save_file({"lin_w" : lin.bias}, r"C:\study\coursework\src\trash\test_unsym_bias_linear.safetensors")
m = lin(test_input)
tensors = load_file(r"C:\study\coursework\src\trash\test_linear_unsym_bias_rust.safetensors")['output_tensor']
delta = m - tensors
print(torch.allclose(tensors, m, 1e-67))
save_file({"lin_w" : m}, r"C:\study\coursework\src\trash\test_linear_unsym_bias_python.safetensors")