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
# lin = torch.nn.Linear(in_features=128, out_features=64, bias=True)
# test_input = torch.rand(3,3,128,128)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_inp_unsym_bias_linear.safetensors")
# save_file({"lin_w" : lin.weight}, r"C:\study\coursework\src\trash\test_weight_unsym_bias_linear.safetensors")
# save_file({"lin_w" : lin.bias}, r"C:\study\coursework\src\trash\test_unsym_bias_linear.safetensors")
# m = lin(test_input)
# tensors = load_file(r"C:\study\coursework\src\trash\test_linear_unsym_bias_rust.safetensors")['output_tensor']
# delta = m - tensors
# print(torch.allclose(tensors, m, 1e-67))
# save_file({"lin_w" : m}, r"C:\study\coursework\src\trash\test_linear_unsym_bias_python.safetensors")

##conv testings
## standart conv without bias
# conv = torch.nn.Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(10, 1280, 32, 32)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_std.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_std.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_std_python.safetensors")
# print(res, res.shape)
## out_channels < in_channels
# conv = torch.nn.Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(10, 640, 32, 32)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_out.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_out.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_out_python.safetensors")
# print(res, res.shape)
## in_channels < out_channels
# conv = torch.nn.Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(10, 4, 128, 128)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_in.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_in.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_in_python.safetensors")
# print(res, res.shape)
## stride = 2
# conv = torch.nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(4, 320, 128, 128)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_stride.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_stride.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_stride_python.safetensors")
# print(res, res.shape)
##kernel and in < out
# conv = torch.nn.Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(4, 640, 64, 64)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_kernel_in.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_kernel_in.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_kernel_in_python.safetensors")
# print(res, res.shape)
##kernel and out < in
conv = torch.nn.Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
conv.bias = None
print(conv.weight.shape)
test_input = torch.rand(4, 960, 64, 64)
res = conv(test_input)
save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_kernel_out.safetensors")
save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_kernel_out.safetensors")
save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_kernel_out_python.safetensors")
print(res, res.shape)

# import diffusers
# import torch
# # test = diffusers.UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")

