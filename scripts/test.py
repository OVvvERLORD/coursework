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




# import torch
# from safetensors.torch import save_file, load_file
# torch.manual_seed(52)
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
# conv = torch.nn.Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
# conv.bias = None
# print(conv.weight.shape)
# test_input = torch.rand(4, 960, 64, 64)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_kernel_out.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_kernel_out.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_kernel_out_python.safetensors")
# print(res, res.shape)

import diffusers
import torch
from safetensors.torch import save_file, load_file
torch.manual_seed(52)
unet = diffusers.UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
# print(unet)
test_image = torch.rand(2, 320, 320, 320)
temb = torch.rand(2, 1280)
# save_file({"test_image": test_image}, r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors")
# save_file({"test_image": temb}, r"C:\study\coursework\src\trash\test_resnet_temb.safetensors")
resnet_list = []
upsample2d_list = []
downsample2d_list = []
for i, down_block in enumerate(unet.down_blocks):
    # print(f"Down Block {i}:")
    for j, resnet in enumerate(down_block.resnets):
        # print(f"  ResNet Layer {j}: {resnet}")
        resnet_list.append(resnet)
    for k, down_block_deeper in enumerate(down_block.named_children()):

        if down_block_deeper[0] == 'downsamplers':
            for r, downsample2d in enumerate(down_block_deeper[1].named_children()):
                downsample2d_list.append(downsample2d[1])
    if i == 0:
        downblock2d = down_block
        # print(down_block)
for i, up_block in enumerate(unet.up_blocks):
    for j, resnet in enumerate(up_block.resnets):
        resnet_list.append(resnet)
    for j, block in enumerate(up_block.named_children()):
        if block[0] == 'upsamplers':
            for k, smth in enumerate(block[1].named_children()):
                upsample2d_list.append(smth[1])
                upsample2d = smth[1]
    if i == 2:
        upblock2d = up_block
    # print(up_block)
for resnet in unet.mid_block.resnets:
     resnet_list.append(resnet)



## downblock testings
downblock2d_resnet_list = []
for i, resnet in enumerate(downblock2d.resnets):
    downblock2d_resnet_list.append(resnet)
for j, block in enumerate(downblock2d.named_children()):
    if j == 1:
        downsample = block[1]


# print(downblock2d_resnet_list)


temb = torch.rand(2, 1280)
save_file({"temb" : temb}, r"C:\study\coursework\src\trash\test_downblock2d_temb.safetensors")
downblock2d_test = torch.rand(2, 320, 128, 128)
save_file({"test" : downblock2d_test}, r"C:\study\coursework\src\trash\test_downblock2d_test.safetensors")



resnet_1 = downblock2d_resnet_list[0]
resnet_1.norm1.affine = False
resnet_1.norm1.weight = None
resnet_1.norm2.weight = None
resnet_1.norm1.bias = None
resnet_1.norm2.bias = None
resnet_1.norm2.affine = False
resnet_1.conv1.bias = None
resnet_1.conv2.bias = None
save_file({"conv1_weight" : resnet_1.conv1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_weight.safetensors")
save_file({"conv2_weight" : resnet_1.conv2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_weight.safetensors")
save_file({"linear_proj" : resnet_1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_weight.safetensors")
save_file({"linear_proj" : resnet_1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_bias.safetensors")

resnet_2 = downblock2d_resnet_list[1]
resnet_2.norm1.affine = False
resnet_2.norm1.weight = None
resnet_2.norm2.weight = None
resnet_2.norm1.bias = None
resnet_2.norm2.bias = None
resnet_2.norm2.affine = False
resnet_2.conv1.bias = None
resnet_2.conv2.bias = None
save_file({"conv1_weight" : resnet_2.conv1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_weight.safetensors")
save_file({"conv2_weight" : resnet_2.conv2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_weight.safetensors")
save_file({"linear_proj" : resnet_2.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_weight.safetensors")
save_file({"linear_proj" : resnet_2.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_bias.safetensors")

for i, block in enumerate(downsample.named_children()):
    conv_downsample = block[1]
conv_downsample.conv.bias = None
save_file({"conv_down_weight": conv_downsample.conv.weight}, r"C:\study\coursework\src\trash\test_downblock2d_downsample.safetensors")
output = downblock2d(downblock2d_test, temb = temb)
save_file({"downsample2d_out": output[0]}, r"C:\study\coursework\src\trash\test_downsample2d_output.safetensors")

save_file({"downsample2d_out_hidden" : output[1][0]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden1.safetensors")
save_file({"downsample2d_out_hidden" : output[1][1]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden2.safetensors")
save_file({"downsample2d_out_hidden" : output[1][2]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden3.safetensors")














## upblock testings
# upblock2d_resnet_list = []
# for i, resnet in enumerate(upblock2d.resnets):
#     upblock2d_resnet_list.append(resnet)
# res_hidden = torch.rand(2, 320, 128, 128)
# upblock2d_test = torch.rand(2, 640, 128, 128)
# temb = torch.rand(2, 1280)
# save_file({"temb" : temb}, r"C:\study\coursework\src\trash\test_upblock2d_temb.safetensors")
# save_file({"rhidden" : res_hidden}, r"C:\study\coursework\src\trash\test_upblock2d_res_hidden.safetensors")
# save_file({"test" : upblock2d_test}, r"C:\study\coursework\src\trash\test_upblock2d_test.safetensors")

# resnet_1 = upblock2d_resnet_list[0]
# resnet_1.norm1.weight = None
# resnet_1.norm2.weight = None
# resnet_1.norm1.bias = None
# resnet_1.norm2.bias = None
# resnet_1.conv1.bias = None
# resnet_1.conv2.bias = None
# resnet_1.conv_shortcut.bias = None
# save_file({"conv1_weight" : resnet_1.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_weight.safetensors")
# save_file({"conv2_weight" : resnet_1.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_weight.safetensors")
# save_file({"linear_proj" : resnet_1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_1.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_weight.safetensors")

# resnet_2 = upblock2d_resnet_list[1]
# resnet_2.norm1.weight = None
# resnet_2.norm2.weight = None
# resnet_2.norm1.bias = None
# resnet_2.norm2.bias = None
# resnet_2.conv1.bias = None
# resnet_2.conv2.bias = None
# resnet_2.conv_shortcut.bias = None
# save_file({"conv1_weight" : resnet_2.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_weight.safetensors")
# save_file({"conv2_weight" : resnet_2.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_weight.safetensors")
# save_file({"linear_proj" : resnet_2.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_2.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_2.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_weight.safetensors")


# resnet_3 = upblock2d_resnet_list[2]
# resnet_3.norm1.weight = None
# resnet_3.norm2.weight = None
# resnet_3.norm1.bias = None
# resnet_3.norm2.bias = None
# resnet_3.conv1.bias = None
# resnet_3.conv2.bias = None
# resnet_3.conv_shortcut.bias = None
# save_file({"conv1_weight" : resnet_3.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_weight.safetensors")
# save_file({"conv2_weight" : resnet_3.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_weight.safetensors")
# save_file({"linear_proj" : resnet_3.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_3.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_3.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_weight.safetensors")


# res_hidden_states_tuple = (res_hidden, res_hidden, res_hidden)
# temb = torch.rand(2, 1280)
# save_file({"temb" : temb}, r"C:\study\coursework\src\trash\test_upblock2d_temb.safetensors")
# save_file({"upblock2d_out": upblock2d(upblock2d_test, res_hidden_states_tuple, temb = temb)}, r"C:\study\coursework\src\trash\test_upblock2d_out.safetensors")



## downsample2d testings
## they share common input 
# test_upsample = torch.rand(2, 640, 128, 128)
# downsample2d_test = downsample2d_list[1]
# downsample2d_test.conv.bias = None
# save_file({"downsample2d_conv" : downsample2d_test.conv.weight}, r"C:\study\coursework\src\trash\test_downsample_conv.safetensors")
# save_file({"downsample_out": downsample2d_test(test_upsample)}, r"C:\study\coursework\src\trash\test_downsample_outp.safetensors")

#upsample test
# test_upsample = torch.rand(2, 640, 128, 128)
# upsample2d.conv.bias = None
# upsample2d_output = upsample2d(test_upsample)
# save_file({"upsample2d_conv" : upsample2d.conv.weight}, r"C:\study\coursework\src\trash\test_upsample_conv.safetensors")
# save_file({"upsample2d_input" : test_upsample}, r"C:\study\coursework\src\trash\test_upsample_inp.safetensors")
# save_file({"upsample2d_output" : upsample2d_output}, r"C:\study\coursework\src\trash\test_upsample_outp.safetensors")

## resnet
# print(resnet_list[0])
# print(resnet_list[0].forward(test_image, temb))
## no bias no shortcut
# testings1 = resnet_list[0]
# # print(testings1.norm1.bias, testings1.nonlinearity, testings1.conv1, testings1.norm2, testings1.nonlinearity, testings1.conv2, testings1.conv_shortcut)
# testings1.norm1.weight = None
# testings1.norm2.weight = None
# testings1.norm1.bias = None
# testings1.norm2.bias = None
# testings1.conv1.bias = None
# testings1.conv2.bias = None
# output = testings1(test_image, temb)
# # for i, layer in enumerate(testings1.named_children):
# #     print('\n\n\nThis is layer {layer}', layer)
# save_file({"test_image": test_image}, r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors")
# save_file({"test_image": temb}, r"C:\study\coursework\src\trash\test_resnet_temb.safetensors")
# save_file({"conv1_weight" : testings1.conv1.weight},  r"C:\study\coursework\src\trash\test_resnet_conv1_weight.safetensors")
# # print(testings1.conv1.weight.shape)
# save_file({"conv2_weight" : testings1.conv2.weight},  r"C:\study\coursework\src\trash\test_resnet_conv2_weight.safetensors")
# save_file({"linear_proj" : testings1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_resnet_linear_weight.safetensors")
# save_file({"linear_proj" : testings1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_resnet_linear_bias.safetensors")
# # print(testings1.time_emb_proj, testings1.time_emb_proj.weight, testings1.time_emb_proj.bias)
# print(testings1.named_children)
# print(output.shape)
# norm1 = testings1.norm1
# norm1.bias = None
# norm1.weight = None
# silu = testings1.nonlinearity
# conv1 = testings1.conv1
# conv1.bias = None
# lin = testings1.time_emb_proj
# temb_act = silu(temb)
# temb_act = lin(temb_act)
# norm2 = testings1.norm2
# drop = testings1.dropout
# conv2 = testings1.conv2
# conv2.bias = None
# # hand_output = conv2(drop(silu(norm2(conv1(silu(norm1(test_image))) + temb_act[:, :, None, None]))))
# # print(hand_output)
# print(output)
# save_file({"resnet_no_shortcut" : output}, r"C:\study\coursework\src\trash\test_resnet_output.safetensors")
## no bias shorcut
# print(resnet_list[2])
# testings2 = resnet_list[2]
# testings2.norm1.weight = None
# testings2.norm2.weight = None
# testings2.norm1.bias = None
# testings2.norm2.bias = None
# testings2.conv1.bias = None
# testings2.conv2.bias = None
# testings2.conv_shortcut.bias = None
# norm1 = testings2.norm1
# norm1.bias = None
# norm1.weight = None
# silu = testings2.nonlinearity
# conv1 = testings2.conv1
# conv1.bias = None
# lin = testings2.time_emb_proj
# temb_act = silu(temb)
# temb_act = lin(temb_act)
# norm2 = testings2.norm2
# drop = testings2.dropout
# conv2 = testings2.conv2
# conv2.bias = None
# conv_short = testings2.conv_shortcut
# conv_short.bias = None
# save_file({"conv1_weight" : testings2.conv1.weight},  r"C:\study\coursework\src\trash\test_resnet_short_conv1_weight.safetensors")
# # print(testings1.conv1.weight.shape)
# save_file({"conv2_weight" : testings2.conv2.weight},  r"C:\study\coursework\src\trash\test_resnet_short_conv2_weight.safetensors")
# save_file({"linear_proj" : testings2.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_resnet_short_linear_weight.safetensors")
# save_file({"linear_proj" : testings2.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_resnet_short_linear_bias.safetensors")
# save_file({"conv_short_weight" : testings2.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_resnet_short_conv_short_weight.safetensors")
# save_file({"resnet_shortcut": testings2(test_image, temb)},  r"C:\study\coursework\src\trash\test_resnet_short_output.safetensors")
# print(testings2(test_image, temb))


# for i, block in enumerate(unet.mid_block.attentions.named_children()):
#     for j, transformer_block in enumerate(block[1].named_children()): # block = transformer2dmodel
#         if transformer_block[0] == 'transformer_blocks':
#             for k, block_basic in enumerate(transformer_block[1].named_children()): # block = basictransformerblock
#                 if block_basic[0] == '0':
#                     for k, block_inside_basic in enumerate(block_basic[1].named_children()): # block_inside_basic = layers in bastictransformerblock
#                         if block_inside_basic[0] == 'ff':
#                             ff = block_inside_basic[1]
#                         if block_inside_basic[0] == "attn1":
#                             attn1 = block_inside_basic[1]

# for i, net in enumerate(ff.named_children()):

#     for j, layer in enumerate(net[1].named_children()):
#         if layer[0] == '0':
#                 geglu = layer[1]
#                 lin1 = layer[1].proj
#         if layer[0] == '2':
#             lin2 = layer[1]

## ff test
# test_ff = torch.rand(2, 2, 1280, 1280)
# save_file({"ff_lin1" : lin1.weight}, r"C:\study\coursework\src\trash\ff_lin1.safetensors")
# save_file({"ff_lin1" : lin1.bias}, r"C:\study\coursework\src\trash\ff_lin1_bias.safetensors")
# save_file({"ff_lin2" : lin2.weight}, r"C:\study\coursework\src\trash\ff_lin2.safetensors")
# save_file({"ff_lin2" : lin2.bias}, r"C:\study\coursework\src\trash\ff_lin2_bias.safetensors")
# save_file({"ff_input" : test_ff}, r"C:\study\coursework\src\trash\ff_input.safetensors")
# output_ff = ff(test_ff)
# save_file({"ff_output" : output_ff}, r"C:\study\coursework\src\trash\ff_output.safetensors")


## attn1 test
# for i, block in enumerate(attn1.named_children()):
#     if block[0] == 'to_q':
#         q_lin = block[1]
#     if block[0] == 'to_k':
#         k_lin = block[1]
#     if block[0] == 'to_v':
#         v_lin = block[1]
#     if block[0] == 'to_out':
#         for j, mod in enumerate(block[1].named_children()):
#             if mod[0] == '0':
#                 out_lin = mod[1]
# attn1_input = torch.rand(2, 1280, 1280)
# save_file({"attn1_input" : attn1_input}, r"C:\study\coursework\src\trash\attn1_input.safetensors")
# save_file({"q_lin" : q_lin.weight}, r"C:\study\coursework\src\trash\attn1_q_lin.safetensors")
# save_file({"k_lin" : k_lin.weight}, r"C:\study\coursework\src\trash\attn1_k_lin.safetensors")
# save_file({"v_lin" : v_lin.weight}, r"C:\study\coursework\src\trash\attn1_v_lin.safetensors")
# save_file({"out_lin" : out_lin.weight}, r"C:\study\coursework\src\trash\attn1_out_lin.safetensors")
# # save_file({"q_lin" : q_lin.bias}, r"C:\study\coursework\src\trash\attn1_q_lin_bias.safetensors") #there is no bias, at least, there
# # save_file({"k_lin" : k_lin.bias}, r"C:\study\coursework\src\trash\attn1_k_lin_bias.safetensors")
# # save_file({"v_lin" : v_lin.bias}, r"C:\study\coursework\src\trash\attn1_v_lin_bias.safetensors")
# save_file({"out_lin" : out_lin.bias}, r"C:\study\coursework\src\trash\attn1_out_lin_bias.safetensors")
# output_attn1 = attn1(attn1_input)
# print(output_attn1, output_attn1.shape)