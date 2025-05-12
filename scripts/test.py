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
# import torch
# from safetensors.torch import save_file, load_file
# torch.manual_seed(52)
# conv = torch.nn.Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
# # conv.bias = None
# # print(conv.weight.shape)
# test_input = torch.rand(4, 640, 64, 64)
# res = conv(test_input)
# save_file({"input_test" : test_input}, r"C:\study\coursework\src\trash\test_conv_inp_kernel_in.safetensors")
# save_file({"weight_test" : conv.weight}, r"C:\study\coursework\src\trash\test_conv_weight_kernel_in.safetensors")
# save_file({"bias_test" : conv.bias}, r"C:\study\coursework\src\trash\test_conv_bias_kernel_in.safetensors")
# save_file({"weight_test" : res}, r"C:\study\coursework\src\trash\test_conv_kernel_in_python.safetensors")
# # print(res, res.shape)
# print(res.shape, conv)
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
import matplotlib.pyplot as plt
import torchvision.utils as vutils
torch.manual_seed(52)
unet = diffusers.UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
# print(unet)
# test_image = torch.rand(2, 320, 320, 320)
# temb = torch.rand(2, 1280)
# # save_file({"test_image": test_image}, r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors")
# # save_file({"test_image": temb}, r"C:\study\coursework\src\trash\test_resnet_temb.safetensors")
resnet_list = []
upsample2d_list = []
downsample2d_list = []
crossattnupblock_list = []
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
    else:
        crossattnupblock_list.append(up_block)
    # print(up_block)
for resnet in unet.mid_block.resnets:
     resnet_list.append(resnet)
for i, block in enumerate(unet.mid_block.named_children()):
    if i == 0:
        for j, attn in enumerate(block[1].named_children()):
            trans2d = attn[1]
            for k, trans in enumerate(attn[1].named_children()):
                
                if k == 2: # basic transformer blocks
                    for r, btb in enumerate(trans[1].named_children()):
                        if r == 3:
                            lnorm = btb[1].norm1
                            for q, btb_layer in enumerate(btb[1].named_children()):
                                if btb_layer[0] == "attn1":
                                    attn1 = btb_layer[1]

# print(unet.conv_norm_out)
# norm_b = unet.conv_norm_out
# save_file({"norm": norm_b.weight}, r"C:\study\coursework\src\trash\test_grnorm_bias_w.safetensors")
# save_file({"norm": norm_b.bias}, r"C:\study\coursework\src\trash\test_grnorm_bias_b.safetensors")

# grnorm_input = torch.rand(2, 320, 128, 128)
# save_file({"norm": grnorm_input}, r"C:\study\coursework\src\trash\test_grnorm_bias_i.safetensors")
# save_file({"norm": norm_b(grnorm_input)}, r"C:\study\coursework\src\trash\test_grnorm_bias_r.safetensors")

# print(norm_b(grnorm_input))


# print(lnorm.weight.shape, lnorm.bias.shape)
# print(lnorm)
# save_file({"norm": lnorm.weight}, r"C:\study\coursework\src\trash\test_lnorm_bias_w.safetensors")
# save_file({"norm": lnorm.bias}, r"C:\study\coursework\src\trash\test_lnorm_bias_b.safetensors")

# lnorm_input = torch.rand(2, 64, 64, 1280)
# save_file({"norm": lnorm_input}, r"C:\study\coursework\src\trash\test_lnorm_bias_i.safetensors")
# save_file({"norm": lnorm(lnorm_input)}, r"C:\study\coursework\src\trash\test_lnorm_bias_r.safetensors")

# print(lnorm(lnorm_input))









# print(unet)






## unet synth testings
## для удобства поставить None в .config.addition_embed_type

# unet.config.addition_embed_type = None
# # print(unet.time_proj)
# unet_encoder = torch.rand(2, 1280, 2048)
# unet_input = torch.rand(2, 4, 8, 8)
# save_file({"input" : unet_input}, r"C:\study\coursework\src\trash\test_unet_input.safetensors")
# save_file({"input" : unet_encoder}, r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")
# save_file({"input" : unet.time_embedding.linear_1.weight}, r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")
# save_file({"input" : unet.time_embedding.linear_1.bias}, r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")
# save_file({"input" : unet.time_embedding.linear_2.weight}, r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")
# save_file({"input" : unet.time_embedding.linear_2.bias}, r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")
# # output = unet(unet_input,  timestep = 0, encoder_hidden_states = unet_encoder)
# # print(output[0].shape)
# # hand_temb = unet.get_time_embed(sample=unet_input, timestep=0)
# # hand_temb = unet.time_embedding(hand_temb)
# # save_file({"output" : hand_temb}, r"C:\study\coursework\src\trash\test_unet_temb_output.safetensors")
# # print(hand_temb, hand_temb.shape)


# for i, block in enumerate(unet.up_blocks):
#     if i == 2:
#         for j, y in enumerate(block.resnets):
#             y.norm1.weight = None
#             y.norm2.weight = None
#             y.norm1.bias = None
#             y.norm2.bias = None
#             y.norm1.affine = False
#             y.norm2.affine = False
#             y.conv1.bias = None
#             y.conv2.bias = None
#             y.conv_shortcut.bias = None
#             save_file({"conv1_weight" : y.conv1.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res{j}_conv1_weight.safetensors")
#             save_file({"conv2_weight" : y.conv2.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res{j}_conv2_weight.safetensors")
#             save_file({"linear_proj" : y.time_emb_proj.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res{j}_linear_weight.safetensors")
#             save_file({"linear_proj" : y.time_emb_proj.bias},  fr"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res{j}_linear_bias.safetensors")
#             save_file({"conv_short_weight" : y.conv_shortcut.weight},   fr"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res{j}_conv_short_weight.safetensors")
#     else:
#         for j, y in enumerate(block.resnets):
#             y.norm1.weight = None
#             y.norm2.weight = None
#             y.norm1.bias = None
#             y.norm2.bias = None
#             y.norm1.affine = False
#             y.norm2.affine = False
#             y.conv1.bias = None
#             y.conv2.bias = None
#             y.conv_shortcut.bias = None
#             save_file({"res_conv1": y.conv1.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_resnet{j}_conv1.safetensors")
#             save_file({"res_conv2": y.conv2.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_resnet{j}_conv2.safetensors")
#             save_file({"res_conv_short": y.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_resnet{j}_conv_short.safetensors")
#             save_file({"res_lin" : y.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_resnet{j}_temb_w.safetensors")
#             save_file({"res_lin" : y.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_resnet{j}_temb_b.safetensors")
#         for j, y in enumerate(block.attentions):
#             # for g, x in enumerate(y.transformer_blocks):
#             #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
#             save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_projin_w_test.safetensors")
#             save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_projin_b_test.safetensors")
#             save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_projout_w_test.safetensors")
#             save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_projout_b_test.safetensors")
#             y.norm.affine = False
#             y.norm.weight = None
#             y.norm.bias = None
#             k=0
#             for x in y.transformer_blocks:
#                 # print(x.num_attention_heads)
#                 x.norm1.bias = None
#                 x.norm1.weight = None
#                 x.norm1.elementwise_affine = False
#                 x.norm2.bias = None
#                 x.norm2.weight = None
#                 x.norm2.elementwise_affine = False
#                 x.norm3.bias = None
#                 x.norm3.weight = None
#                 x.norm3.elementwise_affine = False
#                 save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_q_test.safetensors")
#                 save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_k_test.safetensors")
#                 save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_v_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_out_w_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_out_b_test.safetensors")
                
#                 save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_q_test.safetensors")
#                 save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_k_test.safetensors")
#                 save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_v_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_out_w_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_out_b_test.safetensors")

#                 for g, y in enumerate(x.ff.net.named_children()):
#                     if y[0] == '0':
#                         save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_geglu_w_test.safetensors")
#                         save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_geglu_b_test.safetensors")
#                     elif y[0] == '2':
#                         save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_ff_w_test.safetensors")
#                         save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_trans{j}_btb{k}_ff_b_test.safetensors")
#                 k += 1

#         for r, g in enumerate(block.upsamplers.named_children()):
#             g[1].conv.bias = None
#             save_file({"upsample" :  g[1].conv.weight}, fr"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock{i}_upsample.safetensors")



# for i, x in enumerate(unet.mid_block.resnets):
#     x.norm1.weight = None
#     x.norm2.weight = None
#     x.norm1.bias = None
#     x.norm2.bias = None
#     x.norm1.affine = False
#     x.norm2.affine = False
#     x.conv1.bias = None
#     x.conv2.bias = None
#     save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{i}_conv1.safetensors")
#     save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{i}_conv2.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{i}_temb_w.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{i}_temb_b.safetensors")

# # for j, x in enumerate(y.transformer_blocks):
# #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
# save_file({"in" : unet.mid_block.attentions[0].proj_in.weight} , r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projin_w_test.safetensors")
# save_file({"in" : unet.mid_block.attentions[0].proj_in.bias} , r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projin_b_test.safetensors")
# save_file({"out" : unet.mid_block.attentions[0].proj_out.weight} , r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projout_w_test.safetensors")
# save_file({"out" :unet. mid_block.attentions[0].proj_out.bias} , r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projout_b_test.safetensors")
# unet.mid_block.attentions[0].norm.affine = False
# unet.mid_block.attentions[0].norm.affine = False
# unet.mid_block.attentions[0].norm.weight = None
# unet.mid_block.attentions[0].norm.bias = None
# k=0
# for x in unet.mid_block.attentions[0].transformer_blocks:
#     # print(x.num_attention_heads)
#     x.norm1.bias = None
#     x.norm1.weight = None
#     x.norm1.elementwise_affine = False
#     x.norm2.bias = None
#     x.norm2.weight = None
#     x.norm2.elementwise_affine = False
#     x.norm3.bias = None
#     x.norm3.weight = None
#     x.norm3.elementwise_affine = False
#     save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn1_q_test.safetensors")
#     save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn1_k_test.safetensors")
#     save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn1_v_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn1_out_w_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn1_out_b_test.safetensors")
    
#     save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn2_q_test.safetensors")
#     save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn2_k_test.safetensors")
#     save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn2_v_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn2_out_w_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_attn2_out_b_test.safetensors")

#     for g, y in enumerate(x.ff.net.named_children()):
#         if y[0] == '0':
#             save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_geglu_w_test.safetensors")
#             save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_geglu_b_test.safetensors")
#         elif y[0] == '2':
#             save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_ff_w_test.safetensors")
#             save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{k}_ff_b_test.safetensors")
#     k += 1



# for i, block in enumerate(unet.down_blocks.named_children()):
#     if i == 0: 
#         for j, x in enumerate(block[1].resnets):
#             x.norm1.affine = False
#             x.norm1.weight = None
#             x.norm2.weight = None
#             x.norm1.bias = None
#             x.norm2.bias = None
#             x.norm2.affine = False
#             x.conv1.bias = None
#             x.conv2.bias = None
#             save_file({"conv1_weight" : x.conv1.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res{j}_conv1_weight.safetensors")
#             save_file({"conv2_weight" : x.conv2.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res{j}_conv2_weight.safetensors")
#             save_file({"linear_proj" : x.time_emb_proj.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res{j}_linear_weight.safetensors")
#             save_file({"linear_proj" : x.time_emb_proj.bias},  fr"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res{j}_linear_bias.safetensors")

#         block[1].downsamplers[0].conv.bias = None
#         save_file({"downsample" : block[1].downsamplers[0].conv.weight}, r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_downsample.safetensors")
        
#     else:
#         for j, x in enumerate(block[1].resnets):
#             x.norm1.weight = None
#             x.norm2.weight = None
#             x.norm1.bias = None
#             x.norm2.bias = None
#             x.norm1.affine = False
#             x.norm2.affine = False
#             x.conv1.bias = None
#             x.conv2.bias = None
#             if j == 0:
#                 x.conv_shortcut.bias = None
#                 save_file({"res_conv1": x.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_resnet{j}_conv_short.safetensors")
#             save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_resnet{j}_conv1.safetensors")
#             save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_resnet{j}_conv2.safetensors")
#             save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_resnet{j}_temb_w.safetensors")
#             save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_resnet{j}_temb_b.safetensors")

#         for j, y in enumerate(block[1].attentions):
#             # for k, x in enumerate(y.transformer_blocks):
#             #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)

#             save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_projin_w_test.safetensors")
#             save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_projin_b_test.safetensors")
#             save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_projout_w_test.safetensors")
#             save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_projout_b_test.safetensors")
#             y.norm.affine = False
#             y.norm.weight = None
#             y.norm.bias = None
#             k=0
#             for x in y.transformer_blocks:
#                 # print(x.num_attention_heads)
#                 x.norm1.bias = None
#                 x.norm1.weight = None
#                 x.norm1.elementwise_affine = False
#                 x.norm2.bias = None
#                 x.norm2.weight = None
#                 x.norm2.elementwise_affine = False
#                 x.norm3.bias = None
#                 x.norm3.weight = None
#                 x.norm3.elementwise_affine = False
#                 save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_q_test.safetensors")
#                 save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_k_test.safetensors")
#                 save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_v_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_out_w_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_out_b_test.safetensors")
                
#                 save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_q_test.safetensors")
#                 save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_k_test.safetensors")
#                 save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_v_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_out_w_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_out_b_test.safetensors")

#                 for g, y in enumerate(x.ff.net.named_children()):
#                     if y[0] == '0':
#                         save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_geglu_w_test.safetensors")
#                         save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_geglu_b_test.safetensors")
#                     elif y[0] == '2':
#                         save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_ff_w_test.safetensors")
#                         save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_trans{j}_btb{k}_ff_b_test.safetensors")
#                 k += 1

#         if i == 1:
#             for r, x in enumerate(block[1].downsamplers.named_children()):
#                 x[1].conv.bias = None
#                 save_file({"downsample" :  x[1].conv.weight}, fr"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock{i}_downsample.safetensors")



# unet.conv_in.bias = None
# save_file({"conv" : unet.conv_in.weight}, r"C:\study\coursework\src\trash\test_unet_conv_in.safetensors")
# unet.conv_norm_out.affine = False
# unet.conv_norm_out.bias = None
# unet.conv_norm_out.weight = None
# unet.conv_out.bias = None
# save_file({"conv" : unet.conv_out.weight}, r"C:\study\coursework\src\trash\test_unet_conv_out.safetensors")

# # print(unet)


# output = unet(unet_input,  timestep = 2, encoder_hidden_states = unet_encoder)
# print(output[0], output[0].shape)
# save_file({"output" : output[0]}, r"C:\study\coursework\src\trash\test_unet_output.safetensors")



# hand_temb = unet.get_time_embed(sample=unet_input, timestep=0)
# hand_temb = unet.time_embedding(hand_temb)
# print(hand_temb)
# # sample = unet.conv_in(unet_input)
# # all_res = ()
# # for x in unet.down_blocks:
# #     sample, res_samples = x(sample, temb = hand_temb, encoder_hidden_states = unet_encoder)
# #     all_res += res_samples
# # print(sample, sample.shape)































## up blocks testings
# up_blocks = unet.up_blocks
# for i, block in enumerate(up_blocks):
#     if i == 2:
#         for j, y in enumerate(block.resnets):
#             y.norm1.weight = None
#             y.norm2.weight = None
#             y.norm1.bias = None
#             y.norm2.bias = None
#             y.norm1.affine = False
#             y.norm2.affine = False
#             y.conv1.bias = None
#             y.conv2.bias = None
#             y.conv_shortcut.bias = None
#             save_file({"conv1_weight" : y.conv1.weight},  fr"C:\study\coursework\src\trash\test_upblocks_upblock2d_res{j}_conv1_weight.safetensors")
#             save_file({"conv2_weight" : y.conv2.weight},  fr"C:\study\coursework\src\trash\test_upblocks_upblock2d_res{j}_conv2_weight.safetensors")
#             save_file({"linear_proj" : y.time_emb_proj.weight},  fr"C:\study\coursework\src\trash\test_upblocks_upblock2d_res{j}_linear_weight.safetensors")
#             save_file({"linear_proj" : y.time_emb_proj.bias},  fr"C:\study\coursework\src\trash\test_upblocks_upblock2d_res{j}_linear_bias.safetensors")
#             save_file({"conv_short_weight" : y.conv_shortcut.weight},   fr"C:\study\coursework\src\trash\test_upblocks_upblock2d_res{j}_conv_short_weight.safetensors")
#     else:
#         for j, y in enumerate(block.resnets):
#             y.norm1.weight = None
#             y.norm2.weight = None
#             y.norm1.bias = None
#             y.norm2.bias = None
#             y.norm1.affine = False
#             y.norm2.affine = False
#             y.conv1.bias = None
#             y.conv2.bias = None
#             y.conv_shortcut.bias = None
#             save_file({"res_conv1": y.conv1.weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_resnet{j}_conv1.safetensors")
#             save_file({"res_conv2": y.conv2.weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_resnet{j}_conv2.safetensors")
#             save_file({"res_conv_short": y.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_resnet{j}_conv_short.safetensors")
#             save_file({"res_lin" : y.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_resnet{j}_temb_w.safetensors")
#             save_file({"res_lin" : y.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_resnet{j}_temb_b.safetensors")
#         for j, y in enumerate(block.attentions):
#             # for g, x in enumerate(y.transformer_blocks):
#             #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
#             save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_projin_w_test.safetensors")
#             save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_projin_b_test.safetensors")
#             save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_projout_w_test.safetensors")
#             save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_projout_b_test.safetensors")
#             y.norm.affine = False
#             y.norm.weight = None
#             y.norm.bias = None
#             k=0
#             for x in y.transformer_blocks:
#                 # print(x.num_attention_heads)
#                 x.norm1.bias = None
#                 x.norm1.weight = None
#                 x.norm1.elementwise_affine = False
#                 x.norm2.bias = None
#                 x.norm2.weight = None
#                 x.norm2.elementwise_affine = False
#                 x.norm3.bias = None
#                 x.norm3.weight = None
#                 x.norm3.elementwise_affine = False
#                 save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_q_test.safetensors")
#                 save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_k_test.safetensors")
#                 save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_v_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_out_w_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn1_out_b_test.safetensors")
                
#                 save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_q_test.safetensors")
#                 save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_k_test.safetensors")
#                 save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_v_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_out_w_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_attn2_out_b_test.safetensors")

#                 for g, y in enumerate(x.ff.net.named_children()):
#                     if y[0] == '0':
#                         save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_geglu_w_test.safetensors")
#                         save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_geglu_b_test.safetensors")
#                     elif y[0] == '2':
#                         save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_ff_w_test.safetensors")
#                         save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_trans{j}_btb{k}_ff_b_test.safetensors")
#                 k += 1

#         for r, g in enumerate(block.upsamplers.named_children()):
#             g[1].conv.bias = None
#             save_file({"upsample" :  g[1].conv.weight}, fr"C:\study\coursework\src\trash\test_upblocks_crossattnupblock{i}_upsample.safetensors")



# upblocks_input = torch.rand(2, 1280, 8, 8)
# upblocks_temb = torch.rand(2, 1280)
# upblocks_encoder = torch.rand(2, 1280, 2048)

# for_res_hidden_input = torch.rand(2, 320, 32, 32)
# for_res_hidden_tuple = (for_res_hidden_input, )
# for x in unet.down_blocks:
#     for_res_hidden_input, res_samples = x(for_res_hidden_input, temb = upblocks_temb , encoder_hidden_states = upblocks_encoder)
#     for_res_hidden_tuple += res_samples

# for i, x in enumerate(for_res_hidden_tuple):
#     save_file({"input" : x}, fr"C:\study\coursework\src\trash\test_upblocks_res_hidden{i}.safetensors")
# print(len(for_res_hidden_tuple))
# upblocks_input = unet.mid_block(for_res_hidden_input, temb = upblocks_temb, encoder_hidden_states = upblocks_encoder)
# output = upblocks_input

# for x in up_blocks:
#     res_hidden_states = for_res_hidden_tuple[-3 :]
#     for_res_hidden_tuple = for_res_hidden_tuple[ : -3]
#     output = x(output, res_hidden_states_tuple=res_hidden_states, temb=upblocks_temb, encoder_hidden_states=upblocks_encoder)

# # print(output.shape)
# save_file({"input" : upblocks_temb}, r"C:\study\coursework\src\trash\test_upblocks_temb.safetensors")
# save_file({"input" : upblocks_encoder}, r"C:\study\coursework\src\trash\test_upblocks_encoder.safetensors")
# save_file({"input" : upblocks_input}, r"C:\study\coursework\src\trash\test_upblocks_input.safetensors")

# print(output)
# save_file({"output" : output}, r"C:\study\coursework\src\trash\test_upblocks_output.safetensors")
# # print(up_blocks)










## down blocks testings
# down_blocks = unet.down_blocks

# for i, block in enumerate(down_blocks.named_children()):
#     if i == 0: 
#         for j, x in enumerate(block[1].resnets):
#             x.norm1.affine = False
#             x.norm1.weight = None
#             x.norm2.weight = None
#             x.norm1.bias = None
#             x.norm2.bias = None
#             x.norm2.affine = False
#             x.conv1.bias = None
#             x.conv2.bias = None
#             save_file({"conv1_weight" : x.conv1.weight},  fr"C:\study\coursework\src\trash\test_downblocks_downblock2d_res{j}_conv1_weight.safetensors")
#             save_file({"conv2_weight" : x.conv2.weight},  fr"C:\study\coursework\src\trash\test_downblocks_downblock2d_res{j}_conv2_weight.safetensors")
#             save_file({"linear_proj" : x.time_emb_proj.weight},  fr"C:\study\coursework\src\trash\test_downblocks_downblock2d_res{j}_linear_weight.safetensors")
#             save_file({"linear_proj" : x.time_emb_proj.bias},  fr"C:\study\coursework\src\trash\test_downblocks_downblock2d_res{j}_linear_bias.safetensors")

#         block[1].downsamplers[0].conv.bias = None
#         save_file({"downsample" : block[1].downsamplers[0].conv.weight}, r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors")
        
#     else:
#         for j, x in enumerate(block[1].resnets):
#             x.norm1.weight = None
#             x.norm2.weight = None
#             x.norm1.bias = None
#             x.norm2.bias = None
#             x.norm1.affine = False
#             x.norm2.affine = False
#             x.conv1.bias = None
#             x.conv2.bias = None
#             if j == 0:
#                 x.conv_shortcut.bias = None
#                 save_file({"res_conv1": x.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_resnet{j}_conv_short.safetensors")
#             save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_resnet{j}_conv1.safetensors")
#             save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_resnet{j}_conv2.safetensors")
#             save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_resnet{j}_temb_w.safetensors")
#             save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_resnet{j}_temb_b.safetensors")

#         for j, y in enumerate(block[1].attentions):
#             # for k, x in enumerate(y.transformer_blocks):
#             #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)

#             save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_projin_w_test.safetensors")
#             save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_projin_b_test.safetensors")
#             save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_projout_w_test.safetensors")
#             save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_projout_b_test.safetensors")
#             y.norm.affine = False
#             y.norm.weight = None
#             y.norm.bias = None
#             k=0
#             for x in y.transformer_blocks:
#                 # print(x.num_attention_heads)
#                 x.norm1.bias = None
#                 x.norm1.weight = None
#                 x.norm1.elementwise_affine = False
#                 x.norm2.bias = None
#                 x.norm2.weight = None
#                 x.norm2.elementwise_affine = False
#                 x.norm3.bias = None
#                 x.norm3.weight = None
#                 x.norm3.elementwise_affine = False
#                 save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_q_test.safetensors")
#                 save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_k_test.safetensors")
#                 save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_v_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_out_w_test.safetensors")
#                 save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn1_out_b_test.safetensors")
                
#                 save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_q_test.safetensors")
#                 save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_k_test.safetensors")
#                 save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_v_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_out_w_test.safetensors")
#                 save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_attn2_out_b_test.safetensors")

#                 for g, y in enumerate(x.ff.net.named_children()):
#                     if y[0] == '0':
#                         save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_geglu_w_test.safetensors")
#                         save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_geglu_b_test.safetensors")
#                     elif y[0] == '2':
#                         save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_ff_w_test.safetensors")
#                         save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_trans{j}_btb{k}_ff_b_test.safetensors")
#                 k += 1

#         if i == 1:
#             for r, x in enumerate(block[1].downsamplers.named_children()):
#                 x[1].conv.bias = None
#                 save_file({"downsample" :  x[1].conv.weight}, fr"C:\study\coursework\src\trash\test_downblocks_crossattndownblock{i}_downsample.safetensors")


# print(down_blocks)

# down_blocks_input = torch.rand(2, 320, 128, 128)
# down_blocks_temb = torch.rand(2, 1280)
# down_blocks_encoder = torch.rand(2, 1280, 2048)
# save_file({"input" : down_blocks_input}, r"C:\study\coursework\src\trash\test_downblocks_input.safetensors")
# save_file({"input" : down_blocks_temb}, r"C:\study\coursework\src\trash\test_downblocks_temb.safetensors")
# save_file({"input" : down_blocks_encoder}, r"C:\study\coursework\src\trash\test_downblocks_encoder.safetensors")
# # down_blocks_output = down_blocks(down_blocks_input, down_blocks_temb, down_block_encoder)\
# sample = down_blocks_input
# all_res = ()
# for x in down_blocks:
#     sample, res_samples = x(sample, temb = down_blocks_temb, encoder_hidden_states = down_blocks_encoder)
#     all_res += res_samples
# print(sample, sample.shape)
# save_file({"output" : sample}, r"C:\study\coursework\src\trash\test_downblocks_output.safetensors")









## crossattndownblock2d testings
# crossattndownblock2d = unet.down_blocks[1]
# i = 0
# for x in crossattndownblock2d.resnets:
#     x.norm1.weight = None
#     x.norm2.weight = None
#     x.norm1.bias = None
#     x.norm2.bias = None
#     x.norm1.affine = False
#     x.norm2.affine = False
#     x.conv1.bias = None
#     x.conv2.bias = None
#     if i == 0:
#         x.conv_shortcut.bias = None
#         save_file({"res_conv1": x.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_resnet{i}_conv_short.safetensors")
#     save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_resnet{i}_conv1.safetensors")
#     save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_resnet{i}_conv2.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_resnet{i}_temb_w.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_crossattndownblock_resnet{i}_temb_b.safetensors")
#     i += 1

# for i, y in enumerate(crossattndownblock2d.attentions):
#     # for j, x in enumerate(y.transformer_blocks):
#     #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)

#     save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_projin_w_test.safetensors")
#     save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_projin_b_test.safetensors")
#     save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_projout_w_test.safetensors")
#     save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_projout_b_test.safetensors")
#     y.norm.affine = False
#     y.norm.weight = None
#     y.norm.bias = None
#     k=0
#     for x in y.transformer_blocks:
#         # print(x.num_attention_heads)
#         x.norm1.bias = None
#         x.norm1.weight = None
#         x.norm1.elementwise_affine = False
#         x.norm2.bias = None
#         x.norm2.weight = None
#         x.norm2.elementwise_affine = False
#         x.norm3.bias = None
#         x.norm3.weight = None
#         x.norm3.elementwise_affine = False
#         save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn1_q_test.safetensors")
#         save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn1_k_test.safetensors")
#         save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn1_v_test.safetensors")
#         save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn1_out_w_test.safetensors")
#         save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn1_out_b_test.safetensors")
        
#         save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn2_q_test.safetensors")
#         save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn2_k_test.safetensors")
#         save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn2_v_test.safetensors")
#         save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn2_out_w_test.safetensors")
#         save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_attn2_out_b_test.safetensors")

#         for g, y in enumerate(x.ff.net.named_children()):
#             if y[0] == '0':
#                 save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_geglu_w_test.safetensors")
#                 save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_geglu_b_test.safetensors")
#             elif y[0] == '2':
#                 save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_ff_w_test.safetensors")
#                 save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_crossattndownblock_trans{i}_btb{k}_ff_b_test.safetensors")
#         k += 1

# for i, x in enumerate(crossattndownblock2d.downsamplers.named_children()):
#     x[1].conv.bias = None
#     save_file({"downsample" :  x[1].conv.weight}, r"C:\study\coursework\src\trash\test_crossattndownblock_downsample.safetensors")



# crossattndownblock2d_input = torch.rand(2, 320, 8, 8)
# crossattndownblock2d_encoder = torch.rand(2, 1280, 2048)
# down_temb = torch.rand(2, 1280)
# save_file({"input": crossattndownblock2d_input}, r"C:\study\coursework\src\trash\test_crossattndownblock_input.safetensors")
# save_file({"input": crossattndownblock2d_encoder}, r"C:\study\coursework\src\trash\test_crossattndownblock_encoder.safetensors")
# save_file({"input": down_temb}, r"C:\study\coursework\src\trash\test_crossattndownblock_temb.safetensors")
# print(crossattndownblock2d)


# down_output = crossattndownblock2d(crossattndownblock2d_input, encoder_hidden_states = crossattndownblock2d_encoder, temb=down_temb)

# save_file({"output" : down_output[0]}, r"C:\study\coursework\src\trash\test_crossattndownblock_output.safetensors")
# save_file({"output": down_output[1][0]}, r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden1.safetensors")
# save_file({"output": down_output[1][1]}, r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden2.safetensors")
# save_file({"output": down_output[1][2]}, r"C:\study\coursework\src\trash\test_crossattndownblock_output_hidden3.safetensors")
# print(down_output[1][0], down_output[1][0].shape)













## crossattnmidblock2d testings
# crossattnmidblock2d = unet.mid_block
# for i, x in enumerate(crossattnmidblock2d.resnets):
#     x.norm1.weight = None
#     x.norm2.weight = None
#     x.norm1.bias = None
#     x.norm2.bias = None
#     x.norm1.affine = False
#     x.norm2.affine = False
#     x.conv1.bias = None
#     x.conv2.bias = None
#     save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{i}_conv1.safetensors")
#     save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{i}_conv2.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{i}_temb_w.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_resnet{i}_temb_b.safetensors")

# # for j, x in enumerate(y.transformer_blocks):
# #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
# save_file({"in" : crossattnmidblock2d.attentions[0].proj_in.weight} , r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_w_test.safetensors")
# save_file({"in" : crossattnmidblock2d.attentions[0].proj_in.bias} , r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projin_b_test.safetensors")
# save_file({"out" : crossattnmidblock2d.attentions[0].proj_out.weight} , r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_w_test.safetensors")
# save_file({"out" : crossattnmidblock2d.attentions[0].proj_out.bias} , r"C:\study\coursework\src\trash\test_crossattnmidblock_trans_projout_b_test.safetensors")
# crossattnmidblock2d.attentions[0].norm.affine = False
# crossattnmidblock2d.attentions[0].norm.affine = False
# crossattnmidblock2d.attentions[0].norm.weight = None
# crossattnmidblock2d.attentions[0].norm.bias = None
# k=0
# for x in crossattnmidblock2d.attentions[0].transformer_blocks:
#     # print(x.num_attention_heads)
#     x.norm1.bias = None
#     x.norm1.weight = None
#     x.norm1.elementwise_affine = False
#     x.norm2.bias = None
#     x.norm2.weight = None
#     x.norm2.elementwise_affine = False
#     x.norm3.bias = None
#     x.norm3.weight = None
#     x.norm3.elementwise_affine = False
#     save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn1_q_test.safetensors")
#     save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn1_k_test.safetensors")
#     save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn1_v_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn1_out_w_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn1_out_b_test.safetensors")
    
#     save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn2_q_test.safetensors")
#     save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn2_k_test.safetensors")
#     save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn2_v_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn2_out_w_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_attn2_out_b_test.safetensors")

#     for g, y in enumerate(x.ff.net.named_children()):
#         if y[0] == '0':
#             save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_geglu_w_test.safetensors")
#             save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_geglu_b_test.safetensors")
#         elif y[0] == '2':
#             save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_ff_w_test.safetensors")
#             save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_crossattnmidblock_trans_btb{k}_ff_b_test.safetensors")
#     k += 1

# crossattnmidblock2d_input = torch.rand(2, 1280, 12, 12)
# crossattnmidblock2d_temb = torch.rand(2, 1280)
# crossattnmidblock2d_encoder = torch.rand(2, 1280, 2048)
# save_file({"inp": crossattnmidblock2d_input}, r"C:\study\coursework\src\trash\test_crossattnmidblock_input.safetensors")
# save_file({"inp": crossattnmidblock2d_encoder}, r"C:\study\coursework\src\trash\test_crossattnmidblock_encoder.safetensors")
# save_file({"inp": crossattnmidblock2d_temb}, r"C:\study\coursework\src\trash\test_crossattnmidblock_temb.safetensors")

# output = crossattnmidblock2d(crossattnmidblock2d_input, temb=crossattnmidblock2d_temb, encoder_hidden_states= crossattnmidblock2d_encoder)
# save_file({"output": output}, r"C:\study\coursework\src\trash\test_crossattnmidblock_output.safetensors")
# print(output, output.shape)












## crosattnupblock2d testings
# crosattnupblock2d = unet.up_blocks[0]
# crosattnupblock2d_resnets = crosattnupblock2d.resnets

# # for i, x in enumerate(unet.up_blocks):
# #     if i != 2:
# #         print(x.dump_patches, x.has_cross_attention, x.num_attention_heads, x.resolution_idx)
# #         is_freeu_enabled = (
# #             getattr(x, "s1", None)
# #             and getattr(x, "s2", None)
# #             and getattr(x, "b1", None)
# #             and getattr(x, "b2", None)
# #         )
# #         print(is_freeu_enabled)
                                    

# for i, x in enumerate(crosattnupblock2d_resnets):
#     x.norm1.weight = None
#     x.norm2.weight = None
#     x.norm1.bias = None
#     x.norm2.bias = None
#     x.norm1.affine = False
#     x.norm2.affine = False
#     x.conv1.bias = None
#     x.conv2.bias = None
#     x.conv_shortcut.bias = None
#     save_file({"res_conv1": x.conv1.weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_resnet{i}_conv1.safetensors")
#     save_file({"res_conv2": x.conv2.weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_resnet{i}_conv2.safetensors")
#     save_file({"res_conv_short": x.conv_shortcut.weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_resnet{i}_conv_short.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_resnet{i}_temb_w.safetensors")
#     save_file({"res_lin" : x.time_emb_proj.bias}, fr"C:\study\coursework\src\trash\test_crossattnupblock_resnet{i}_temb_b.safetensors")

# for i, y in enumerate(crosattnupblock2d.attentions):
#         # for j, x in enumerate(y.transformer_blocks):
#         #     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
#     save_file({"in" : y.proj_in.weight} , fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_projin_w_test.safetensors")
#     save_file({"in" : y.proj_in.bias} , fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_projin_b_test.safetensors")
#     save_file({"out" : y.proj_out.weight} , fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_projout_w_test.safetensors")
#     save_file({"out" : y.proj_out.bias} , fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_projout_b_test.safetensors")
#     y.norm.affine = False
#     y.norm.weight = None
#     y.norm.bias = None
#     k=0
#     for x in y.transformer_blocks:
#         # print(x.num_attention_heads)
#         x.norm1.bias = None
#         x.norm1.weight = None
#         x.norm1.elementwise_affine = False
#         x.norm2.bias = None
#         x.norm2.weight = None
#         x.norm2.elementwise_affine = False
#         x.norm3.bias = None
#         x.norm3.weight = None
#         x.norm3.elementwise_affine = False
#         save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn1_q_test.safetensors")
#         save_file({"k" : x.attn1.to_k.weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn1_k_test.safetensors")
#         save_file({"v" : x.attn1.to_v.weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn1_v_test.safetensors")
#         save_file({"out" : x.attn1.to_out[0].weight},  fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn1_out_w_test.safetensors")
#         save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn1_out_b_test.safetensors")
        
#         save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn2_q_test.safetensors")
#         save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn2_k_test.safetensors")
#         save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn2_v_test.safetensors")
#         save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn2_out_w_test.safetensors")
#         save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_attn2_out_b_test.safetensors")

#         for g, y in enumerate(x.ff.net.named_children()):
#             if y[0] == '0':
#                 save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_geglu_w_test.safetensors")
#                 save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_geglu_b_test.safetensors")
#             elif y[0] == '2':
#                 save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_ff_w_test.safetensors")
#                 save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_crossattnupblock_trans{i}_btb{k}_ff_b_test.safetensors")
#         k += 1

# crossattnupblock2d_input = torch.rand(2, 1280, 2, 2)
# crossattnupblock2d_encoder = torch.rand(2, 1280, 2048)
# crossattnupblock2d_temb = torch.rand(2, 1280)
# res_hidden_big = torch.rand(2, 1280, 2, 2)
# res_hidden_small = torch.rand(2, 640, 2, 2)
# save_file({"inp": crossattnupblock2d_input}, r"C:\study\coursework\src\trash\test_crossattnupblock_input.safetensors")
# save_file({"inp": crossattnupblock2d_encoder}, r"C:\study\coursework\src\trash\test_crossattnupblock_encoder.safetensors")
# save_file({"inp": crossattnupblock2d_temb}, r"C:\study\coursework\src\trash\test_crossattnupblock_temb.safetensors")
# save_file({"inp": res_hidden_small}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1.safetensors")
# save_file({"inp": res_hidden_big}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2.safetensors")
# save_file({"inp": res_hidden_big}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3.safetensors")
# for i, x in enumerate(crosattnupblock2d.upsamplers.named_children()):
#     x[1].conv.bias = None
#     save_file({"upsample" :  x[1].conv.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_upsample.safetensors")
# # crosattnupblock2d.s1 = 1
# # crosattnupblock2d.s2 = 1
# # crosattnupblock2d.b1 = 1
# # crosattnupblock2d.b2 = 1

# # output = crosattnupblock2d(crossattnupblock2d_input, res_hidden_states_tuple=res_hidden_states_tuple, temb=crossattnupblock2d_temb, encoder_hidden_states=crossattnupblock2d_encoder)
# # print(output)
# # save_file({"output": output}, r"C:\study\coursework\src\trash\test_crossattnupblock_output.safetensors")
# crossattnupblock2d_input_large = torch.rand(2, 1280, 16, 16)
# res_hidden_big_large = torch.rand(2, 1280, 16, 16)
# res_hidden_small_large = torch.rand(2, 640, 16, 16)
# res_hidden_states_tuple = (res_hidden_small_large, res_hidden_big_large, res_hidden_big_large)
# save_file({"inp": res_hidden_small_large}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid1_large.safetensors")
# save_file({"inp": res_hidden_big_large}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid2_large.safetensors")
# save_file({"inp": res_hidden_big_large}, r"C:\study\coursework\src\trash\test_crossattnupblock_reshid3_large.safetensors")
# save_file({"inp": crossattnupblock2d_input_large}, r"C:\study\coursework\src\trash\test_crossattnupblock_input_large.safetensors")
# output_large = crosattnupblock2d(crossattnupblock2d_input_large, res_hidden_states_tuple=res_hidden_states_tuple, temb=crossattnupblock2d_temb, encoder_hidden_states=crossattnupblock2d_encoder)
# print(output_large)
# save_file({"output": output_large}, r"C:\study\coursework\src\trash\test_crossattnupblock_output_large.safetensors")

# # hidden_states = torch.cat([crossattnupblock2d_input, res_hidden_big], dim = 1)
# # # print(hidden_states)
# # hand_output = crosattnupblock2d.resnets[0](hidden_states, crossattnupblock2d_temb)


# # print(hand_output)
# # resnet1_hand = crosattnupblock2d.resnets[0].norm1(hidden_states)
# # resnet1_hand = crosattnupblock2d.resnets[0].nonlinearity(resnet1_hand)
# # resnet1_hand = crosattnupblock2d.resnets[0].conv1(resnet1_hand)
# # act_temb = crosattnupblock2d.resnets[0].nonlinearity(temb)
# # print(temb)
# # f_temb = crosattnupblock2d.resnets[0].time_emb_proj(act_temb)
# # print(crosattnupblock2d.resnets[0].conv1.weight)
# # print(crossattnupblock2d_temb)

# # trans1_output = crosattnupblock2d.attentions[0].norm(hand_output)
# # trans1_output = trans1_output.permute(0, 2, 3, 1).reshape(2, 2 * 2, 1280)
# # trans1_output = crosattnupblock2d.attentions[0].proj_in(trans1_output)
# # trans1_output = crosattnupblock2d.attentions[0].transformer_blocks[0](trans1_output, encoder_hidden_states = crossattnupblock2d_encoder)
# # x = crosattnupblock2d.attentions[0].transformer_blocks[0]
# # x.norm1.bias = None
# # x.norm1.weight = None
# # x.norm1.elementwise_affine = False
# # x.norm2.bias = None
# # x.norm2.weight = None
# # x.norm2.elementwise_affine = False
# # x.norm3.bias = None
# # x.norm3.weight = None
# # x.norm3.elementwise_affine = False
# # save_file({"q" : x.attn1.to_q.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn1_q_test.safetensors")
# # save_file({"k" : x.attn1.to_k.weight},  r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn1_k_test.safetensors")
# # save_file({"v" : x.attn1.to_v.weight},  r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn1_v_test.safetensors")
# # save_file({"out" : x.attn1.to_out[0].weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn1_out_w_test.safetensors")
# # save_file({"out" : x.attn1.to_out[0].bias}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn1_out_b_test.safetensors")

# # save_file({"q" : x.attn2.to_q.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn2_q_test.safetensors")
# # save_file({"k" : x.attn2.to_k.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn2_k_test.safetensors")
# # save_file({"v" : x.attn2.to_v.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn2_v_test.safetensors")
# # save_file({"out" : x.attn2.to_out[0].weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn2_out_w_test.safetensors")
# # save_file({"out" : x.attn2.to_out[0].bias}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_attn2_out_b_test.safetensors")

# # for g, y in enumerate(x.ff.net.named_children()):
# #     if y[0] == '0':
# #         save_file({"proj" : y[1].proj.weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_geglu_w_test.safetensors")
# #         save_file({"proj" : y[1].proj.bias}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_geglu_b_test.safetensors")
# #     elif y[0] == '2':
# #         save_file({"ff" : y[1].weight}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_ff_w_test.safetensors")
# #         save_file({"ff" : y[1].bias}, r"C:\study\coursework\src\trash\test_crossattnupblock_hand_ff_b_test.safetensors")





# # hand_output = crosattnupblock2d.attentions[0](hand_output, encoder_hidden_states = crossattnupblock2d_encoder)

# # hand_output = torch.cat([hand_output.to_tuple()[0], res_hidden_big], dim = 1)
# # hand_output = crosattnupblock2d.resnets[1](hand_output, crossattnupblock2d_temb)
# # hand_output = crosattnupblock2d.attentions[1](hand_output, encoder_hidden_states = crossattnupblock2d_encoder)
# # hand_output = torch.cat([hand_output.to_tuple()[0], res_hidden_small], dim = 1)
# # hand_output = crosattnupblock2d.resnets[2](hand_output, crossattnupblock2d_temb)
# # hand_output = crosattnupblock2d.attentions[2](hand_output, encoder_hidden_states = crossattnupblock2d_encoder)
# # hand_output = crosattnupblock2d.upsamplers[0](hand_output.to_tuple()[0])

# # print(hand_output)
# # print(crosattnupblock2d.resnets)

## transformer testings
# trans_input = torch.rand(2, 1280, 4, 4)
# trans_encoder = torch.rand(2, 1280, 2048)
# k = 0
# for x in trans2d.transformer_blocks:
#     print(x.attention_bias, x.attention_head_dim, x.dim, x.double_self_attention, x.num_attention_heads, x.num_positional_embeddings, x.only_cross_attention, x.positional_embeddings, x.pos_embed)
#     save_file({"n" : x.norm1.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm1_w_test.safetensors")
#     save_file({"n" : x.norm1.bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm1_b_test.safetensors")
#     save_file({"n" : x.norm2.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm2_w_test.safetensors")
#     save_file({"n" : x.norm2.bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm2_b_test.safetensors")
#     save_file({"n" : x.norm3.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm3_w_test.safetensors")
#     save_file({"n" : x.norm3.bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_norm3_b_test.safetensors")
#     save_file({"q" : x.attn1.to_q.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn1_q_test.safetensors")
#     save_file({"k" : x.attn1.to_k.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn1_k_test.safetensors")
#     save_file({"v" : x.attn1.to_v.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn1_v_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn1_out_w_test.safetensors")
#     save_file({"out" : x.attn1.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn1_out_b_test.safetensors")
    
#     save_file({"q" : x.attn2.to_q.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn2_q_test.safetensors")
#     save_file({"k" : x.attn2.to_k.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn2_k_test.safetensors")
#     save_file({"v" : x.attn2.to_v.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn2_v_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn2_out_w_test.safetensors")
#     save_file({"out" : x.attn2.to_out[0].bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_attn2_out_b_test.safetensors")

#     for i, y in enumerate(x.ff.net.named_children()):
#         if y[0] == '0':
#             save_file({"proj" : y[1].proj.weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_geglu_w_test.safetensors")
#             save_file({"proj" : y[1].proj.bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_geglu_b_test.safetensors")
#         elif y[0] == '2':
#             save_file({"ff" : y[1].weight}, fr"C:\study\coursework\src\trash\test_trans_{k}_ff_w_test.safetensors")
#             save_file({"ff" : y[1].bias}, fr"C:\study\coursework\src\trash\test_trans_{k}_ff_b_test.safetensors")
#     k += 1

# save_file({"inp" : trans_input}, r"C:\study\coursework\src\trash\test_trans_test.safetensors")    
# save_file({"inp" : trans_encoder}, r"C:\study\coursework\src\trash\test_trans_encoder_test.safetensors")    
# save_file({"in" : trans2d.proj_in.weight} , r"C:\study\coursework\src\trash\test_trans_projin_w_test.safetensors")
# save_file({"in" : trans2d.proj_in.bias} , r"C:\study\coursework\src\trash\test_trans_projin_b_test.safetensors")
# save_file({"out" : trans2d.proj_out.weight} , r"C:\study\coursework\src\trash\test_trans_projout_w_test.safetensors")
# save_file({"out" : trans2d.proj_out.bias} , r"C:\study\coursework\src\trash\test_trans_projout_b_test.safetensors")
# save_file({"in" : trans2d.norm.weight} , r"C:\study\coursework\src\trash\test_trans_norm_w_test.safetensors")
# save_file({"in" : trans2d.norm.bias} , r"C:\study\coursework\src\trash\test_trans_norm_b_test.safetensors")
# # print(trans2d.proj_out.weight)
# # print(trans2d.transformer_blocks[0].ff)
# # hand_output = trans2d.norm(trans_input)

# # hand_output = hand_output.permute(0, 2, 3, 1).reshape(2, 4 * 4, 1280)
# # hand_output = trans2d.proj_in(hand_output)
# # print(hand_output)
# # k = 0
# # for x in trans2d.transformer_blocks:
# #     hand_output = x(hand_output, encoder_hidden_states = trans_encoder)

# #     # if k == 9:
# #     #     print(hand_output)

# #     k+=1

# # hand_output = trans2d.proj_out(hand_output)
# # # print(hand_output)

# # # hand_output = (
# # #     hand_output.reshape(2, 4, 4, 1280).permute(0, 3, 1, 2).contiguous()
# # # )
# # # hand_output = hand_output + trans_input
# # output = trans2d(trans_input, encoder_hidden_states = trans_encoder)
# # save_file({"output": output.to_tuple()[0]}, r"C:\study\coursework\src\trash\test_trans_output2_test.safetensors")
# # print(output.to_tuple()[0])\

# trans2d_large_input = torch.rand(2, 1280, 64, 20)
# save_file({"input_large" : trans2d_large_input}, r"C:\study\coursework\src\trash\test_trans_large_input_test.safetensors")
# large_output = trans2d(trans2d_large_input, encoder_hidden_states = trans_encoder)
# save_file({"output": large_output.to_tuple()[0]}, r"C:\study\coursework\src\trash\test_trans_large_output_test.safetensors")
# print(trans2d, large_output.to_tuple()[0])


# hand_output = trans2d.norm(trans2d_large_input)
# hand_output = hand_output.permute(0, 2, 3, 1).reshape(2, 64*20, 1280)
# hand_output = trans2d.proj_in(hand_output)
# # print(hand_output)
# k = 0
# for x in trans2d.transformer_blocks:
#     hand_output = x(hand_output, encoder_hidden_states = trans_encoder)
#     # if k == 9:
#     #     print(hand_output)

#     k+=1

# hand_output = trans2d.proj_out(hand_output)
# # print(hand_output)

# hand_output = (
#     hand_output.reshape(2, 64, 20, 1280).permute(0, 3, 1, 2).contiguous()
# )
# hand_output = hand_output + trans2d_large_input

# output = trans2d(trans_input, encoder_hidden_states = trans_encoder)
# save_file({"output": output.to_tuple()[0]}, r"C:\study\coursework\src\trash\test_trans_output2_test.safetensors")
# print(output.to_tuple()[0])\

























# # print(hand_output)
# # print(output.to_tuple()[0].shape)
# # # print(torch.allclose(hand_output, output.to_tuple()[0], 1e-67))
# # rust_output = load_file(r"C:\study\coursework\src\trash\test_trans_rust_test.safetensors")['output_tensor']
# # print(rust_output.shape)
# # grid = vutils.make_grid(rust_output[:, :3], nrow=8, normalize=True)
# # plt.figure(figsize=(20, 20))
# # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# # plt.axis('off')
# # plt.show()


## Basic transformer block testings
# crossattnupblock1 = crossattnupblock_list[0]
# btbsup = []
# for i, trans in enumerate(crossattnupblock1.attentions.named_children()):
#     if trans[0] == '1':
#         for j in trans[1].transformer_blocks:
#             btbsup.append(j)
    # for x in trans[1].transformer_blocks:
    #     print(x.dim, x.attention_head_dim, x.num_attention_heads, x.attention_bias, x.positional_embeddings, x.num_positional_embeddings, x._chunk_size, x.only_cross_attention )

# btb1_test = torch.rand(2, 1280, 1280)
# btb1_encoder = torch.rand(2, 1280, 2048)
# btb1 = btbsup[0]

# btb1.norm1.elementwise_affine= False
# btb1.norm1.weight = None
# btb1.norm1.bias = None
# btb1.norm2.elementwise_affine= False
# btb1.norm2.weight = None
# btb1.norm2.bias = None
# btb1.norm3.elementwise_affine= False
# btb1.norm3.weight = None
# btb1.norm3.bias = None

# save_file({"inp" : btb1_test}, r"C:\study\coursework\src\trash\test_btb1_test.safetensors")
# save_file({"inp": btb1_encoder}, r"C:\study\coursework\src\trash\test_btb1_encoder.safetensors")
# save_file({"n" : btb1.norm1.weight}, r"C:\study\coursework\src\trash\test_btb1_norm1_w_test.safetensors")
# save_file({"n" : btb1.norm1.bias}, r"C:\study\coursework\src\trash\test_btb1_norm1_b_test.safetensors")
# save_file({"n" : btb1.norm2.weight}, r"C:\study\coursework\src\trash\test_btb1_norm2_w_test.safetensors")
# save_file({"n" : btb1.norm2.bias}, r"C:\study\coursework\src\trash\test_btb1_norm2_b_test.safetensors")
# save_file({"n" : btb1.norm3.weight}, r"C:\study\coursework\src\trash\test_btb1_norm3_w_test.safetensors")
# save_file({"n" : btb1.norm3.bias}, r"C:\study\coursework\src\trash\test_btb1_norm3_b_test.safetensors")

# save_file({"q" : btb1.attn1.to_q.weight}, r"C:\study\coursework\src\trash\test_btb1_attn1_q_test.safetensors")
# save_file({"k" : btb1.attn1.to_k.weight}, r"C:\study\coursework\src\trash\test_btb1_attn1_k_test.safetensors")
# save_file({"v" : btb1.attn1.to_v.weight}, r"C:\study\coursework\src\trash\test_btb1_attn1_v_test.safetensors")
# save_file({"out" : btb1.attn1.to_out[0].weight}, r"C:\study\coursework\src\trash\test_btb1_attn1_out_w_test.safetensors")
# save_file({"out" : btb1.attn1.to_out[0].bias}, r"C:\study\coursework\src\trash\test_btb1_attn1_out_b_test.safetensors")

# save_file({"q" : btb1.attn2.to_q.weight}, r"C:\study\coursework\src\trash\test_btb1_attn2_q_test.safetensors")
# save_file({"k" : btb1.attn2.to_k.weight}, r"C:\study\coursework\src\trash\test_btb1_attn2_k_test.safetensors")
# save_file({"v" : btb1.attn2.to_v.weight}, r"C:\study\coursework\src\trash\test_btb1_attn2_v_test.safetensors")
# save_file({"out" : btb1.attn2.to_out[0].weight}, r"C:\study\coursework\src\trash\test_btb1_attn2_out_w_test.safetensors")
# save_file({"out" : btb1.attn2.to_out[0].bias}, r"C:\study\coursework\src\trash\test_btb1_attn2_out_b_test.safetensors")

# for i, x in enumerate(btb1.ff.net.named_children()):
#     if x[0] == '0':
#         print(x[1].proj)
#         save_file({"proj" : x[1].proj.weight}, r"C:\study\coursework\src\trash\test_btb1_geglu_w_test.safetensors")
#         save_file({"proj" : x[1].proj.bias}, r"C:\study\coursework\src\trash\test_btb1_geglu_b_test.safetensors")
#     elif x[0] == '2':
#         print(x[1])
#         save_file({"ff" : x[1].weight}, r"C:\study\coursework\src\trash\test_btb1_ff_w_test.safetensors")
#         save_file({"ff" : x[1].bias}, r"C:\study\coursework\src\trash\test_btb1_ff_b_test.safetensors")

# # output = btb1(btb1_test, encoder_hidden_states=btb1_encoder) 
# # print(output)
# # save_file({"outp": output}, r"C:\study\coursework\src\trash\test_btb1_output_test.safetensors")
# norm = btb1.norm1(btb1_test)
# # save_file({"norm1" : norm}, r"C:\study\coursework\src\trash\test_btb1_piece.safetensors")
# # rust_norm1 = load_file(r"C:\study\coursework\src\trash\test_btb1_rust_norm1.safetensors")['output_tensor']
# # # print(torch.allclose(rust_norm1, norm, 1e-01))
# # print(torch.max(norm - rust_norm1))
# # print(torch.allclose(norm, rust_norm1, rtol = 1e-03, atol = 1e-03))
# norm_attn = btb1.attn1(norm)
# # save_file({"norm1_attn" : norm_attn}, r"C:\study\coursework\src\trash\test_btb1_piece2.safetensors")
# # rust_norm1_attn1 = load_file(r"C:\study\coursework\src\trash\test_btb1_rust_norm1_attn1.safetensors")['output_tensor']
# # print(torch.allclose(norm_attn, rust_norm1_attn1, rtol = 1e-03, atol = 1e-02))

# # print(norm_attn, norm_attn.shape)
# s = norm_attn + btb1_test
# # save_file({"residual1": s}, r"C:\study\coursework\src\trash\test_btb1_piece3.safetensors")
# # print(s)
# # norm2 = btb1.norm2(s)

# # norm2_attn2 = btb1.attn2(norm2, btb1_encoder)

# # s = norm2_attn2 + s

# # norm3 = btb1.norm3(s)

# # ff = btb1.ff(norm3)

# # s = ff + s
# # print(s)

# btb1_bchw = torch.rand(2, 1280, 2, 1280)
# save_file({"bchw" : btb1_bchw}, r"C:\study\coursework\src\trash\test_btb1_bchw_test.safetensors")
# output_bchw = btb1(btb1_bchw, encoder_hidden_states=btb1_encoder)
# save_file({"bchw" : output_bchw}, r"C:\study\coursework\src\trash\test_btb1_bchw_output.safetensors")
# print(output_bchw.shape, btb1, output_bchw)





## attn testings
# attn1_test = torch.rand(2, 1280, 1280)
# # # output = attn1(attn1_test)
# # # print(output, output.shape)
# save_file({"attn1_test" : attn1_test}, r"C:\study\coursework\src\trash\test_attn1_test.safetensors")
# print(attn1)
# save_file({"q" : attn1.to_q.weight}, r"C:\study\coursework\src\trash\test_attn1_q_test.safetensors")
# save_file({"k" : attn1.to_k.weight}, r"C:\study\coursework\src\trash\test_attn1_k_test.safetensors")
# save_file({"v" : attn1.to_v.weight}, r"C:\study\coursework\src\trash\test_attn1_v_test.safetensors")
# save_file({"out" : attn1.to_out[0].weight}, r"C:\study\coursework\src\trash\test_attn1_out_w_test.safetensors")
# save_file({"out" : attn1.to_out[0].bias}, r"C:\study\coursework\src\trash\test_attn1_out_b_test.safetensors")
# # # # # save_file({"outp" : attn1(attn1_test)}, r"C:\study\coursework\src\trash\test_attn1_output_test.safetensors")
# attn1_test_bchw = torch.rand(2, 1280, 16, 128)
# save_file({"attn1_test" : attn1_test_bchw}, r"C:\study\coursework\src\trash\test_attn1_test_bchw.safetensors")
# save_file({"outp" : attn1(attn1_test_bchw).contiguous()}, r"C:\study\coursework\src\trash\test_attn1_output_bchw_test.safetensors")
# encoder_tensor = torch.rand(2, 1280, 1280)
# save_file({"attn1_test" : encoder_tensor}, r"C:\study\coursework\src\trash\test_attn1_encoder_test.safetensors")
# # save_file({"outp" : attn1(attn1_test, encoder_hidden_states = encoder_tensor)}, r"C:\study\coursework\src\trash\test_attn1_output_encoder_test.safetensors")
# save_file({"outp" : attn1(attn1_test_bchw, encoder_hidden_states = encoder_tensor).contiguous()}, r"C:\study\coursework\src\trash\test_attn1_output_bchw_encoder_test.safetensors")
# print(attn1(attn1_test_bchw).contiguous())
# print(attn1.heads)
## crossattnupbloc testings
# print(crossattnupblock_list)







## downblock testings
# downblock2d_resnet_list = []
# for i, resnet in enumerate(downblock2d.resnets):
#     downblock2d_resnet_list.append(resnet)
# for j, block in enumerate(downblock2d.named_children()):
#     if j == 1:
#         downsample = block[1]


# # # print(downblock2d_resnet_list)


# temb = torch.rand(2, 1280)
# save_file({"temb" : temb}, r"C:\study\coursework\src\trash\test_downblock2d_temb.safetensors")
# downblock2d_test = torch.rand(2, 320, 128, 128)
# save_file({"test" : downblock2d_test}, r"C:\study\coursework\src\trash\test_downblock2d_test.safetensors")



# resnet_1 = downblock2d_resnet_list[0]

# save_file({"conv1_weight" : resnet_1.conv1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_weight.safetensors")
# save_file({"conv1_weight" : resnet_1.conv1.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_bias.safetensors")
# save_file({"conv2_weight" : resnet_1.conv2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_weight.safetensors")
# save_file({"conv1_weight" : resnet_1.conv2.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_bias.safetensors")

# save_file({"conv1_weight" : resnet_1.norm1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_norm1_weight.safetensors")
# save_file({"conv1_weight" : resnet_1.norm1.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_norm1_bias.safetensors")
# save_file({"conv2_weight" : resnet_1.norm2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_norm2_weight.safetensors")
# save_file({"conv1_weight" : resnet_1.norm2.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_norm2_bias.safetensors")


# save_file({"linear_proj" : resnet_1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_bias.safetensors")

# resnet_2 = downblock2d_resnet_list[1]

# save_file({"conv1_weight" : resnet_2.conv1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_weight.safetensors")
# save_file({"conv1_weight" : resnet_2.conv1.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_bias.safetensors")
# save_file({"conv2_weight" : resnet_2.conv2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_weight.safetensors")
# save_file({"conv1_weight" : resnet_2.conv2.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_bias.safetensors")

# save_file({"conv1_weight" : resnet_2.norm1.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_norm1_weight.safetensors")
# save_file({"conv1_weight" : resnet_2.norm1.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_norm1_bias.safetensors")
# save_file({"conv2_weight" : resnet_2.norm2.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_norm2_weight.safetensors")
# save_file({"conv1_weight" : resnet_2.norm2.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_norm2_bias.safetensors")


# save_file({"linear_proj" : resnet_2.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_2.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_bias.safetensors")

# for i, block in enumerate(downsample.named_children()):
#     conv_downsample = block[1]

# save_file({"conv_down_weight": conv_downsample.conv.weight}, r"C:\study\coursework\src\trash\test_downblock2d_downsample.safetensors")
# save_file({"conv_down_weight": conv_downsample.conv.bias}, r"C:\study\coursework\src\trash\test_downblock2d_downsample_b.safetensors")
# output = downblock2d(downblock2d_test, temb = temb)
# save_file({"downsample2d_out": output[0]}, r"C:\study\coursework\src\trash\test_downsample2d_output.safetensors")

# save_file({"downsample2d_out_hidden" : output[1][0]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden1.safetensors")
# save_file({"downsample2d_out_hidden" : output[1][1]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden2.safetensors")
# save_file({"downsample2d_out_hidden" : output[1][2]}, r"C:\study\coursework\src\trash\test_downsample2d_output_hidden3.safetensors")
# print(output[1][0])













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
# save_file({"conv1_weight" : resnet_1.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_weight.safetensors")
# save_file({"conv1_weight" : resnet_1.conv1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_bias.safetensors")
# save_file({"conv2_weight" : resnet_1.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_weight.safetensors")
# save_file({"conv2_weight" : resnet_1.conv2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_bias.safetensors")
# save_file({"norm1_weight" : resnet_1.norm1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_norm1_weight.safetensors")
# save_file({"norm1_weight" : resnet_1.norm1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_norm1_bias.safetensors")
# save_file({"norm2_weight" : resnet_1.norm2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_norm2_weight.safetensors")
# save_file({"norm2_weight" : resnet_1.norm2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_norm2_bias.safetensors")
# save_file({"linear_proj" : resnet_1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_1.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_weight.safetensors")
# save_file({"conv_short_weight" : resnet_1.conv_shortcut.bias},   r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_bias.safetensors")

# resnet_2 = upblock2d_resnet_list[1]
# save_file({"conv1_weight" : resnet_2.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_weight.safetensors")
# save_file({"conv1_weight" : resnet_2.conv1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_bias.safetensors")
# save_file({"conv2_weight" : resnet_2.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_weight.safetensors")
# save_file({"conv2_weight" : resnet_2.conv2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_bias.safetensors")
# save_file({"norm1_weight" : resnet_2.norm1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_norm1_weight.safetensors")
# save_file({"norm1_weight" : resnet_2.norm1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_norm1_bias.safetensors")
# save_file({"norm2_weight" : resnet_2.norm2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_norm2_weight.safetensors")
# save_file({"norm2_weight" : resnet_2.norm2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_norm2_bias.safetensors")
# save_file({"linear_proj" : resnet_2.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_2.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_2.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_weight.safetensors")
# save_file({"conv_short_weight" : resnet_2.conv_shortcut.bias},   r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_bias.safetensors")


# resnet_3 = upblock2d_resnet_list[2]
# save_file({"conv1_weight" : resnet_3.conv1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_weight.safetensors")
# save_file({"conv1_weight" : resnet_3.conv1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_bias.safetensors")
# save_file({"conv2_weight" : resnet_3.conv2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_weight.safetensors")
# save_file({"conv2_weight" : resnet_3.conv2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_bias.safetensors")
# save_file({"norm1_weight" : resnet_3.norm1.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_norm1_weight.safetensors")
# save_file({"norm1_weight" : resnet_3.norm1.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_norm1_bias.safetensors")
# save_file({"norm2_weight" : resnet_3.norm2.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_norm2_weight.safetensors")
# save_file({"norm2_weight" : resnet_3.norm2.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_norm2_bias.safetensors")
# save_file({"linear_proj" : resnet_3.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_weight.safetensors")
# save_file({"linear_proj" : resnet_3.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_bias.safetensors")
# save_file({"conv_short_weight" : resnet_3.conv_shortcut.weight},   r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_weight.safetensors")
# save_file({"conv_short_weight" : resnet_3.conv_shortcut.bias},   r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_bias.safetensors")


# res_hidden_states_tuple = (res_hidden, res_hidden, res_hidden)
# temb = torch.rand(2, 1280)
# save_file({"temb" : temb}, r"C:\study\coursework\src\trash\test_upblock2d_temb.safetensors")
# save_file({"upblock2d_out": upblock2d(upblock2d_test, res_hidden_states_tuple, temb = temb)}, r"C:\study\coursework\src\trash\test_upblock2d_out.safetensors")
# print(upblock2d, upblock2d(upblock2d_test, res_hidden_states_tuple, temb = temb))


## downsample2d testings
## they share common input 
# test_upsample = torch.rand(2, 640, 128, 128)
# save_file({"downsample_in": test_upsample}, r"C:\study\coursework\src\trash\test_downsample_inp.safetensors")
# downsample2d_test = downsample2d_list[1]

# save_file({"downsample2d_conv" : downsample2d_test.conv.weight}, r"C:\study\coursework\src\trash\test_downsample_conv.safetensors")
# save_file({"downsample2d_conv" : downsample2d_test.conv.bias}, r"C:\study\coursework\src\trash\test_downsample_conv_b.safetensors")
# save_file({"downsample_out": downsample2d_test(test_upsample)}, r"C:\study\coursework\src\trash\test_downsample_outp.safetensors")
# print(downsample2d)
## upsample test
# test_upsample = torch.rand(2, 640, 128, 128)
# upsample2d_output = upsample2d(test_upsample)
# print(upsample2d)
# save_file({"upsample2d_conv" : upsample2d.conv.weight}, r"C:\study\coursework\src\trash\test_upsample_conv.safetensors")
# save_file({"upsample2d_conv" : upsample2d.conv.bias}, r"C:\study\coursework\src\trash\test_upsample_b_conv.safetensors")
# save_file({"upsample2d_input" : test_upsample}, r"C:\study\coursework\src\trash\test_upsample_inp.safetensors")
# save_file({"upsample2d_output" : upsample2d_output}, r"C:\study\coursework\src\trash\test_upsample_outp.safetensors")
# print(upsample2d.conv.bias.shape)


## resnet
# print(resnet_list[0])
# print(resnet_list[0].forward(test_image, temb))
## no bias no shortcut
# testings1 = resnet_list[0]
# # print(testings1.norm1.bias, testings1.nonlinearity, testings1.conv1, testings1.norm2, testings1.nonlinearity, testings1.conv2, testings1.conv_shortcut)
# test_image = torch.rand(2, 320, 320, 320)
# temb = torch.rand(2, 1280)
# output = testings1(test_image, temb)
# # for i, layer in enumerate(testings1.named_children):
# #     print('\n\n\nThis is layer {layer}', layer)
# save_file({"test_image": test_image}, r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors")
# save_file({"test_image": temb}, r"C:\study\coursework\src\trash\test_resnet_temb.safetensors")
# save_file({"conv1_weight" : testings1.conv1.weight},  r"C:\study\coursework\src\trash\test_resnet_conv1_weight.safetensors")
# save_file({"conv1_weight" : testings1.conv1.bias},  r"C:\study\coursework\src\trash\test_resnet_conv1_bias.safetensors")
# save_file({"norm1" : testings1.norm1.weight},  r"C:\study\coursework\src\trash\test_resnet_norm1_weight.safetensors")
# save_file({"norm1" : testings1.norm1.bias},  r"C:\study\coursework\src\trash\test_resnet_norm1_bias.safetensors")
# save_file({"norm2" : testings1.norm2.weight},  r"C:\study\coursework\src\trash\test_resnet_norm2_weight.safetensors")
# save_file({"norm2" : testings1.norm2.bias},  r"C:\study\coursework\src\trash\test_resnet_norm2_bias.safetensors")
# # print(testings1.conv1.weight.shape)
# save_file({"conv2_weight" : testings1.conv2.weight},  r"C:\study\coursework\src\trash\test_resnet_conv2_weight.safetensors")
# save_file({"conv2_weight" : testings1.conv2.bias},  r"C:\study\coursework\src\trash\test_resnet_conv2_bias.safetensors")
# save_file({"linear_proj" : testings1.time_emb_proj.weight},  r"C:\study\coursework\src\trash\test_resnet_linear_weight.safetensors")
# save_file({"linear_proj" : testings1.time_emb_proj.bias},  r"C:\study\coursework\src\trash\test_resnet_linear_bias.safetensors")
# print(testings1)
# print(testings1.time_emb_proj, testings1.time_emb_proj.weight, testings1.time_emb_proj.bias)
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
# hand_output = conv2(drop(silu(norm2(conv1(silu(norm1(test_image))) + temb_act[:, :, None, None]))))
# print(hand_output)
# print(output)
# save_file({"resnet_no_shortcut" : output}, r"C:\study\coursework\src\trash\test_resnet_output.safetensors")
## no bias shorcut
# testings2 = resnet_list[2]
# test_image = torch.rand(2, 320, 320, 320)
# temb = torch.rand(2, 1280)
# save_file({"test_image": test_image}, r"C:\study\coursework\src\trash\test_resnet_short_test_image.safetensors")
# save_file({"test_image": temb}, r"C:\study\coursework\src\trash\test_resnet_short_temb.safetensors")
# save_file({"conv1_weight" : testings2.conv1.weight},  r"C:\study\coursework\src\trash\test_resnet_short_conv1_weight.safetensors")
# save_file({"conv1_weight" : testings2.conv1.bias},  r"C:\study\coursework\src\trash\test_resnet_short_conv1_bias.safetensors")
# save_file({"conv1_weight" : testings2.conv2.weight},  r"C:\study\coursework\src\trash\test_resnet_short_conv2_weight.safetensors")
# save_file({"conv1_weight" : testings2.conv2.bias},  r"C:\study\coursework\src\trash\test_resnet_short_conv2_bias.safetensors")
# save_file({"norm1" : testings2.norm1.weight},  r"C:\study\coursework\src\trash\test_resnet_short_norm1_weight.safetensors")
# save_file({"norm1" : testings2.norm1.bias},  r"C:\study\coursework\src\trash\test_resnet_short_norm1_bias.safetensors")
# save_file({"norm2" : testings2.norm2.weight},  r"C:\study\coursework\src\trash\test_resnet_short_norm2_weight.safetensors")
# save_file({"norm2" : testings2.norm2.bias},  r"C:\study\coursework\src\trash\test_resnet_short_norm2_bias.safetensors")
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
# save_file({"conv_short_weight" : testings2.conv_shortcut.bias},   r"C:\study\coursework\src\trash\test_resnet_short_conv_short_bias.safetensors")
# save_file({"resnet_shortcut": testings2(test_image, temb)},  r"C:\study\coursework\src\trash\test_resnet_short_output.safetensors")
# print(testings2)
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