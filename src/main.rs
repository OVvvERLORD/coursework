use cudarc::{self};
use core::{f32};

mod func;
mod layers;
mod blocks;
mod main_parts;

use crate::{
    func::functions::{Tensor_Mul, input, nearest_neighbour_interpolation, output}, 
    layers::{
        params::{Transformer2D_params, BasicTransofmerBlock_params, Resnet2d_params, CrossAttnUpBlock2D_params, CrossAttnDownBlock2D_params},
        layer::Layer,
        act::{SiLU, GeLU},
        norm::{GroupNorm, LayerNorm},
        linear::Linear,
        conv::Conv2d,
        upsample::Upsample2D,
        downsample::DownSample2D
    },
    blocks::{
        resnet::Resnet2d,
        ff::FeedForward,
        attn::{
            Attention,
            CrossAttnUpBlock2D,
            CrossAttnDownBlock2D
        },
        btb::BasicTransofmerBlock,
        trans::Transformer2D,
        upblock::UpBlock2d,
        mid::mid_block,
        up::Up_blocks,
        downblock::DownBlock2D,
        down::Down_blocks
    },
    main_parts::unet
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_inp.safetensors".to_string()).unwrap();
    print!("{:?}", test_vec_shape);
    let (weight_vec, weight_vec_shape) = input(r"C:\study\coursework\src\trash\test_conv_weight.safetensors".to_string()).unwrap();
    let conv = Conv2d {kernel_size: 1, in_channels: 640, out_channels: 320, padding: 0, stride: 1, kernel_weights: weight_vec.to_vec()};
    let (res_vec, res_vec_shape) = conv.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    for i in 0..128{
        print!("{:?} ", res_vec[i]);
    }
    print!("\n{:?}", res_vec_shape);
    Ok(())
}
