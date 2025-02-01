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
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_inp.safetensors".to_string())?;
    print!("{:?}\n\n\n{:?}", test_vec_shape, test_vec);
    let _ = output(r"C:\study\coursework\src\trash\test_inp_rust.safetensors".to_string(), test_vec.to_vec(), test_vec_shape.to_vec());
    Ok(())
}
