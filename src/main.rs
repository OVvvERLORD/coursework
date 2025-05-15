use std::{rc::Rc, sync::mpsc::Receiver};
use std::cell::{Ref, RefCell};
use ndarray_einsum::einsum;
use core::f32;
mod func;
mod layers;
mod blocks;
mod main_parts;

// mod main_parts;
use crate::{
    func::functions::{Tensor_Mul, input, nearest_neighbour_interpolation, output, scalar_timestep_embedding}, 
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
    main_parts::unet::Unet2dConditionModel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder_cross = input(format!(r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")).unwrap();
    let encoder_cross = Rc::new(RefCell::new(encoder_cross.remove_axis(ndarray::Axis(0))));
    let time_emb = Rc::new(RefCell::new(ndarray::Array4::from_elem([1, 1, 1, 1], 1.)));

    let res_hidden = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));

    let mut unet = Unet2dConditionModel::new_weights_bias(encoder_cross, time_emb, res_hidden);
    let add_time_ids = input(format!(r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")).unwrap();
    let add_text_embs = input(format!(r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")).unwrap();
    let kwargs = Rc::new(RefCell::new((add_time_ids, add_text_embs)));
    let mut tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_input.safetensors")).unwrap();
    let _ = unet.operation(&mut tensor, 1., kwargs).unwrap();
    print!("{:?}", tensor);
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_unet_output.safetensors")).unwrap();
    let shape = tensor.shape();
    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-4);
                }
            }
        }
    }
    Ok(())
}
