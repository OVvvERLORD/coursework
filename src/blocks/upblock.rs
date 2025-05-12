use crate::{
    layers::{
        params::Resnet2d_params,
        layer::Layer
    },
    blocks::resnet::Resnet2d,
    func::functions::input
};

use std::rc::Rc;
use std::cell::RefCell;

pub struct UpBlock2d {
    pub operations : Vec<Box<dyn Layer>>,
    pub res_hidden_tensors: Rc<RefCell<Vec<ndarray::Array4<f32>>>>,
}

impl UpBlock2d {
    pub fn new(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        params_for_resnet3 : Resnet2d_params,
        hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::new(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::new(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let resnet3 = Resnet2d::new(params_for_resnet3);
        vec.push(Box::new(resnet3));
        Self { operations: vec, res_hidden_tensors: hidden_states}
    }
}

impl Layer for UpBlock2d {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut hidden_tensors = self.res_hidden_tensors.borrow_mut();
        let mut idx = &hidden_tensors.len() - 1;
        for layer in operations {
            let res_hidden_tensor= hidden_tensors.pop().unwrap();
            *args = ndarray::concatenate(ndarray::Axis(1), &[args.view(), res_hidden_tensor.view()])
            .unwrap()
            .as_standard_layout()
            .to_owned();
            let _ = layer.operation(args)?;
            idx = if idx > 0 {idx - 1} else {0};
        } 
        Ok(())
    }
}

#[test]
fn test_unbiased () {
    let mut hidden_vec = Vec::<ndarray::Array4<f32>>::new();
    let temb = input(r"C:\study\coursework\src\trash\test_upblock2d_temb.safetensors".to_string()).unwrap();
    let hidden_tensor = input(r"C:\study\coursework\src\trash\test_upblock2d_res_hidden.safetensors".to_string()).unwrap();
    let mut tensor = input(r"C:\study\coursework\src\trash\test_upblock2d_test.safetensors".to_string()).unwrap();
    let temb = Rc::new(RefCell::new(temb));
    hidden_vec.push(hidden_tensor.clone());
    hidden_vec.push(hidden_tensor.clone());
    hidden_vec.push(hidden_tensor);
    let hidden_vec = Rc::new(RefCell::new(hidden_vec));
    
    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv1_bias.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv2_bias.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_norm1_bias.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_norm2_bias.safetensors".to_string()).unwrap();  
    let kernels = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res1_conv_short_bias.safetensors".to_string()).unwrap();
    let params1 = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 960, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: true,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
        bias_s: cs_b, is_bias_s: true,
        time_emb:Rc::clone(&temb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv1_bias.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv2_bias.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_linear_bias.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_norm1_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_norm1_bias.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_norm2_weight.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_norm2_bias.safetensors".to_string()).unwrap();  
    let kernels = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res2_conv_short_bias.safetensors".to_string()).unwrap();
    let params2 = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 640, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: true,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
        bias_s: cs_b, is_bias_s: true,
        time_emb:Rc::clone(&temb)
    };


    let kernel1 = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv1_bias.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_weight.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv2_bias.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_linear_bias.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_norm1_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_norm1_bias.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_norm2_weight.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_norm2_bias.safetensors".to_string()).unwrap();  
    let kernels = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_upblock2d_res3_conv_short_bias.safetensors".to_string()).unwrap();
    let params3 = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 640, 
        out_channels_1: 320, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 320, 
        out_channels_2: 320, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: true,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
        bias_s: cs_b, is_bias_s: true,
        time_emb:Rc::clone(&temb)
    };
    let upblock = UpBlock2d::new(
        params1,
        params2,
        params3,
        hidden_vec
    );
    let _ = upblock.operation(&mut tensor).unwrap();
    let shape = tensor.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_upblock2d_out.safetensors")).unwrap();

    assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-02);
                }
            }
        }
    }

}