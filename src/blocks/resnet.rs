use std::rc::Rc;
use std::cell::RefCell;

use crate::layers::{
    params::Resnet2d_params,
    norm::GroupNorm,
    act::SiLU,
    linear::Linear,
    conv::Conv2d,
    layer::Layer
};
use crate::func::functions::{input};

pub struct Resnet2d {
    pub if_shortcut:bool,
    pub operations: Vec<Box<dyn Layer>>,
    pub time_emb : Rc<RefCell<ndarray::Array4<f32>>>
}

impl Resnet2d {
    pub fn new (
        params : Resnet2d_params
        ) -> Self {
            let mut layer_vec : Vec<Box<dyn Layer>> = Vec::new();
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_1, eps : params.eps_1, gamma : params.gamma_1, beta : params.beta_1}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {
                in_channels : params.in_channels_1, 
                out_channels : params.out_channels_1, 
                padding : params.padding_1, 
                stride: params.stride_2, 
                kernel_size : params.kernel_size_1, 
                kernel_weights : params.kernel_weights_1,
                bias: params.bias_c1, is_bias: params.is_bias_c1
            }));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Linear{weights : params.weights, bias : params.bias, is_bias : params.is_bias}));
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_2, eps : params.eps_2, gamma : params.gamma_2, beta : params.beta_2}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {
                in_channels : params.in_channels_2, 
                out_channels : params.out_channels_2, 
                padding : params.padding_2, 
                stride: params.stride_2, 
                kernel_size : params.kernel_size_2, 
                kernel_weights : params.kernel_weights_2,
                bias: params.bias_c2,
                is_bias: params.is_bias_c2
            }));
            if params.is_shortcut {
                layer_vec.push(Box::new(Conv2d {
                    in_channels : params.in_channels_short, 
                    out_channels : params.out_channels_short, 
                    stride: params.stride_short, 
                    padding : params.padding_short, 
                    kernel_size : params.kernel_size_short, 
                    kernel_weights : params.kernel_weights_short,
                    bias: params.bias_s,
                    is_bias: params.is_bias_s
                }));
            }
            Self { if_shortcut: params.is_shortcut, operations: layer_vec, time_emb : params.time_emb}
    }   
}

impl Layer for Resnet2d {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let mut initial_tensor = args.clone();
        for i in 0..self.operations.len()-(self.if_shortcut as usize) {
            if i == 3 {
                let mut time_emb = self.time_emb.borrow().to_owned();
                let _ = self.operations[i].operation(&mut time_emb).unwrap(); // act
                let _ = self.operations[i + 1].operation(&mut time_emb).unwrap(); // lin
                // let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
                // let time_tensor = ndarray::Array4::from_shape_vec((lin_res.1[2], lin_res.1[3], 1, 1), lin_res.0)?;
                // curr_tensor = curr_tensor.clone() + time_tensor.broadcast(curr_tensor.dim()).unwrap();
                let shape = time_emb.dim();
                time_emb = time_emb.into_shape_with_order((shape.2, shape.3, 1, 1)).unwrap().to_owned();
                let target = args.shape();
                time_emb = time_emb.broadcast((target[0], target[1], target[2], target[3])).unwrap().to_owned();
                *args += &time_emb;
                // res_vec = curr_tensor.into_raw_vec_and_offset().0;
                continue;
            }
            if i == 4{
                continue;
            }
            let _ = self.operations[i].operation(args)?;
            // res_vec = res.0.clone();
            // res_shape_vec = res.1.clone();
        }
        if self.if_shortcut {
            let _ = self.operations[self.operations.len() - 1].operation(&mut initial_tensor)?;
            // let shortcut_vec = shortcut_res.0;
            // let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
            // let short_tensor = ndarray::Array4::from_shape_vec((shortcut_res.1[0], shortcut_res.1[1], shortcut_res.1[2], shortcut_res.1[3]), shortcut_vec.clone())?;
            // curr_tensor = curr_tensor.clone() + short_tensor.broadcast(curr_tensor.dim()).unwrap();
            *args += &initial_tensor.broadcast(args.shape()).unwrap();
            // res_vec = curr_tensor.into_raw_vec_and_offset().0;
        } else {
            // let mut curr_tensor =  ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
            // let input_tensor =  ndarray::Array4::from_shape_vec((args.1[0],args.1[1], args.1[2], args.1[3]), args.0)?;
            // curr_tensor = curr_tensor + input_tensor;
            // res_vec = curr_tensor.into_raw_vec_and_offset().0;
            *args += &initial_tensor;
        }
        Ok(())
    }
}

#[test]
fn test_resnet_no_shortcut_no_bias() {
    let kernel1 = input(r"C:\study\coursework\src\trash\test_resnet_conv1_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_resnet_conv1_bias.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_resnet_conv2_weight.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_resnet_conv2_bias.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_resnet_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_resnet_linear_bias.safetensors".to_string()).unwrap();
    let mut tensor = input(r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors".to_string()).unwrap();
    let temb = input(r"C:\study\coursework\src\trash\test_resnet_temb.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_resnet_norm1_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_resnet_norm1_bias.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_resnet_norm2_weight.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_resnet_norm2_bias.safetensors".to_string()).unwrap();  
    let params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
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
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem([1, 1, 1, 1], 1.), is_bias_s: true,
        time_emb: Rc::new(RefCell::new(temb))
    };
    let resnet = Resnet2d::new(params);
    let _ = resnet.operation(&mut tensor);
    let shape = tensor.shape();
    let py_tensor = input(r"C:\study\coursework\src\trash\test_resnet_output.safetensors".to_string()).unwrap();
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

#[test]
fn test_resnet_shortcut_no_bias() {
    let kernel1 = input(r"C:\study\coursework\src\trash\test_resnet_short_conv1_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_resnet_short_conv1_bias.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_resnet_short_conv2_weight.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_resnet_short_conv2_bias.safetensors".to_string()).unwrap();
    let linear_weight = input(r"C:\study\coursework\src\trash\test_resnet_short_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_resnet_short_linear_bias.safetensors".to_string()).unwrap();
    let mut tensor = input(r"C:\study\coursework\src\trash\test_resnet_short_test_image.safetensors".to_string()).unwrap();
    let temb = input(r"C:\study\coursework\src\trash\test_resnet_short_temb.safetensors".to_string()).unwrap();
    let norm1_w = input(r"C:\study\coursework\src\trash\test_resnet_short_norm1_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_resnet_short_norm1_bias.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_resnet_short_norm2_weight.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_resnet_short_norm2_bias.safetensors".to_string()).unwrap();  
    let kernels = input(r"C:\study\coursework\src\trash\test_resnet_short_conv_short_weight.safetensors".to_string()).unwrap();
    let cs_b = input(r"C:\study\coursework\src\trash\test_resnet_short_conv_short_bias.safetensors".to_string()).unwrap();
    let params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
        in_channels_1: 320, 
        out_channels_1: 640, 
        padding_1: 1, 
        stride_1 : 1, 
        kernel_size_1 : 3, 
        kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
        bias_c1: c1_b, is_bias_c1: true,
        weights: linear_weight, bias : linear_bias, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
        in_channels_2: 640, 
        out_channels_2: 640, 
        padding_2: 1, stride_2 : 1, 
        kernel_size_2 : 3, 
        kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
        bias_c2: c2_b, is_bias_c2: true,
        is_shortcut: true,
        in_channels_short: 320, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
        bias_s: cs_b, is_bias_s: true,
        time_emb: Rc::new(RefCell::new(temb))
    };
    let resnet = Resnet2d::new(params);
    let _ = resnet.operation(&mut tensor);
    let shape = tensor.shape();
    let py_tensor = input(r"C:\study\coursework\src\trash\test_resnet_short_output.safetensors".to_string()).unwrap();
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