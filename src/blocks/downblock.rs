use crate::{
    layers::{
        layer::Layer,
        params::Resnet2d_params,
        downsample::DownSample2D
    },
    blocks::{
        resnet::Resnet2d,
    },
    func::functions::input
};
use std::rc::Rc;
use std::cell::RefCell;

pub struct DownBlock2D {
    pub operations : Vec<Box<dyn Layer>>,
    pub res_hidden_tensors: Rc<RefCell<Vec<ndarray::Array4<f32>>>>
}

impl DownBlock2D {
    pub fn new(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
        bias: ndarray::Array4<f32>, is_bias: bool,
        hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::new(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::new(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let downsample = DownSample2D::new(in_channels, out_channels, padding, stride, kernel_size, kernel_weights, bias, is_bias);
        vec.push(Box::new(downsample));
        Self { operations: vec, res_hidden_tensors: hidden_states}
    }
}

impl Layer for DownBlock2D {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let res_hidden_tensors = self.res_hidden_tensors.borrow_mut();
        let mut res_hidden_tensors = res_hidden_tensors;
        for layer in operations {
            let _ = layer.operation(args)?;
            res_hidden_tensors.push(args.clone());
        } 
        Ok(())
    }
}

#[test]
fn test_downblock2d_biased() {
    let mut tensor = input(r"C:\study\coursework\src\trash\test_downblock2d_test.safetensors".to_string()).unwrap();
    let temb = input(r"C:\study\coursework\src\trash\test_downblock2d_temb.safetensors".to_string()).unwrap();

    let r1k1= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let r1k2= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let r1c1b= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_bias.safetensors".to_string()).unwrap();
    let r1c2b= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_bias.safetensors".to_string()).unwrap();

    let r1n1w= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let r1n2w= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let r1n1b= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_norm1_bias.safetensors".to_string()).unwrap();
    let r1n2b= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_norm2_bias.safetensors".to_string()).unwrap();



    let lw1= input(r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let lb1 = input(r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let kdown = input(r"C:\study\coursework\src\trash\test_downblock2d_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblock2d_downsample_b.safetensors".to_string()).unwrap();


    let time_emb = Rc::new(RefCell::new(temb));
    
    let res1_params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: r1n1w, beta_1: r1n1b,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: r1k1.into_raw_vec_and_offset().0,
        bias_c1: r1c1b, is_bias_c1: true,
        weights: lw1 , bias: lb1, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: r1n2w, beta_2: r1n2b, 
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: r1k2.into_raw_vec_and_offset().0,
        bias_c2: r1c2b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem((1, 1, 1, 1), 1.),
        is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };



    let r1k1= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let r1k2= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let r1c1b= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_bias.safetensors".to_string()).unwrap();
    let r1c2b= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_bias.safetensors".to_string()).unwrap();

    let r1n1w= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_norm1_weight.safetensors".to_string()).unwrap();
    let r1n2w= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_norm2_weight.safetensors".to_string()).unwrap();
    let r1n1b= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_norm1_bias.safetensors".to_string()).unwrap();
    let r1n2b= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_norm2_bias.safetensors".to_string()).unwrap();



    let lw1= input(r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let lb1 = input(r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_bias.safetensors".to_string()).unwrap();


    
    let res2_params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: r1n1w, beta_1: r1n1b,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: r1k1.into_raw_vec_and_offset().0,
        bias_c1: r1c1b, is_bias_c1: true,
        weights: lw1 , bias: lb1, is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: r1n2w, beta_2: r1n2b, 
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: r1k2.into_raw_vec_and_offset().0,
        bias_c2: r1c2b, is_bias_c2: true,
        is_shortcut: false,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: vec![1.],
        bias_s: ndarray::Array4::from_elem((1, 1, 1, 1), 1.),
        is_bias_s: true,
        time_emb: Rc::clone(&time_emb)
    };

    let hidden_states: Rc<RefCell<Vec<ndarray::Array4<f32>>>> = Rc::new(RefCell::new(Vec::new()));
    let downblock = DownBlock2D::new(
        res1_params, 
        res2_params, 
        320, 320, 1, 2, 3, 
        kdown.into_raw_vec_and_offset().0, 
        cdown_b, true,
        Rc::clone(&hidden_states)
    );
    
    let _ = downblock.operation(&mut tensor).unwrap();
    let shape = tensor.shape();

    let py_tensor    = input(r"C:\study\coursework\src\trash\test_downsample2d_output.safetensors".to_string()).unwrap();
        assert!(shape == py_tensor.shape());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for r in 0..shape[2] {
                for k in 0..shape[3] {
                    assert!((tensor[[i, j, r, k]] - py_tensor[[i, j, r, k]]).abs() <= 1e-03);
                }
            }
        }
    }

    let hid1 = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden1.safetensors".to_string()).unwrap();
    let hid2 = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden2.safetensors".to_string()).unwrap();
    let hid3 = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden3.safetensors".to_string()).unwrap();
    let vec = hidden_states.borrow();
    assert!(hid1.shape() ==  vec[0].shape());
    assert!(hid2.shape() == vec[1].shape());
    assert!(hid3.shape() ==  vec[2].shape());
    let sh_h1 =  vec[0].shape();
        for i in 0..sh_h1[0] {
        for j in 0..sh_h1[1] {
            for r in 0..sh_h1[2] {
                for k in 0..sh_h1[3] {
                    assert!((vec[0][[i, j, r, k]] - hid1[[i, j, r, k]]).abs() <= 1e-02);
                }
            }
        }
    }


    let sh_h1 =  vec[1].shape();
        for i in 0..sh_h1[0] {
        for j in 0..sh_h1[1] {
            for r in 0..sh_h1[2] {
                for k in 0..sh_h1[3] {
                    assert!((vec[1][[i, j, r, k]] - hid2[[i, j, r, k]]).abs() <= 1e-02);
                }
            }
        }
    }

    let sh_h1 =  vec[2].shape();
        for i in 0..sh_h1[0] {
        for j in 0..sh_h1[1] {
            for r in 0..sh_h1[2] {
                for k in 0..sh_h1[3] {
                    assert!((vec[2][[i, j, r, k]] - hid3[[i, j, r, k]]).abs() <= 1e-02);
                }
            }
        }
    }

}