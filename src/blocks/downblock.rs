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
    pub res_hidden_tensors: Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>,
}

impl DownBlock2D {
    pub fn DownBlock2D_constr(
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, out_channels : usize, padding : i32, stride : i32, kernel_size : usize, kernel_weights : Vec<f32>,
        hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let resnet1 = Resnet2d::Resnet2d_constr(params_for_resnet1);
        vec.push(Box::new(resnet1));
        let resnet2 = Resnet2d::Resnet2d_constr(params_for_resnet2);
        vec.push(Box::new(resnet2));
        let downsample = DownSample2D::DownSample2D_constr(in_channels, out_channels, padding, stride, kernel_size, kernel_weights);
        vec.push(Box::new(downsample));
        Self { operations: vec, res_hidden_tensors: hidden_states}
    }
}

impl Layer for DownBlock2D {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        let res_hidden_tensors = self.res_hidden_tensors.borrow_mut();
        let mut res_hidden_tensors = res_hidden_tensors;
        for layer in operations {
            let (temp_vec, temp_vec_shape) = layer.operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
            res_hidden_tensors.push((res_vec.clone(), res_vec_shape.clone()));
        } 
        Ok((res_vec, res_vec_shape))
    }
}

#[test]
fn test_downblock2d_unbiased() {
    let (test_vec, test_shape_vec) = input(r"C:\study\coursework\src\trash\test_downblock2d_test.safetensors".to_string()).unwrap();
    let (temb, temb_shape) = input(r"C:\study\coursework\src\trash\test_downblock2d_temb.safetensors".to_string()).unwrap();

    let (conv1_res1_vec, _) = input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res1_vec, lin_res1_vec_shape) = input(r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res1_bias, lin_res1_bias_shape) = input(r"C:\study\coursework\src\trash\test_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let (conv_down, _ ) = input(r"C:\study\coursework\src\trash\test_downblock2d_downsample.safetensors".to_string()).unwrap();

    let res1_params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res1_vec.to_vec(),
        weigths: lin_res1_vec.to_vec(), weights_shape: lin_res1_vec_shape.to_vec(), bias: lin_res1_bias.to_vec(), bias_shape: lin_res1_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res1_vec.to_vec(),
        is_shortcut: false,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone(),
        time_emb: temb.to_vec(), time_emb_shape: temb_shape.to_vec()
    };

    let (conv1_res2_vec, _) = input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_downblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res2_vec, lin_res2_vec_shape) = input(r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res2_bias, lin_res2_bias_shape) = input(r"C:\study\coursework\src\trash\test_downblock2d_res2_linear_bias.safetensors".to_string()).unwrap();

    let res2_params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res2_vec.to_vec(),
        weigths: lin_res2_vec.to_vec(), weights_shape: lin_res2_vec_shape.to_vec(), bias: lin_res2_bias.to_vec(), bias_shape: lin_res2_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: temb.to_vec(), time_emb_shape: temb_shape.to_vec(),
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res2_vec.to_vec(),
        is_shortcut: false,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone()
    };
    let hidden_states: Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>> = Rc::new(RefCell::new(Vec::<(Vec<f32>, Vec<usize>)>::new()));
    let downblock = DownBlock2D::DownBlock2D_constr(res1_params, res2_params, 320, 320, 1, 2, 3, conv_down.to_vec(), Rc::clone(&hidden_states));
    let (res_vec, res_vec_shape) = downblock.operation((test_vec.to_vec(), test_shape_vec.to_vec())).unwrap();
    let (py_vec, py_vec_shape) = input(r"C:\study\coursework\src\trash\test_downsample2d_output.safetensors".to_string()).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!( (res_vec[i] - py_vec[i]).abs() <= 1e-03)
    }
    let (py_hid1, py_hid1_shape) = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden1.safetensors".to_string()).unwrap();
    let (py_hid2, py_hid2_shape) = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden2.safetensors".to_string()).unwrap();
    let (py_hid3, py_hid3_shape) = input(r"C:\study\coursework\src\trash\test_downsample2d_output_hidden3.safetensors".to_string()).unwrap();
    assert!(py_hid1_shape.to_vec() ==  hidden_states.borrow()[0].1);
    assert!(py_hid2_shape.to_vec() ==  hidden_states.borrow()[1].1);
    assert!(py_hid3_shape.to_vec() ==  hidden_states.borrow()[2].1);
    for i in 0..py_hid1.len() {
        assert!((py_hid1[i] - hidden_states.borrow()[0].0[i]).abs() <= 1e-03);
    }
    for i in 0..py_hid2.len() {
        assert!((py_hid2[i] - hidden_states.borrow()[1].0[i]).abs() <= 1e-03);
    }
    for i in 0..py_hid3.len() {
        assert!((py_hid3[i] - hidden_states.borrow()[2].0[i]).abs() <= 1e-03);
    }
}