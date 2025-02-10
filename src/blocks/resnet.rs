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
    pub time_emb : Vec<f32>,
    pub time_emb_shape : Vec<usize>,
}

impl Resnet2d {
    pub fn Resnet2d_constr (
        params : Resnet2d_params
        ) -> Self {
            let mut layer_vec : Vec<Box<dyn Layer>> = Vec::new();
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_1, eps : params.eps_1, gamma : params.gamma_1, beta : params.beta_1}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_1, out_channels : params.out_channels_1, padding : params.padding_1, stride: params.stride_2, kernel_size : params.kernel_size_1, kernel_weights : params.kernel_weights_1}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Linear{weigths : params.weigths, weights_shape : params.weights_shape, bias : params.bias, bias_shape : params.bias_shape, is_bias : params.is_bias}));
            layer_vec.push(Box::new(GroupNorm {number_of_groups: params.number_of_groups_2, eps : params.eps_2, gamma : params.gamma_2, beta : params.beta_2}));
            layer_vec.push(Box::new(SiLU));
            layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_2, out_channels : params.out_channels_2, padding : params.padding_2, stride: params.stride_2, kernel_size : params.kernel_size_2, kernel_weights : params.kernel_weights_2}));
            if params.is_shortcut {
                layer_vec.push(Box::new(Conv2d {in_channels : params.in_channels_short, out_channels : params.out_channels_short, stride: params.stride_short, padding : params.padding_short, kernel_size : params.kernel_size_short, kernel_weights : params.kernel_weights_short}));
            }
            Self { if_shortcut: params.is_shortcut, operations: layer_vec, time_emb : params.time_emb, time_emb_shape : params.time_emb_shape}
    }   
}
impl Layer for Resnet2d {
    fn operation(&self, args:(Vec<f32>, Vec<usize>)) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let mut res_vec = args.0.clone();
        let mut res_shape_vec = args.1.clone();
        for i in 0..self.operations.len()-(self.if_shortcut as usize) {
            if i == 3 {
                let act_lin_res = self.operations[i].operation((self.time_emb.clone(), self.time_emb_shape.clone()))?;
                let lin_res = self.operations[i + 1].operation((act_lin_res.0, act_lin_res.1))?;
                let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
                let time_tensor = ndarray::Array4::from_shape_vec((lin_res.1[2], lin_res.1[3], 1, 1), lin_res.0)?;
                curr_tensor = curr_tensor + time_tensor;
                res_vec = curr_tensor.into_raw_vec_and_offset().0;
                continue;
            }
            if i == 4{
                continue;
            }
            let res = self.operations[i].operation((res_vec, res_shape_vec))?;
            res_vec = res.0.clone();
            res_shape_vec = res.1.clone();
        }
        if self.if_shortcut {
            let shortcut_res = self.operations[self.operations.len() - 1].operation(args.clone())?;
            let shortcut_vec = shortcut_res.0;
            let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
            let short_tensor = ndarray::Array4::from_shape_vec((shortcut_res.1[0], shortcut_res.1[1], shortcut_res.1[2], shortcut_res.1[3]), shortcut_vec.clone())?;
            curr_tensor = curr_tensor + short_tensor;
            res_vec = curr_tensor.into_raw_vec_and_offset().0;
        } else {
            let mut curr_tensor =  ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
            let input_tensor =  ndarray::Array4::from_shape_vec((args.1[0],args.1[1], args.1[2], args.1[3]), args.0)?;
            curr_tensor = curr_tensor + input_tensor;
            res_vec = curr_tensor.into_raw_vec_and_offset().0;
        }
        Ok((res_vec, res_shape_vec))
    }
}

#[test]
fn test_resnet_no_shortcut_no_bias() {
    let (conv1_vec, conv1_shape_vec) = input(r"C:\study\coursework\src\trash\test_resnet_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_vec, conv2_shape_vec) = input(r"C:\study\coursework\src\trash\test_resnet_conv2_weight.safetensors".to_string()).unwrap();
    let (linear_weight, linear_weight_shape) = input(r"C:\study\coursework\src\trash\test_resnet_linear_weight.safetensors".to_string()).unwrap();
    let (linear_bias, linear_bias_shape) = input(r"C:\study\coursework\src\trash\test_resnet_linear_bias.safetensors".to_string()).unwrap();
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors".to_string()).unwrap();
    let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_resnet_temb.safetensors".to_string()).unwrap();
    let mut temb_vec_shape = temb_vec_shape.to_vec();
    temb_vec_shape.insert(0, 1);
    temb_vec_shape.insert(0, 1);
    let params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 320, padding_1: 1, stride_1 : 1, kernel_size_1 : 3, kernel_weights_1: conv1_vec.to_vec(), 
        weigths: linear_weight.to_vec(), weights_shape: linear_weight_shape.to_vec(), bias : linear_bias.to_vec(), bias_shape : linear_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0.,
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2 : 1, kernel_size_2 : 3, kernel_weights_2: conv2_vec.to_vec(),
        is_shortcut: false,
        in_channels_short: 320, out_channels_short: 320, padding_short: 1, stride_short : 1, kernel_size_short : 3, kernel_weights_short: conv2_vec.to_vec(),
        time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
    };
    let resnet = Resnet2d::Resnet2d_constr(params);
    let (res_vec, res_vec_shape) = resnet.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_resnet, py_resnet_shape) = input(r"C:\study\coursework\src\trash\test_resnet_output.safetensors".to_string()).unwrap();
    assert!( res_vec_shape == py_resnet_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - py_resnet[i]).abs() <= 1e-02);
    }
}

#[test]
fn test_resnet_shortcut_no_bias() {
    let (conv1_vec, conv1_shape_vec) = input(r"C:\study\coursework\src\trash\test_resnet_short_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_vec, conv2_shape_vec) = input(r"C:\study\coursework\src\trash\test_resnet_short_conv2_weight.safetensors".to_string()).unwrap();
    let (conv_short_vec, conv_short_shape_vec) = input(r"C:\study\coursework\src\trash\test_resnet_short_conv_short_weight.safetensors".to_string()).unwrap();
    let (linear_weight, linear_weight_shape) = input(r"C:\study\coursework\src\trash\test_resnet_short_linear_weight.safetensors".to_string()).unwrap();
    let (linear_bias, linear_bias_shape) = input(r"C:\study\coursework\src\trash\test_resnet_short_linear_bias.safetensors".to_string()).unwrap();
    let (test_vec, test_vec_shape) = input(r"C:\study\coursework\src\trash\test_resnet_test_image.safetensors".to_string()).unwrap();
    let (temb_vec, temb_vec_shape) = input(r"C:\study\coursework\src\trash\test_resnet_temb.safetensors".to_string()).unwrap();
    let mut temb_vec_shape = temb_vec_shape.to_vec();
    temb_vec_shape.insert(0, 1);
    temb_vec_shape.insert(0, 1);
    let params = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 640, padding_1: 1, stride_1 : 1, kernel_size_1 : 3, kernel_weights_1: conv1_vec.to_vec(), 
        weigths: linear_weight.to_vec(), weights_shape: linear_weight_shape.to_vec(), bias : linear_bias.to_vec(), bias_shape : linear_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0.,
        in_channels_2: 640, out_channels_2: 640, padding_2: 1, stride_2 : 1, kernel_size_2 : 3, kernel_weights_2: conv2_vec.to_vec(),
        is_shortcut: true,
        in_channels_short: 320, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: conv_short_vec.to_vec(),
        time_emb: temb_vec.to_vec(), time_emb_shape: temb_vec_shape.to_vec()
    };
    let resnet = Resnet2d::Resnet2d_constr(params);
    let (res_vec, res_vec_shape) = resnet.operation((test_vec.to_vec(), test_vec_shape.to_vec())).unwrap();
    let (py_resnet, py_resnet_shape) = input(r"C:\study\coursework\src\trash\test_resnet_short_output.safetensors".to_string()).unwrap();
    assert!( res_vec_shape == py_resnet_shape.to_vec());
    for i in 0..res_vec.len() {
        assert!((res_vec[i] - py_resnet[i]).abs() <= 1e-01);
    }
}