use crate::layers::{
    params::Resnet2d_params,
    norm::GroupNorm,
    act::SiLU,
    linear::Linear,
    conv::Conv2d,
    layer::Layer
};

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
                let lin_res = self.operations[i].operation((self.time_emb.clone(), self.time_emb_shape.clone()))?;
                let mut curr_tensor = ndarray::Array4::from_shape_vec((res_shape_vec[0],res_shape_vec[1],res_shape_vec[2],res_shape_vec[3]), res_vec.clone())?;
                let time_tensor = ndarray::Array2::from_shape_vec((lin_res.1[0], lin_res.1[1]), lin_res.0)?;
                curr_tensor = curr_tensor.clone() + time_tensor.broadcast(curr_tensor.dim()).unwrap();
                res_vec = curr_tensor.into_raw_vec_and_offset().0;
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
            let mut short_tensor = ndarray::Array4::from_shape_vec((shortcut_res.1[0], shortcut_res.1[1], shortcut_res.1[2], shortcut_res.1[3]), shortcut_vec.clone())?;
            curr_tensor = curr_tensor + short_tensor;
            res_vec = curr_tensor.into_raw_vec_and_offset().0;
        }
        Ok((res_vec, res_shape_vec))
    }
}