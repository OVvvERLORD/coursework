use crate::{
    blocks::{
        down::Down_blocks, mid::mid_block, resnet::Resnet2d, trans::Transformer2D, up::Up_blocks
    }, func::functions::{input, scalar_timestep_embedding}, layers::{
        act::SiLU, conv::Conv2d, layer::Layer, linear::Linear, norm::GroupNorm, params::{
            BasicTransofmerBlock_params, CrossAttnDownBlock2D_params, CrossAttnUpBlock2D_params, Resnet2d_params, Transformer2D_params
        }
    }
};

// use core::time;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Unet2dConditionModel {
    pub operations: Vec<Box<dyn Layer>>,
    pub time_emb: Rc<RefCell<(Vec<f32>, Vec<usize>)>>,
    pub timestep: f32,
    pub hidden_states: Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>
}
impl Unet2dConditionModel {
    pub fn unet2d_condition_model_constr(
        timestep : f32, // time
        time_emb : Rc<RefCell<(Vec<f32>, Vec<usize>)>>,
        weigths1: Vec<f32>, weights_shape1 : Vec<usize>, bias1: Vec<f32>, bias_shape1 : Vec<usize>, is_bias1 : bool,
        weigths2 : Vec<f32>, weights_shape2 : Vec<usize>, bias2: Vec<f32>, bias_shape2 : Vec<usize>, is_bias2 : bool,

        in_channels_in : usize, out_channels_in: usize, padding_in : i32, stride_in : i32, kernel_size_in : usize, kernel_weights_in : Vec<f32>, //in

        params_for_resnet1_down : Resnet2d_params, // down
        params_for_resnet2_down : Resnet2d_params,
        in_channels_down : usize, out_channels_down : usize, padding_down : i32, stride_down : i32, kernel_size_down : usize, kernel_weights_down : Vec<f32>,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,

        params_for_crossblock1 : CrossAttnUpBlock2D_params, // up
        params_for_crossblock2 : CrossAttnUpBlock2D_params,
        params_for_resnet1_up : Resnet2d_params,
        params_for_resnet2_up : Resnet2d_params,
        params_for_resnet3_up : Resnet2d_params,
        hidden_states : Rc<RefCell<Vec<(Vec<f32>, Vec<usize>)>>>,

        params_for_transformer2d :Transformer2D_params, // mid
        params_for_resnet_1_mid: Resnet2d_params,
        params_for_resnet_2_mid: Resnet2d_params,

        number_of_groups_out: usize, eps_out: f32, gamma_out: f32, beta_out: f32, //out
        in_channels_out : usize, out_channels_out: usize, padding_out : i32, stride_out : i32, kernel_size_out : usize, kernel_weights_out : Vec<f32>,
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        vec.push(Box::new(Linear {weigths : weigths1, weights_shape : weights_shape1, bias : bias1, bias_shape : bias_shape1, is_bias : is_bias1}));
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Linear {weigths : weigths2, weights_shape : weights_shape2, bias : bias2, bias_shape : bias_shape2, is_bias : is_bias2}));
        vec.push(Box::new(Conv2d{in_channels : in_channels_in, out_channels : out_channels_in, padding : padding_in, stride : stride_in, kernel_size : kernel_size_in, kernel_weights : kernel_weights_in}));
        let down = Down_blocks::Down_blocks_constr(params_for_resnet1_down, params_for_resnet2_down, in_channels_down, out_channels_down, padding_down, stride_down, kernel_size_down, kernel_weights_down, params_for_crattbl1, params_for_crattbl2, Rc::clone(&hidden_states));
        vec.push(Box::new(down));
        let mid = mid_block::mid_block_constr(params_for_transformer2d, params_for_resnet_1_mid, params_for_resnet_2_mid);
        vec.push(Box::new(mid));
        let up = Up_blocks::Up_block_constr(params_for_crossblock1, params_for_crossblock2, params_for_resnet1_up, params_for_resnet2_up, params_for_resnet3_up, Rc::clone(&hidden_states));
        vec.push(Box::new(up));
        vec.push(Box::new(GroupNorm{number_of_groups : number_of_groups_out, eps : eps_out, gamma : gamma_out, beta: beta_out}));
        vec.push(Box::new(SiLU));
        vec.push(Box::new(Conv2d{in_channels: in_channels_out, out_channels : out_channels_out, padding : padding_out, stride: stride_out, kernel_size : kernel_size_out, kernel_weights : kernel_weights_out}));
        Self { operations: vec, time_emb: Rc::clone(&time_emb), timestep: timestep , hidden_states: hidden_states}
    }
}

impl Unet2dConditionModel { 
    pub fn operation(&mut self, args:(Vec<f32>, Vec<usize>), timestep:f32) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        // let mut temb = self.time_emb.borrow_mut();
        // let mut in_temb = temb.0.clone();
        // let mut in_temb_shape = temb.1.clone();
        self.timestep = timestep;
        let (in_temb, in_temb_shape) = scalar_timestep_embedding(self.timestep, args.1[0], 320).unwrap();
        let (temp_emb_vec, temp_emb_shape) = operations[0].operation((in_temb, in_temb_shape))?;
        let (time_emb_vec, time_emb_shape) = operations[1].operation((temp_emb_vec, temp_emb_shape))?;
        let (time_emb_vec, time_emb_shape) = operations[2].operation((time_emb_vec, time_emb_shape))?;
        {*self.time_emb.borrow_mut() = (time_emb_vec.clone(), time_emb_shape);} // remove clone
        let mut res_vec = args.0;
        let mut res_vec_shape = args.1;
        for i in 3..operations.len(){
            let (temp_vec, temp_vec_shape) = operations[i].operation((res_vec.clone(), res_vec_shape.clone()))?;
            res_vec = temp_vec;
            res_vec_shape = temp_vec_shape;
        } 
        Ok((res_vec, res_vec_shape))
    }
}

#[test]
fn test_timeemb() {
    let (temb, temb_shape) = scalar_timestep_embedding(0., 2, 320).unwrap();
    let (l1w, l1ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
    let (l1b, l1bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
    let linear1 = Linear{weigths:l1w.to_vec(), weights_shape: l1ws.to_vec(), bias: l1b.to_vec(), bias_shape: l1bs.to_vec(), is_bias: true};
    let (l2w, l2ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
    let (l2b, l2bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();
    let linear2 = Linear{weigths:l2w.to_vec(), weights_shape: l2ws.to_vec(), bias: l2b.to_vec(), bias_shape: l2bs.to_vec(), is_bias: true};
    let silu = SiLU;
    let (temb, temb_shape) = linear1.operation((temb.clone(), temb_shape.clone())).unwrap();
    let (temb, temb_shape) = silu.operation((temb.clone(), temb_shape.clone())).unwrap();
    let (temb, temb_shape) = linear2.operation((temb.clone(), temb_shape.clone())).unwrap();
    let (py_temb, py_temb_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_output.safetensors")).unwrap();
    assert!(py_temb_shape.to_vec() == temb_shape);
    for i in 0..py_temb.len() {
        let d = (temb[i] - py_temb[i]).abs();
        assert!(!d.is_nan());
        assert!(d <= 1e-05);
    }
}

#[test]
fn test_unet_simple_unbiased() {
    let time_emb = Rc::new(RefCell::new((Vec::<f32>::new(), Vec::<usize>::new())));
    let timestep = 0;
    let mut res_hidden_states = Rc::new(RefCell::new(Vec::<(Vec::<f32>, Vec::<usize>)>::new()));
    
    let (encoder_vec, encoder_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")).unwrap();



    let (conv1_res1_vec, _) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res1_vec, lin_res1_vec_shape) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res1_bias, lin_res1_bias_shape) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let (conv_down, _ ) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_downsample.safetensors".to_string()).unwrap();

    let res1_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res1_vec.to_vec(),
        weigths: lin_res1_vec.to_vec(), weights_shape: lin_res1_vec_shape.to_vec(), bias: lin_res1_bias.to_vec(), bias_shape: lin_res1_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res1_vec.to_vec(),
        is_shortcut: false,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone(),
        time_emb: Rc::clone(&time_emb)
    };

    let (conv1_res2_vec, _) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res2_vec, lin_res2_vec_shape) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res2_bias, lin_res2_bias_shape) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();

    let res2_params_down = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 320, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res2_vec.to_vec(),
        weigths: lin_res2_vec.to_vec(), weights_shape: lin_res2_vec_shape.to_vec(), bias: lin_res2_bias.to_vec(), bias_shape: lin_res2_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res2_vec.to_vec(),
        is_shortcut: false,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_down.to_vec().clone()
    };

    let (downsample_conv, _) = input(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_downsample.safetensors".to_string()).unwrap();
    let mut trans1_params_down : Option<Transformer2D_params> = None;
    let mut trans2_params_down : Option<Transformer2D_params> = None;
    let mut resnet1_params_down : Option<Resnet2d_params> = None;
    let mut resnet2_params_down : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short) = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_resnet{}_conv_short.safetensors", i)).unwrap().0}
        else
        {kernel_weights_1.clone()};
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 640, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 640, out_channels_2: 640, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: Rc::clone(&time_emb)
        };
        if i == 0 {
            resnet1_params_down = Some(resnet_par);
        } else {
            resnet2_params_down = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 640, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 640,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 640,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 10,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 10,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
        };
        if j == 0 {
            trans1_params_down = Some(params);
        } else{
            trans2_params_down = Some(params);
        }
    }
    let crossattndownblock2d_1_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params_down.unwrap(),
        params_for_transformer2: trans2_params_down.unwrap(),
        params_for_resnet1: resnet1_params_down.unwrap(),
        params_for_resnet2: resnet2_params_down.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: downsample_conv.to_vec(),
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short) = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_resnet{}_conv_short.safetensors", i)).unwrap().0}
        else
        {kernel_weights_1.clone()};
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {640} else {1280};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: Rc::clone(&time_emb)
        };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else {
            resnet22_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_crossattndownblock2_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else{
            trans22_params = Some(params);
        }
    }
    let crossattndownblock2d_2_params = CrossAttnDownBlock2D_params {
        is_downsample2d: false,
        params_for_transformer1: trans12_params.unwrap(),
        params_for_transformer2: trans22_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 13123124, kernel_weights: downsample_conv.to_vec(),
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let (downblock2d_downsample, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_downblocks_downblock2d_downsample.safetensors")).unwrap();



    let mut trans1:Transformer2D;
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut resnet1:Resnet2d;
    let mut resnet1_params_mid : Option<Resnet2d_params> = None;
    let mut resnet2:Resnet2d;
    let mut resnet2_params_mid : Option<Resnet2d_params> = None;
    for i in 0..2 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{}_conv2.safetensors", i)).unwrap();
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = 1280;
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: false,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_1.to_vec(),
            time_emb: Rc::clone(&time_emb)
        };
        if i == 0 {
            resnet1_params_mid = Some(resnet_par);
        } else {
            resnet2_params_mid = Some(resnet_par);
        }
    }
    let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
    for i in 0..10 {
        let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn1_q_test.safetensors", i)).unwrap();
        let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn1_k_test.safetensors", i)).unwrap();
        let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn1_v_test.safetensors", i)).unwrap();
        let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn1_out_w_test.safetensors", i)).unwrap(); 
        let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn1_out_b_test.safetensors", i)).unwrap(); 
    
        let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn2_q_test.safetensors", i)).unwrap();
        let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn2_k_test.safetensors", i)).unwrap();
        let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn2_v_test.safetensors", i)).unwrap();
        let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn2_out_w_test.safetensors", i)).unwrap(); 
        let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_attn2_out_b_test.safetensors", i)).unwrap(); 
    
        let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_geglu_w_test.safetensors", i)).unwrap();
        let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_geglu_b_test.safetensors", i)).unwrap();
    
        let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_ff_w_test.safetensors", i)).unwrap();
        let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_btb{}_ff_b_test.safetensors", i)).unwrap();
    
        let btb1_params = BasicTransofmerBlock_params {
        eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
        eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
        eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
        weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
        weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
        weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
        weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
        encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
    
        weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
        weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
        weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
        weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
        encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
    
        weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
        weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
        };
        param_vec.push(btb1_params);
    }
    let (weigths_in,  weights_shape_in) = input(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projin_w_test.safetensors".to_string()).unwrap(); 
    let (bias_in, bias_shape_in) = input(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projin_b_test.safetensors".to_string()).unwrap(); 

    let (weigths_out,  weights_shape_out) = input(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projout_w_test.safetensors".to_string()).unwrap(); 
    let (bias_out, bias_shape_out) = input(r"C:\study\coursework\src\trash\test_unet_crossattnmidblock_trans_projout_b_test.safetensors".to_string()).unwrap(); 

    let params_trans_mid = Transformer2D_params{
        number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
        weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
        weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
        params_for_basics_vec: param_vec
    };



    let (conv1_res1_vec, _) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res1_vec, lin_res1_vec_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res1_bias, lin_res1_bias_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let (conv_short_res1_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res0_conv_short_weight.safetensors".to_string()).unwrap();

    let res1_params_up = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 960, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res1_vec.to_vec(),
        weigths: lin_res1_vec.to_vec(), weights_shape: lin_res1_vec_shape.to_vec(), bias: lin_res1_bias.to_vec(), bias_shape: lin_res1_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res1_vec.to_vec(),
        is_shortcut: true,
        in_channels_short: 960, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res1_vec.to_vec(),
        time_emb: Rc::clone(&time_emb)
    };

    let (conv1_res2_vec, _) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res2_vec, lin_res2_vec_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res2_bias, lin_res2_bias_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res1_linear_bias.safetensors".to_string()).unwrap();
    let (conv_short_res2_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res1_conv_short_weight.safetensors".to_string()).unwrap();
    let res2_params_up = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 640, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res2_vec.to_vec(),
        weigths: lin_res2_vec.to_vec(), weights_shape: lin_res2_vec_shape.to_vec(), bias: lin_res2_bias.to_vec(), bias_shape: lin_res2_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res2_vec.to_vec(),
        is_shortcut: true,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res2_vec.to_vec()
    };


    let (conv1_res3_vec, _) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res2_conv1_weight.safetensors".to_string()).unwrap();
    let (conv2_res3_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res2_conv2_weight.safetensors".to_string()).unwrap();
    let (lin_res3_vec, lin_res3_vec_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res2_linear_weight.safetensors".to_string()).unwrap();
    let (lin_res3_bias, lin_res3_bias_shape) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res2_linear_bias.safetensors".to_string()).unwrap();
    let (conv_short_res3_vec, _ ) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_upblock2d_res2_conv_short_weight.safetensors".to_string()).unwrap();
    let res3_params_up = Resnet2d_params{
        number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0.,
        in_channels_1: 640, out_channels_1: 320, kernel_size_1: 3, stride_1: 1, padding_1: 1, kernel_weights_1: conv1_res3_vec.to_vec(),
        weigths: lin_res3_vec.to_vec(), weights_shape: lin_res3_vec_shape.to_vec(), bias: lin_res3_bias.to_vec(), bias_shape: lin_res3_bias_shape.to_vec(), is_bias: true,
        number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., time_emb: Rc::clone(&time_emb),
        in_channels_2: 320, out_channels_2: 320, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: conv2_res3_vec.to_vec(),
        is_shortcut: true,
        in_channels_short: 640, out_channels_short: 320, padding_short: 0, stride_short: 1, kernel_size_short: 1, kernel_weights_short: conv_short_res3_vec.to_vec()
    };

    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut trans3_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;
    let mut resnet3_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_resnet{}_conv_short.safetensors", i)).unwrap();
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 || i == 1 {2560} else {1920};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 1280, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 1280, out_channels_2: 1280, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: Rc::clone(&time_emb)
        };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else if i == 1 {
            resnet2_params = Some(resnet_par);
        } else {
            resnet3_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..10 {
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 20,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 20,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else if j == 1 {
            trans2_params = Some(params);
        } else {
            trans3_params = Some(params);
        }
    }
    let (upsample1_conv, _) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock0_upsample.safetensors".to_string()).unwrap();
    let crossattnupblock_1_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans1_params.unwrap(), 
        params_for_transformer2: trans2_params.unwrap(),
        params_for_transformer3: trans3_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        params_for_resnet3: resnet3_params.unwrap(),
        in_channels: 1280, out_channels: 1280, padding: 1, stride: 1, kernel_size: 3, kernel_weights: upsample1_conv.to_vec(),
        hidden_states: Rc::clone(&res_hidden_states)
    };

    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut trans32_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;
    let mut resnet32_params : Option<Resnet2d_params> = None;

    for i in 0..3 {
        let (kernel_weights_1, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let (kernel_weights_2, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let (kernel_weights_short, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_resnet{}_conv_short.safetensors", i)).unwrap();
        let (weigths, weights_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let (bias, bias_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {1920} else if i == 1 {1280} else {960};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: 1., beta_1: 0., 
            in_channels_1: in_ch, out_channels_1: 640, padding_1: 1, stride_1: 1, kernel_size_1: 3, kernel_weights_1 : kernel_weights_1.to_vec(),
            weigths: weigths.to_vec(), weights_shape : weights_shape.to_vec(), bias: bias.to_vec(), bias_shape: bias_shape.to_vec(), is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: 1., beta_2: 0., 
            in_channels_2: 640, out_channels_2: 640, padding_2: 1, stride_2: 1, kernel_size_2: 3, kernel_weights_2: kernel_weights_2.to_vec(),
            is_shortcut: true,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short:1, kernel_size_short: 1, kernel_weights_short: kernel_weights_short.to_vec(),
            time_emb: Rc::clone(&time_emb)
        };
        if i == 0 {
            resnet12_params = Some(resnet_par);
        } else if i == 1 {
            resnet22_params = Some(resnet_par);
        } else {
            resnet32_params = Some(resnet_par);
        }
    }
    for j in 0..3 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let (btb1_weigths_1, btb1_weights_shape_1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_2, btb1_weights_shape_2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_3, btb1_weights_shape_3) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_4, btb1_weights_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_4, btb1_bias_shape_4) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_5, btb1_weights_shape_5) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_6, btb1_weights_shape_6) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_7, btb1_weights_shape_7) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let (btb1_weigths_8, btb1_weights_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let (btb1_bias_8, btb1_bias_shape_8) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let (btb1_weigths_ff1, btb1_weights_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff1, btb1_bias_shape_ff1) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let (btb1_weigths_ff2, btb1_weights_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let (btb1_bias_ff2, btb1_bias_shape_ff2) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let btb1_params = BasicTransofmerBlock_params {
            eps_1: 1e-05, gamma_1: 1., beta_1: 0., number_1: 1280, 
            eps_2: 1e-05, gamma_2: 1., beta_2: 0., number_2: 1280,
            eps_3: 1e-05, gamma_3: 1., beta_3: 0., number_3: 1280,
            weigths_1: btb1_weigths_1.to_vec(), weights_shape_1: btb1_weights_shape_1.to_vec(), bias_1: btb1_weigths_1.to_vec(), bias_shape_1: btb1_weights_shape_1.to_vec(), is_bias_1: false,
            weigths_2: btb1_weigths_2.to_vec(), weights_shape_2: btb1_weights_shape_2.to_vec(), bias_2: btb1_weigths_2.to_vec(), bias_shape_2: btb1_weights_shape_2.to_vec(), is_bias_2: false,
            weigths_3: btb1_weigths_3.to_vec(), weights_shape_3: btb1_weights_shape_3.to_vec(), bias_3: btb1_weigths_3.to_vec(), bias_shape_3: btb1_weights_shape_3.to_vec(), is_bias_3: false,
            weigths_4: btb1_weigths_4.to_vec(), weights_shape_4: btb1_weights_shape_4.to_vec(), bias_4: btb1_bias_4.to_vec(), bias_shape_4: btb1_bias_shape_4.to_vec(), is_bias_4: true,
            encoder_hidden_tensor_1: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_1: false, number_of_heads_1: 10,
        
            weigths_5: btb1_weigths_5.to_vec(), weights_shape_5: btb1_weights_shape_5.to_vec(), bias_5: btb1_weigths_5.to_vec(), bias_shape_5: btb1_weights_shape_5.to_vec(), is_bias_5: false,
            weigths_6: btb1_weigths_6.to_vec(), weights_shape_6: btb1_weights_shape_6.to_vec(), bias_6: btb1_weigths_6.to_vec(), bias_shape_6: btb1_weights_shape_6.to_vec(), is_bias_6: false,
            weigths_7: btb1_weigths_7.to_vec(), weights_shape_7: btb1_weights_shape_7.to_vec(), bias_7: btb1_weigths_7.to_vec(), bias_shape_7: btb1_weights_shape_7.to_vec(), is_bias_7: false,
            weigths_8: btb1_weigths_8.to_vec(), weights_shape_8: btb1_weights_shape_8.to_vec(), bias_8: btb1_bias_8.to_vec(), bias_shape_8: btb1_bias_shape_8.to_vec(), is_bias_8: true,
            encoder_hidden_tensor_2: Rc::new(RefCell::new((encoder_vec.to_vec(), encoder_vec_shape.to_vec()))), if_encoder_tensor_2: true, number_of_heads_2: 10,
        
            weigths_ff1: btb1_weigths_ff1.to_vec(), weights_shape_ff1: btb1_weights_shape_ff1.to_vec(), bias_ff1: btb1_bias_ff1.to_vec(), bias_shape_ff1: btb1_bias_shape_ff1.to_vec(), is_bias_ff1: true,
            weigths_ff2: btb1_weigths_ff2.to_vec(), weights_shape_ff2: btb1_weights_shape_ff2.to_vec(), bias_ff2: btb1_bias_ff2.to_vec(), bias_shape_ff2: btb1_bias_shape_ff2.to_vec(), is_bias_ff2: true
            };
            param_vec.push(btb1_params);
        }
        let (weigths_in,  weights_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let (bias_in, bias_shape_in) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let (weigths_out,  weights_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let (bias_out, bias_shape_out) = input(format!(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: 1., beta: 0.,
            weigths_in: weigths_in.to_vec(), weights_shape_in: weights_shape_in.to_vec(), bias_in: bias_in.to_vec(), bias_shape_in: bias_shape_in.to_vec(), is_bias_in: true,
            weigths_out: weigths_out.to_vec(), weights_shape_out: weights_shape_out.to_vec(), bias_out: bias_out.to_vec(), bias_shape_out: bias_shape_out.to_vec(), is_bias_out: true,
            params_for_basics_vec: param_vec
        };
        if j == 0 {
            trans12_params = Some(params);
        } else if j == 1 {
            trans22_params = Some(params);
        } else {
            trans32_params = Some(params);
        }
    }
    let (upsample2_conv, _) = input(r"C:\study\coursework\src\trash\test_unet_upblocks_crossattnupblock1_upsample.safetensors".to_string()).unwrap();
    let crossattnupblock_2_params = CrossAttnUpBlock2D_params {
        params_for_transformer1: trans12_params.unwrap(), 
        params_for_transformer2: trans22_params.unwrap(),
        params_for_transformer3: trans32_params.unwrap(),
        params_for_resnet1: resnet12_params.unwrap(),
        params_for_resnet2: resnet22_params.unwrap(),
        params_for_resnet3: resnet32_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 1, kernel_size: 3, kernel_weights: upsample2_conv.to_vec(),
        hidden_states: Rc::clone(&res_hidden_states)
    };

    let (unet_conv_in, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_in.safetensors")).unwrap();
    let (unet_conv_out, _) = input(format!(r"C:\study\coursework\src\trash\test_unet_conv_out.safetensors")).unwrap();


    let (l1w, l1ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_w.safetensors")).unwrap();
    let (l1b, l1bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l1_b.safetensors")).unwrap();
    let (l2w, l2ws) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_w.safetensors")).unwrap();
    let (l2b, l2bs) = input(format!(r"C:\study\coursework\src\trash\test_unet_temb_l2_b.safetensors")).unwrap();

    let mut unet = Unet2dConditionModel::unet2d_condition_model_constr(0., Rc::clone(&time_emb), 
        l1w.to_vec(), l1ws.to_vec(), l1b.to_vec(), l1bs.to_vec(), true, 
        l2w.to_vec(), l2ws.to_vec(), l2b.to_vec(), l2bs.to_vec(), true, 

        4, 320, 1, 1, 3, unet_conv_in.to_vec(), 
        res1_params_down, res2_params_down, 
        320, 320, 1, 2, 3, downblock2d_downsample.to_vec(),
        crossattndownblock2d_1_params, 
        crossattndownblock2d_2_params, 

        crossattnupblock_1_params, 
        crossattnupblock_2_params, 
        res1_params_up, 
        res2_params_up, 
        res3_params_up, 
        Rc::clone(&res_hidden_states), 

        params_trans_mid, 
        resnet1_params_mid.unwrap(), 
        resnet2_params_mid.unwrap(), 
        32, 1e-05, 1., 0., 
        320, 4, 1, 1, 3, unet_conv_out.to_vec());

    let (input_vec, input_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_input.safetensors")).unwrap();
    let (res_vec, res_vec_shape) = unet.operation((input_vec.to_vec(), input_vec_shape.to_vec()), 0.).unwrap();
    let (py_vec, py_vec_shape) = input(format!(r"C:\study\coursework\src\trash\test_unet_output.safetensors")).unwrap();
    assert!(res_vec_shape == py_vec_shape.to_vec());
    for i in 0..py_vec.len() {
        let d = (res_vec[i] - py_vec[i]).abs();
        assert!(!d.is_nan());
        assert!(d <= 1e-04);
    }
}