use crate::{
    layers::{
        layer::Layer,
        params::{
            Resnet2d_params,
            CrossAttnDownBlock2D_params,
            BasicTransofmerBlock_params,
            Transformer2D_params
        },
    
    },
    func::functions::input,
    blocks::{
        downblock::DownBlock2D,
        attn::CrossAttnDownBlock2D
    }
};
use std::rc::Rc;
use std::cell::RefCell;

pub struct Down_blocks {
    pub operations : Vec<Box<dyn Layer>>,
    pub hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
}

impl Down_blocks {
    pub fn new (
        params_for_resnet1 : Resnet2d_params,
        params_for_resnet2 : Resnet2d_params,
        in_channels : usize, 
        out_channels : usize, 
        padding : i32, 
        stride : i32, 
        kernel_size : usize, 
        kernel_weights : Vec<f32>,
        bias: ndarray::Array4<f32>,
        is_bias: bool,
        params_for_crattbl1 : CrossAttnDownBlock2D_params,
        params_for_crattbl2 : CrossAttnDownBlock2D_params,
        hidden_states : Rc<RefCell<Vec<ndarray::Array4<f32>>>>
    ) -> Self {
        let mut vec = Vec::<Box<dyn Layer>>::new();
        let downblock = DownBlock2D::new(
            params_for_resnet1, 
            params_for_resnet2, 
            in_channels, 
            out_channels, 
            padding, 
            stride, 
            kernel_size, 
            kernel_weights, 
            bias, 
            is_bias, 
            Rc::clone(&hidden_states)
        );
        vec.push(Box::new(downblock));
        let crossattnblock1 = CrossAttnDownBlock2D::new(params_for_crattbl1);
        vec.push(Box::new(crossattnblock1));
        let crossattnblock2 = CrossAttnDownBlock2D::new(params_for_crattbl2);
        vec.push(Box::new(crossattnblock2));
        Self { operations: vec, hidden_states: Rc::clone(&hidden_states) }
    }
}

impl Layer for Down_blocks {
    fn operation(&self,  args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let operations = &self.operations;
        {
            let mut hidden_states = self.hidden_states.borrow_mut();
            hidden_states.push(args.clone());
        }
        for layer in operations {
            let _ = layer.operation(args)?;
        } 

        Ok(())
    }
}

#[test]
fn test_downblocks() {
    let temb = input(format!(r"C:\study\coursework\src\trash\test_downblocks_temb.safetensors")).unwrap();
    let encoder = input(format!(r"C:\study\coursework\src\trash\test_downblocks_encoder.safetensors")).unwrap();
    let mut res_hidden_states = Rc::new(RefCell::new(Vec::<ndarray::Array4<f32>>::new()));
    let time_emb = Rc::new(RefCell::new(temb));
    let encoder = Rc::new(RefCell::new(encoder.remove_axis(ndarray::Axis(0))));
    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res0_linear_bias.safetensors".to_string()).unwrap();
    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors".to_string()).unwrap();
    let res1_params = Resnet2d_params{
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
        time_emb: Rc::clone(&time_emb)
    };

    let kernel1 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight.safetensors".to_string()).unwrap();
    let kernel2 = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight.safetensors".to_string()).unwrap();
    let c1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv1_weight_b.safetensors".to_string()).unwrap();
    let c2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_conv2_weight_b.safetensors".to_string()).unwrap();

    let norm1_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight.safetensors".to_string()).unwrap();
    let norm2_w = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight.safetensors".to_string()).unwrap();
    let norm1_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm1_weight_b.safetensors".to_string()).unwrap();
    let norm2_b = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_norm2_weight_b.safetensors".to_string()).unwrap();

    let linear_weight= input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_weight.safetensors".to_string()).unwrap();
    let linear_bias = input(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_res1_linear_bias.safetensors".to_string()).unwrap();

    let res2_params = Resnet2d_params{
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
        time_emb: Rc::clone(&time_emb)
    };

    let kernel_down = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample.safetensors".to_string()).unwrap();
    let cdown_b = input(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_downsample_b.safetensors".to_string()).unwrap();
    let mut trans1_params : Option<Transformer2D_params> = None;
    let mut trans2_params : Option<Transformer2D_params> = None;
    let mut resnet1_params : Option<Resnet2d_params> = None;
    let mut resnet2_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1.safetensors", i)).unwrap();
        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2.safetensors", i)).unwrap();
        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv1_b.safetensors", i)).unwrap();
        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv2_b.safetensors", i)).unwrap();
        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1.safetensors", i)).unwrap();
        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2.safetensors", i)).unwrap();
        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm1_b.safetensors", i)).unwrap();
        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_norm2_b.safetensors", i)).unwrap();
        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};
        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {320} else {640};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
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
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 640, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
            time_emb: Rc::clone(&time_emb)
    };
        if i == 0 {
            resnet1_params = Some(resnet_par);
        } else {
            resnet2_params = Some(resnet_par);
        } 
    }
    for j in 0..2 {
        let mut param_vec = Vec::<BasicTransofmerBlock_params>::new();
        for i in 0..2 {
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();



            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 640, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 640, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 640, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder), if_encoder_tensor_1 : false, number_of_heads_1: 10, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder), if_encoder_tensor_2 : true, number_of_heads_2: 10, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out= input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock1_trans{}_projout_b_test.safetensors", j)).unwrap(); 
        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
        };
        if j == 0 {
            trans1_params = Some(params);
        } else{
            trans2_params = Some(params);
        }
    }
    let crossattndownblock2d_1_params = CrossAttnDownBlock2D_params {
        is_downsample2d: true,
        params_for_transformer1: trans1_params.unwrap(),
        params_for_transformer2: trans2_params.unwrap(),
        params_for_resnet1: resnet1_params.unwrap(),
        params_for_resnet2: resnet2_params.unwrap(),
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 3, kernel_weights: kernel_down.into_raw_vec_and_offset().0,
        is_bias: true, bias: cdown_b,
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let mut trans12_params : Option<Transformer2D_params> = None;
    let mut trans22_params : Option<Transformer2D_params> = None;
    let mut resnet12_params : Option<Resnet2d_params> = None;
    let mut resnet22_params : Option<Resnet2d_params> = None;

    for i in 0..2 {
        let kernel1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1.safetensors", i)).unwrap();

        let kernel2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2.safetensors", i)).unwrap();

        let c1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv1_b.safetensors", i)).unwrap();

        let c2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv2_b.safetensors", i)).unwrap();

        let norm1_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1.safetensors", i)).unwrap();

        let norm2_w = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2.safetensors", i)).unwrap();

        let norm1_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm1_b.safetensors", i)).unwrap();

        let norm2_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_norm2_b.safetensors", i)).unwrap();

        let kernels = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short.safetensors", i)).unwrap()}
        else
        {kernel1.clone()};
        let cs_b = if i == 0 {input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_conv_short_b.safetensors", i)).unwrap()}
        else
        {c1_b.clone()};

        let linear_weight = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_w.safetensors", i)).unwrap();
        let linear_bias = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_resnet{}_temb_b.safetensors", i)).unwrap();
        let in_ch = if i == 0 {640} else {1280};
        let shortcut_flag = if i == 0 {true} else {false};
        let resnet_par = Resnet2d_params{
            number_of_groups_1: 32, eps_1: 1e-05, gamma_1: norm1_w, beta_1: norm1_b,
            in_channels_1: in_ch, 
            out_channels_1: 1280, 
            padding_1: 1, 
            stride_1 : 1, 
            kernel_size_1 : 3, 
            kernel_weights_1: kernel1.into_raw_vec_and_offset().0, 
            bias_c1: c1_b, is_bias_c1: true,
            weights: linear_weight, bias : linear_bias, is_bias: true,
            number_of_groups_2: 32, eps_2: 1e-05, gamma_2: norm2_w, beta_2: norm2_b,
            in_channels_2: 1280, 
            out_channels_2: 1280, 
            padding_2: 1, stride_2 : 1, 
            kernel_size_2 : 3, 
            kernel_weights_2: kernel2.into_raw_vec_and_offset().0,
            bias_c2: c2_b, is_bias_c2: true,
            is_shortcut: shortcut_flag,
            in_channels_short: in_ch, out_channels_short: 1280, padding_short: 0, stride_short : 1, kernel_size_short : 1, kernel_weights_short: kernels.into_raw_vec_and_offset().0,
            bias_s: cs_b, is_bias_s: true,
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
            let weights_1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_q_test.safetensors", j, i)).unwrap();
            let weights_2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_k_test.safetensors", j, i)).unwrap();
            let weights_3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_v_test.safetensors", j, i)).unwrap();
            let weights_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_4 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn1_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_5 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_q_test.safetensors", j, i)).unwrap();
            let weights_6 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_k_test.safetensors", j, i)).unwrap();
            let weights_7 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_v_test.safetensors", j, i)).unwrap();
            let weights_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_w_test.safetensors", j, i)).unwrap(); 
            let bias_8 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_attn2_out_b_test.safetensors", j, i)).unwrap(); 
        
            let weights_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_w_test.safetensors", j, i)).unwrap();
            let bias_ff1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_geglu_b_test.safetensors", j, i)).unwrap();
        
            let weights_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_w_test.safetensors", j, i)).unwrap();
            let bias_ff2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_ff_b_test.safetensors", j, i)).unwrap();
        
            let gamma1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_w_test.safetensors", j, i)).unwrap();
            let gamma2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_w_test.safetensors", j, i)).unwrap();
            let gamma3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_w_test.safetensors", j, i)).unwrap();

            let beta1 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm1_b_test.safetensors", j, i)).unwrap();
            let beta2 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm2_b_test.safetensors", j, i)).unwrap();
            let beta3 = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_btb{}_norm3_b_test.safetensors", j, i)).unwrap();


            let btb1_params = BasicTransofmerBlock_params {
            eps_1 : 1e-05, gamma_1 : gamma1, beta_1 : beta1, number_1 : 1280, // LayerNorm 
            eps_2 : 1e-05, gamma_2 : gamma2, beta_2 : beta2, number_2 : 1280, // LayerNorm 
            eps_3 : 1e-05, gamma_3 : gamma3, beta_3 : beta3, number_3 : 1280, // LayerNorm 
            weights_1: weights_1, bias_1: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_1 : false,  // Attn1
            weights_2: weights_2, bias_2: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_2 : false,
            weights_3: weights_3, bias_3: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_3 : false,
            weights_4: weights_4, bias_4: bias_4, is_bias_4 : true,
            encoder_hidden_tensor_1 : Rc::clone(&encoder), if_encoder_tensor_1 : false, number_of_heads_1: 20, 

            weights_5: weights_5, bias_5: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_5 : false,  // Attn2
            weights_6: weights_6, bias_6: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_6 : false,
            weights_7: weights_7, bias_7: ndarray::Array4::from_elem((1,1,1,1), 1.), is_bias_7 : false,
            weights_8: weights_8, bias_8: bias_8, is_bias_8 : true,
            encoder_hidden_tensor_2 : Rc::clone(&encoder), if_encoder_tensor_2 : true, number_of_heads_2: 20, 

            weights_ff1: weights_ff1, bias_ff1: bias_ff1, is_bias_ff1 : true, // FeedForward
            weights_ff2: weights_ff2, bias_ff2: bias_ff2, is_bias_ff2 : true,
            };
            param_vec.push(btb1_params);
        }
        let weights_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_w_test.safetensors", j)).unwrap(); 
        let bias_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projin_b_test.safetensors", j)).unwrap(); 

        let weights_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_w_test.safetensors", j)).unwrap(); 
        let bias_out = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_projout_b_test.safetensors", j)).unwrap(); 

        let gamma_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_w_test.safetensors", j)).unwrap(); 
        let beta_in = input(format!(r"C:\study\coursework\src\trash\test_downblocks_crossattndownblock2_trans{}_norm_b_test.safetensors", j)).unwrap(); 

        let params = Transformer2D_params{
            number_of_groups: 32, eps: 1e-06, gamma: gamma_in, beta: beta_in,
        weigths_in: weights_in, bias_in: bias_in, is_bias_in : true,
        weigths_out: weights_out, bias_out: bias_out, is_bias_out : true,
        params_for_basics_vec : param_vec
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
        in_channels: 640, out_channels: 640, padding: 1, stride: 2, kernel_size: 13123124, kernel_weights: vec![1.],
        bias: ndarray::Array4::from_shape_vec([1, 1, 1, 1], vec![1.]).unwrap(), is_bias: false,
        hidden_states: Rc::clone(&res_hidden_states)
    };
    let kernel_downblock2d_downsample = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample.safetensors")).unwrap();
    let c_downblock2d_downsample_b = input(format!(r"C:\study\coursework\src\trash\test_downblocks_downblock2d_downsample_b.safetensors")).unwrap();
    let down_blocks = Down_blocks::new(
        res1_params, res2_params, 
        320, 320, 1, 2, 3, kernel_downblock2d_downsample.into_raw_vec_and_offset().0,
        c_downblock2d_downsample_b, true,
        crossattndownblock2d_1_params, 
        crossattndownblock2d_2_params, 
        Rc::clone(&res_hidden_states));
    let mut tensor = input(format!(r"C:\study\coursework\src\trash\test_downblocks_input.safetensors")).unwrap();

    let _ = down_blocks.operation(&mut tensor);

    let shape = tensor.shape();
    let py_tensor = input(format!(r"C:\study\coursework\src\trash\test_downblocks_output.safetensors")).unwrap();
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
    let testings = res_hidden_states.borrow();
    assert!(testings.len() == 9);
}