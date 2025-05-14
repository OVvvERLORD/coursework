use std::{fs::File, io::{Read, Write}, sync::Arc};
use ndarray::{self, Axis};
use safetensors::{serialize, SafeTensors};
use rayon::prelude::*;
use ndarray_einsum::einsum;

pub fn Tensor_Mul(A: &mut ndarray::Array4<f32>, B: &ndarray::Array4<f32>) ->  Result<(), Box<dyn std::error::Error>> {
    // let A = ndarray::Array4::from_shape_vec((args.1[0], args.1[1], args.1[2], args.1[3]), args.0)?;
    // let B = ndarray::Array4::from_shape_vec((args.3[0], args.3[1], args.3[2], args.3[3]), args.2)?;
    // let shape = A.shape();
    // // let B = B.broadcast([shape[0], shape[1], shape[2], shape[3]]).unwrap();
    // let B = B.broadcast((shape[0], shape[1], B.shape()[2], B.shape()[3])).unwrap();
    // let mut result_shape = shape.to_vec();
    // result_shape[3] = B.shape()[2];
    // let mut result = ndarray::Array4::<f32>::zeros((result_shape[0], result_shape[1], result_shape[2], result_shape[3]));
    // for ((mut res, a_batch), b_batch) in result
    //     .axis_iter_mut(ndarray::Axis(0))
    //     .zip(A.axis_iter(ndarray::Axis(0)))
    //     .zip(B.axis_iter(ndarray::Axis(0)))
    // { 
    //     for ((mut res_slice, a_slice), b_slice) in res
    //         .axis_iter_mut(ndarray::Axis(0))
    //         .zip(a_batch.axis_iter(ndarray::Axis(0)))
    //         .zip(b_batch.axis_iter(ndarray::Axis(0)))
    //     {
    //         let b_transposed = b_slice.view().reversed_axes();
    //         res_slice.assign(&a_slice.dot(&b_transposed));
    //     }
    // }
    // *A = result;
    // let shape_a = A.dim();
    // let temp_a = A
    // .view()
    // .into_shape_with_order([shape_a.0 * shape_a.1 * shape_a.2, shape_a.3])
    // .unwrap();

    // let mut shape_b = B.shape();
    // let temp_b = B
    // .view()
    // .into_shape_with_order([shape_b[0] * shape_b[1] * shape_b[2], shape_b[3]])
    // .unwrap()
    // .reversed_axes();
    // // shape_b = temp_b.shape();
    // print!("\nMUL: {:?} {:?}\n", temp_a.shape(), temp_b.shape());
    // print!("\n{:?}\n", shape_b[2]);
    // *A = temp_a.dot(&temp_b).into_shape_with_order([shape_a.0, shape_a.1, shape_a.2, shape_b[2]]).unwrap();
    // print!("\nMUL ITOG :{:?}\n", A.shape());

    // let a_shape = A.dim();
    // let b_shape = B.dim();
    // let temp_b = B
    // .clone()
    // .permuted_axes([0, 1, 3, 2])
    // .broadcast([a_shape.0, a_shape.1, b_shape.3, b_shape.2])
    // .unwrap()
    // .to_owned();

    // *A = einsum("bchw,bcws->bchs", &[A, &temp_b])
    // .unwrap()
    // .into_dimensionality()
    // .unwrap();


    let shape = A.shape();
    // let B = B.broadcast([shape[0], shape[1], shape[2], shape[3]]).unwrap();
    let B = B.broadcast((shape[0], shape[1], B.shape()[2], B.shape()[3])).unwrap();
    let mut result_shape = shape.to_vec();
    result_shape[3] = B.shape()[2];
    let mut result = ndarray::Array4::<f32>::zeros((result_shape[0], result_shape[1], result_shape[2], result_shape[3]));
    result
    .axis_iter_mut(ndarray::Axis(0))
    .into_par_iter()
    .zip(A.axis_iter(ndarray::Axis(0)).into_par_iter())
    .zip(B.axis_iter(ndarray::Axis(0)).into_par_iter())
    .for_each(|((mut batch_res, batch_a), batch_b)| {
        batch_res
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .zip(batch_a.axis_iter(ndarray::Axis(0)).into_par_iter())
        .zip(batch_b.axis_iter(ndarray::Axis(0)).into_par_iter())
        .for_each(
            |((mut ch_res, ch_a), ch_b)| {
                let b_trans = ch_b.view().reversed_axes();
                ch_res.assign(&ch_a.dot(&b_trans));
            }
        );
    }
    );
    *A = result;


    Ok(())
}

pub fn input (input_name: String) -> Result<ndarray::Array4<f32>, Box<dyn std::error::Error>> { 
    let mut file = File::open(input_name.to_string())?;
    let mut buffer = Vec::<u8>::new();
    let _ = file.read_to_end(&mut buffer);

    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut input_vec = Vec::<f32>::new();
    let mut input_vec_shape = Vec::<usize>::new();
    for (_, tensor) in tensors.tensors() {
        input_vec = bytemuck::cast_slice(tensor.data()).to_vec();
        input_vec_shape = tensor.shape().to_vec();
    }
    if input_vec_shape.len() == 3 {
        input_vec_shape.insert(0, 1);
    }
    if input_vec_shape.len() == 2 {
        input_vec_shape.insert(0, 1);
        input_vec_shape.insert(0, 1);
    }
    if input_vec_shape.len() == 1 {
        input_vec_shape.insert(0, 1);
        input_vec_shape.insert(0, 1);
        input_vec_shape.insert(0, 1);
    }
    // if input_vec_shape.len() == 3 {
    //     input_vec_shape.insert(0, 1);
    // }
    let tensor = ndarray::Array4::from_shape_vec([input_vec_shape[0], input_vec_shape[1], input_vec_shape[2], input_vec_shape[3]], input_vec).unwrap();

    Ok(tensor)
}

pub fn output(output_name: String, tensor_vec:Vec<f32>, shape_vec:Vec<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let binding = tensor_vec.to_vec();
    let prep_output:&[u8] = bytemuck::cast_slice(&binding);
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("output_tensor".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape_vec.clone(), prep_output)?);
    let serialized_data = serialize(&tensors, &None)?;
    let mut file = File::create(output_name.to_string())?;
    file.write_all(&serialized_data)?;
    Ok(())
}

pub fn nearest_neighbour_interpolation (args:&mut ndarray::Array4<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let input_vec_shape = args.shape();
    let mut output_tensor = ndarray::Array4::<f32>::zeros((input_vec_shape[0], input_vec_shape[1], input_vec_shape[2] * 2, input_vec_shape[3] * 2));
    for batches in 0..input_vec_shape[0] {
        for channels in 0..input_vec_shape[1] {
            for height in 0..input_vec_shape[2] * 2 {
                for width in 0..input_vec_shape[3] * 2 {
                    let round_height = height / 2;
                    let round_width = width / 2;
                    output_tensor[[batches, channels, height, width]] = args[[batches, channels, round_height, round_width]];
                }
            }
        }
    }
    *args = output_tensor;
    Ok(())
}

pub fn scalar_timestep_embedding(timestep: f32, batch_size: usize, dim: usize) ->  Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let mut scalar = ndarray::Array1::from_shape_vec([1], [timestep].to_vec()).unwrap();
    scalar = scalar.broadcast(batch_size)
    .unwrap().
    to_owned(); // timestep -> tensor[timestep,..., timestep] 1D 
    let half_dim = (dim / 2) as f32;
    let mut exponent_arr = ndarray::Array1::from_iter(ndarray::range(0., half_dim, 1.));
    exponent_arr = exponent_arr * ( - f32::ln( 10000.0));
    exponent_arr = exponent_arr / (half_dim - 1.);

    for i in 0..exponent_arr.len() {
        exponent_arr[i] = f32::exp(exponent_arr[i]);
    }
    
    let timesteps_2d = scalar.insert_axis(ndarray::Axis(1));
    let emb_2d = exponent_arr.insert_axis(ndarray::Axis(0));
    let emb = timesteps_2d * emb_2d;
    let mut emb_sin = emb.clone();
    let mut emb_cos = emb;
    for batch in 0..emb_sin.shape()[0] {
        for i in 0..emb_sin.shape()[1] {
            emb_sin[(batch, i)] = f32::sin(emb_sin[(batch, i)]);
            emb_cos[(batch, i)] = f32::cos(emb_cos[(batch, i)]);
        }
    }

    let result = ndarray::
    concatenate(ndarray::Axis(1), &[emb_cos.view(), emb_sin.view()])
    .unwrap();
    let mut res_vec_shape = result.shape().to_vec();
    if res_vec_shape.len() == 2 {
        res_vec_shape.insert(0, 1);
        res_vec_shape.insert(0, 1);
    }
    let res_vec = result.as_standard_layout()
    .to_owned()
    .into_raw_vec_and_offset().0;
    Ok((res_vec, res_vec_shape))
}