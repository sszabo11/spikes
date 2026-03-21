use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{action::Action, sense::Sense};

pub struct Brain {
    num_neurons: usize,
    num_layers: usize,
    num_conns: usize,
    neurons: Vec<Array2<f32>>,
    conns: Vec<Array2<usize>>,
    weights: Vec<Array2<f32>>,
    senses: Vec<Box<dyn Sense>>,
    actions: Vec<Box<dyn Action>>,
}

impl Brain {
    pub fn new(num_neurons: usize, num_layers: usize, num_conns: usize) -> Self {
        Self {
            num_neurons,
            num_layers,
            actions: vec![],
            senses: vec![],
            num_conns,
            neurons: vec![Array2::zeros((num_layers, num_neurons)); num_layers],
            weights: vec![Array2::random(
                (num_layers, num_neurons),
                Uniform::new(-1.0, 1.0).unwrap(),
            )],
            conns: vec![Array2::random(
                (num_layers, num_neurons),
                Uniform::new(0, 1).unwrap(),
            )],
        }
    }
}
