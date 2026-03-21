use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub struct SpikingLayer {
    pub in_n: usize,
    pub out_n: usize,
    pub num_conns: usize,
    pub beta: f32,
    pub refactory: Array1<u32>,
    pub neurons: Array1<f32>,      // Each layer has neuron with single value
    pub conns: Array2<usize>,      // Each layer has neuron with multiple conns
    pub weights: Array2<f32>,      // Each layer has neuron with multiple weights
    pub thresholds: Array1<f32>,   // Each layer has neuron with single threshold
    pub firing_rates: Array1<f32>, // Each layer has neuron with moving average of fires
}

impl SpikingLayer {
    pub fn new(in_n: usize, out_n: usize, num_conns: usize) -> Self {
        Self {
            in_n,
            out_n,
            num_conns,
            beta: 0.9,
            //thresholds: Array1::random(out_n, Uniform::new(0.4, 0.8).unwrap()),
            thresholds: Array1::from_elem(out_n, 0.8),
            neurons: Array1::zeros(out_n),
            refactory: Array1::zeros(out_n),
            firing_rates: Array1::zeros(out_n),
            weights: Array2::random((out_n, num_conns), Uniform::new(-1.0, 1.0).unwrap()),
            conns: Array2::random((out_n, num_conns), Uniform::new(0, in_n - 1).unwrap()),
        }
    }

    pub fn decay(&self) {}

    pub fn step(&mut self, pre_spikes: &Array1<f32>) -> Array1<f32> {
        let input = self.weights.dot(pre_spikes);

        for j in 0..self.out_n {
            if self.refactory[j] == 0 {
                self.neurons[j] = self.beta * self.neurons[j] + input[j]
            }
        }

        input
    }
}
