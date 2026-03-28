use ndarray::{Array1, Array2};

use crate::mesh::builder::Builder;

#[allow(non_snake_case)]
pub struct Mesh {
    pub T: usize,
    pub n_neurons: usize,
    pub n_conns_neuron: usize,
    pub beta: f32,
    pub tau_pre: f32,  // Amount to decay pre spikes. Amount of memory to retain
    pub tau_post: f32, // Amount to decay post spikes. Amount of memory to retain
    pub refactory: Array1<u32>, // Store time left for refactory period for each neuron

    pub pre_trace: Array1<f32>, // Output neurons after WTA in previous layer
    pub post_trace: Array1<f32>, // Output neurons after WTA in this layer

    pub neurons: Array1<f32>, // Each layer has neuron with single value. Membrane voltage
    pub conns: Array2<usize>, // Each layer has neuron with multiple conns
    pub weights: Array2<f32>, // Each layer has neuron with multiple weights

    pub thresholds: Array1<f32>, // Each layer has neuron with single threshold
    pub firing_rates: Array1<f32>, // Each layer has neuron with moving average of fires

    pub wta_k: usize,

    pub a_plus: f32,  // How much to change weight when potentionting
    pub a_minus: f32, // How much to change weight when depressionting
    pub w_min: f32,   // Min weight
    pub w_max: f32,   // Max weight

    pub learn: bool,
    pub fired: usize,
    pub n_winners: u32,
}

impl Default for Mesh {
    fn default() -> Self {
        let n_neurons = 100_000;
        let n_conns_neuron = 100;

        Self {
            n_neurons,
            n_conns_neuron,
            beta: 0.95,
            tau_pre: 20.0,
            tau_post: 20.0,
            refactory: Array1::zeros(n_neurons),
            pre_trace: Array1::zeros(n_neurons),
            post_trace: Array1::zeros(n_neurons),
            neurons: Array1::zeros(n_neurons),
            conns: Array2::zeros((n_neurons, n_conns_neuron)),
            weights: Array2::from_elem((n_neurons, n_conns_neuron), 0.2),
            thresholds: Array1::from_elem(n_neurons, 1.0),
            firing_rates: Array1::zeros(n_neurons),
            wta_k: 10,
            a_plus: 0.01,
            a_minus: 0.01,
            w_min: 0.0,
            w_max: 1.0,
            T: 10,
            learn: true,
            fired: 0,
            n_winners: 0,
        }
    }
}

impl Mesh {
    pub fn new(n_neurons: usize, n_conns_neuron: usize) -> Self {
        Self {
            n_neurons,
            n_conns_neuron,
            beta: 0.0,
            tau_pre: 0.0,
            tau_post: 0.0,
            refactory: Array1::zeros(n_neurons),
            pre_trace: Array1::zeros(n_neurons),
            post_trace: Array1::zeros(n_neurons),
            neurons: Array1::zeros(n_neurons),
            conns: Array2::zeros((n_neurons, n_conns_neuron)),
            weights: Array2::from_elem((n_neurons, n_conns_neuron), 0.2),
            thresholds: Array1::from_elem(n_neurons, 1.0),
            firing_rates: Array1::zeros(n_neurons),
            wta_k: 0,
            a_plus: 0.0,
            a_minus: 0.0,
            w_min: 0.0,
            T: 0,
            w_max: 0.0,
            learn: true,
            fired: 0,
            n_winners: 0,
        }
    }

    pub fn run(&mut self) {}
}

impl Builder for Mesh {}
