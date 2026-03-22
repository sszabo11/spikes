use colored::Colorize;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::spiking::layer::SpikingLayer;

#[allow(non_snake_case)]
pub struct SpikingNetwork {
    n_layers: usize,
    n_conns: usize,
    n_neurons: usize,
    layers: Vec<SpikingLayer>,

    pub tau_pre: f32, // Decay rate for pre_trace
    pub tau_post: f32,
    pub w_plus: f32,  // Weight update when strengthed
    pub w_minus: f32, // Weight update when weakened

    pub w_min: f32, // Min weight value
    pub w_max: f32, // Max weight value

    pub T: usize,  // Timesteps
    pub beta: f32, // Leak
    fires: usize,
}

#[allow(non_snake_case)]
pub struct SpikingBuilder {
    n_layers: usize,
    layers: Vec<SpikingLayer>,
    beta: f32, // Leak
    n_conns: usize,
    n_neurons: usize,

    pub T: usize, // Timesteps
}

impl SpikingBuilder {
    pub fn input_layer(mut self, num_neurons: usize, num_conns: usize) -> Self {
        assert!(self.n_layers == 0 && self.layers.is_empty());
        self.layers
            .push(SpikingLayer::new(0, num_neurons, num_conns));

        self.n_layers = 1;
        self.n_neurons += num_neurons;
        self.n_conns += num_conns;

        self
    }
    #[allow(non_snake_case)]
    pub fn timesteps(mut self, T: usize) -> Self {
        self.T = T;
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }
    pub fn layer(mut self, num_neurons: usize, num_conns: usize) -> Self {
        let in_neurons = self.layers[self.n_layers - 1].out_n;
        self.layers
            .push(SpikingLayer::new(in_neurons, num_neurons, num_conns));

        self.n_layers += 1;
        self.n_neurons += num_neurons;
        self.n_conns += num_conns;

        self
    }

    pub fn build(self) -> SpikingNetwork {
        assert!(
            self.T > 0,
            "Please specify number of timesteps. Example: `builder.timesteps(10)`"
        );
        assert!(self.n_neurons > 0);
        assert!(self.n_layers > 0);
        assert!(self.n_conns > 0);
        SpikingNetwork {
            tau_pre: 0.0,
            tau_post: 0.0,
            w_plus: 0.05,
            w_min: -1.0,
            T: self.T,
            w_max: 1.0,
            w_minus: 0.08,
            n_layers: self.n_layers,
            n_conns: self.n_conns,
            n_neurons: self.n_neurons,
            fires: 0,
            beta: self.beta,
            layers: self.layers,
        }
    }
}

impl SpikingNetwork {
    pub fn builder() -> SpikingBuilder {
        SpikingBuilder {
            n_layers: 0,
            layers: vec![],
            beta: 0.9,
            n_conns: 0,
            T: 0,
            n_neurons: 0,
        }
    }

    pub fn set_input(&mut self, input: Array1<f32>) {
        assert!(self.layers[0].neurons.len() == input.len());

        for (i, &val) in input.iter().enumerate() {
            self.layers[0].neurons[i] = val;
        }
    }

    pub fn step(&mut self, input: Array1<f32>) {
        // Layer 0 now has t0 valus
        self.set_input(input); // TODO: Spike train

        for l in 1..self.n_layers {
            let (prev_layer, layer) = self.layers.split_at_mut(l);

            let prev_layer = &mut prev_layer[prev_layer.len() - 1];
            let layer = &mut layer[0];
        }
    }

    fn decay(&mut self) {}

    pub fn forward(&mut self) {
        for l in 1..self.n_layers {
            let (prev_layer, layer) = self.layers.split_at_mut(l);

            let prev_layer = &mut prev_layer[prev_layer.len() - 1];
            println!("{}: {:?}", "prev layer".blue(), prev_layer.neurons);
            let layer = &mut layer[0];

            'n: for i in 0..layer.out_n {
                // Get connections from current layer to previous layer
                let connections = layer.conns.row(i);
                let mut weights = layer.weights.row_mut(i);
                let threshold = layer.thresholds[i];

                //println!(
                //    "Prev neurons: ({}, {}) | Curr neurons: {}",
                //    prev_layer.out_n, layer.in_n, layer.out_n
                //);
                for (j, &conn_idx) in connections.iter().enumerate() {
                    //println!("Neuron {}: {}v", i, layer.neurons[i]);
                    let weight = weights[j];

                    let sender_voltage = prev_layer.neurons[conn_idx];
                    // Add to voltage
                    //assert!(l == 1 && sender_voltage != 1.0 || sender_voltage != 0.0);
                    layer.neurons[i] += sender_voltage * weight;

                    // Fire if connected was above threshold
                    if layer.neurons[i] > threshold {
                        self.fires += 1;
                        weights[j] += self.w_plus;
                        layer.neurons[i] = 1.0;

                        //continue 'n;
                    }
                }
            }
            self.pretty_print_voltage(l);
        }
    }

    pub fn get_output_layer(&self) -> Array1<f32> {
        self.layers[self.n_layers - 1].neurons.to_owned()
    }

    pub fn reset(&mut self) {
        for i in 1..self.n_layers {
            let layer = &mut self.layers[i];
            layer.neurons.fill(0.0);
        }
    }
    pub fn reset_all(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.neurons.fill(0.0);
        }
    }

    pub fn pretty_print_voltage(&self, layer: usize) {
        let l = &self.layers[layer];

        print!("\nVoltage: [");
        for &n in l.neurons.iter() {
            if n > l.thresholds[0] {
                print!("{}", n.to_string().green());
            } else {
                print!("{}", n);
            }
            print!(", ")
        }
        print!("]\n");
    }

    pub fn print_details(&self) {
        println!(
            "Neurons: {} | Connections: {} | Threshold: {} | Fired: {} Fired %: {}%",
            self.n_neurons,
            self.n_conns * self.n_neurons,
            self.layers[0].thresholds[0],
            self.fires,
            self.fires as f32 / self.n_neurons as f32 * 100.0
        );
    }
}
