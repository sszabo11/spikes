use std::collections::HashMap;

use colored::Colorize;

use ndarray::{Array1, Array2};

use crate::spiking::layer::{LayerConfig, SpikingLayer};

#[allow(non_snake_case)]
pub struct SpikingNetwork {
    n_layers: usize,
    n_conns: usize,
    n_neurons: usize,
    pub layers: Vec<SpikingLayer>,

    pub tau_pre: f32, // Decay rate for pre trace
    pub tau_post: f32,
    pub w_plus: f32,  // Weight update when strengthed
    pub w_minus: f32, // Weight update when weakened

    pub w_min: f32, // Min weight value
    pub w_max: f32, // Max weight value

    pub learn: bool,
    pub output_winner_map: HashMap<usize, u32>,
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

    tau_post: f32,
    tau_pre: f32,
    threshold: f32,
    learn: bool,
    a_plus: f32,
    a_minus: f32,

    pub T: usize, // Timesteps
}

impl SpikingBuilder {
    pub fn input_layer(
        mut self,
        num_neurons: usize,
        num_conns: usize,
        input_len: usize,
        k: usize,
    ) -> Self {
        assert!(self.threshold != -10.0, "Please specify threshold");
        assert!(self.n_layers == 0 && self.layers.is_empty());
        assert!(self.tau_pre != 0.0, "Please specify `tau_pre`");
        assert!(self.tau_post != 0.0, "Please specify `tau_post`");

        println!("tay pre: {}", self.tau_pre);
        self.layers.push(SpikingLayer::new(LayerConfig {
            in_n: input_len,
            out_n: num_neurons,
            num_conns,
            tau_pre: self.tau_pre,
            tau_post: self.tau_post,
            threshold: self.threshold,
            top_k: k,
            learn: self.learn,
            id: 1,
            beta: self.beta,
            w_max: 1.5,
            w_min: 0.0,
            a_minus: self.a_minus,
            a_plus: self.a_plus,
        }));

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
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
    pub fn layer(mut self, num_neurons: usize, num_conns: usize, k: usize) -> Self {
        let in_neurons = self.layers[self.n_layers - 1].out_n;
        assert!(self.tau_pre != 0.0, "Please specify `tau_pre`");
        assert!(self.tau_post != 0.0, "Please specify `tau_post`");
        self.layers.push(SpikingLayer::new(LayerConfig {
            in_n: in_neurons,
            out_n: num_neurons,
            num_conns,
            tau_pre: self.tau_pre,
            tau_post: self.tau_post,
            threshold: self.threshold,
            top_k: k,
            id: self.n_layers + 1,
            learn: self.learn,
            beta: self.beta,
            w_max: 1.0,
            w_min: 0.0,
            a_minus: self.a_minus,
            a_plus: self.a_plus,
        }));

        self.n_layers += 1;
        self.n_neurons += num_neurons;
        self.n_conns += num_conns;

        self
    }

    pub fn tau_pre(mut self, tau_pre: f32) -> Self {
        self.tau_pre = tau_pre;
        self
    }
    pub fn tau_post(mut self, tau_post: f32) -> Self {
        self.tau_post = tau_post;
        self
    }

    pub fn build(self) -> SpikingNetwork {
        assert!(
            self.T > 0,
            "Please specify number of timesteps. Example: `builder.timesteps(10)`"
        );

        assert!(
            self.tau_pre > 0.0,
            "`tay_pre` controls decay rate for pre_trace and must be positive"
        );
        assert!(
            self.tau_post > 0.0,
            "`tay_post` controls decay rate for post_trace and must be positive"
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
            learn: self.learn,
            w_max: 1.0,
            output_winner_map: HashMap::new(),
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
            threshold: -10.0, // Starting threshold so user specifies
            layers: vec![],
            beta: 0.9,
            learn: true,
            tau_post: 0.0,
            a_plus: 0.01,
            a_minus: 0.015,
            tau_pre: 0.0,
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

    // Run model. Input is spike train over T steps
    pub fn run(&mut self, input: Array2<usize>) -> Array1<f32> {
        let input = input.mapv(|v| v as f32);
        let mut spike_counts: Array1<f32> = Array1::zeros(self.layers.last().unwrap().out_n);

        for t in 0..input.nrows() {
            let mut pre_spikes = input.row(t).to_owned();
            //println!("T: {}", t);

            for l in 0..self.n_layers {
                let layer = self.layers.get_mut(l).unwrap();
                //println!(
                //    "Layer {}: {} in neurons | {} out neurons",
                //    l, layer.in_n, layer.out_n
                //);
                layer.learn = self.learn;
                pre_spikes = layer.step(&pre_spikes);
            }
            spike_counts += &pre_spikes

            // Print and record
            //self.record_outputs();
            //self.print_output_layer()
        }

        spike_counts
    }

    pub fn get_output_active_neuron(&self) -> usize {
        let output_layer = self.get_output_layer();

        let vec_n = output_layer.to_vec();
        let mut ns: Vec<(usize, &f32)> = vec_n.iter().enumerate().collect();

        ns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        ns[0].0
    }

    pub fn record_outputs(&mut self) {
        let output_layer = self.get_output_layer();

        let vec_n = output_layer.to_vec();
        let mut ns: Vec<(usize, &f32)> = vec_n.iter().enumerate().collect();

        ns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let winner_idx = ns[0].0;

        for (n, v) in output_layer.iter().enumerate() {
            if n == winner_idx {
                let prev = self.output_winner_map.get(&n).unwrap_or(&0);

                self.output_winner_map.insert(n, prev + 1);
            }
        }
    }

    pub fn print_output_layer(&mut self) {
        let output_layer = self.get_output_layer();

        let vec_n = output_layer.to_vec();
        let mut ns: Vec<(usize, &f32)> = vec_n.iter().enumerate().collect();

        ns.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let winner_idx = ns[0].0;

        for (n, v) in output_layer.iter().enumerate() {
            if n == winner_idx {
                //println!("{}: {}", n, v.to_string().green());
                let prev = self.output_winner_map.get(&n).unwrap_or(&0);

                self.output_winner_map.insert(n, prev + 1);
            } else {
                //println!("{}: {}", n, v);
            }
        }
    }
    //pub fn forward(&mut self) {
    //    for l in 1..self.n_layers {
    //        let (prev_layer, layer) = self.layers.split_at_mut(l);

    //        let prev_layer = &mut prev_layer[prev_layer.len() - 1];
    //        println!("{}: {:?}", "prev layer".blue(), prev_layer.neurons);
    //        let layer = &mut layer[0];

    //        'n: for i in 0..layer.out_n {
    //            // Get connections from current layer to previous layer
    //            let connections = layer.conns.row(i);
    //            let mut weights = layer.weights.row_mut(i);
    //            let threshold = layer.thresholds[i];

    //            //println!(
    //            //    "Prev neurons: ({}, {}) | Curr neurons: {}",
    //            //    prev_layer.out_n, layer.in_n, layer.out_n
    //            //);
    //            for (j, &conn_idx) in connections.iter().enumerate() {
    //                //println!("Neuron {}: {}v", i, layer.neurons[i]);
    //                let weight = weights[j];

    //                let sender_voltage = prev_layer.neurons[conn_idx];
    //                // Add to voltage
    //                //assert!(l == 1 && sender_voltage != 1.0 || sender_voltage != 0.0);
    //                layer.neurons[i] += sender_voltage * weight;

    //                // Fire if connected was above threshold
    //                if layer.neurons[i] > threshold {
    //                    self.fires += 1;
    //                    weights[j] += self.w_plus;
    //                    layer.neurons[i] = 1.0;

    //                    //continue 'n;
    //                }
    //            }
    //        }
    //        self.pretty_print_voltage(l);
    //    }
    //}

    pub fn get_output_layer(&self) -> Array1<f32> {
        self.layers[self.n_layers - 1].neurons.to_owned()
    }

    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.neurons.fill(0.0);
            layer.refactory.fill(0);
            layer.pre_trace.fill(0.0);
            layer.post_trace.fill(0.0);
        }
    }
    //pub fn reset_all(&mut self) {
    //    for layer in self.layers.iter_mut() {
    //        layer.neurons.fill(0.0);
    //    }
    //}

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

    pub fn get_output_winner(&self) -> (usize, u32) {
        let mut vec: Vec<(&usize, &u32)> = self.output_winner_map.iter().collect();
        vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        (*vec[0].0, *vec[0].1)
    }

    pub fn print_details(&self) {
        let fired: usize = self.layers.iter().map(|l| l.fired).sum();

        //let winner = self.get_output_winner();

        println!(
            "Neurons: {} | Connections: {} | Threshold: {} | Fired: {} Fired %: {}% ",
            self.n_neurons,
            self.n_conns * self.n_neurons,
            self.layers[0].thresholds[0],
            fired,
            fired as f32 / self.n_neurons as f32 * 100.0 / self.T as f32,
        );
    }
}
