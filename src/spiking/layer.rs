use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub struct SpikingLayer {
    pub in_n: usize,
    pub out_n: usize,
    pub num_conns: usize,
    pub beta: f32,
    pub id: usize,

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

    pub fired: usize,
}

pub struct LayerConfig {
    pub in_n: usize,
    pub out_n: usize,
    pub num_conns: usize,
    pub tau_pre: f32,
    pub tau_post: f32,
    pub threshold: f32,
    pub top_k: usize,
    pub id: usize,
}

impl SpikingLayer {
    pub fn new(config: LayerConfig) -> Self {
        Self {
            in_n: config.in_n,
            wta_k: config.top_k,
            out_n: config.out_n,
            fired: 0,
            num_conns: config.num_conns,
            id: config.id,
            tau_pre: config.tau_pre,
            tau_post: config.tau_post,
            beta: 0.9,
            pre_trace: Array1::zeros(config.in_n),
            post_trace: Array1::zeros(config.out_n),
            //thresholds: Array1::random(out_n, Uniform::new(0.4, 0.8).unwrap()),
            thresholds: Array1::from_elem(config.out_n, config.threshold),
            neurons: Array1::zeros(config.out_n),
            refactory: Array1::zeros(config.out_n),
            firing_rates: Array1::zeros(config.out_n),
            weights: Array2::random(
                (config.out_n, config.num_conns),
                Uniform::new(-1.0, 1.0).unwrap(),
            ),
            conns: Array2::random(
                (config.out_n, config.num_conns),
                Uniform::new(0, config.in_n - 1).unwrap(),
            ),
        }
    }

    pub fn step(&mut self, pre_spikes: &Array1<f32>) -> Array1<f32> {
        assert!(
            pre_spikes.len() == self.in_n,
            "Input spikes is not correct Expeteced: {} Got: {}. Layer {}",
            self.in_n,
            pre_spikes.len(),
            self.id
        );
        // Decay
        self.pre_trace *= self.tau_pre;
        self.post_trace *= self.tau_post;

        let mut input: Array1<f32> = Array1::zeros(self.out_n);

        // Membrane update
        for i in 0..self.out_n {
            let idxs = self.conns.row(i);

            for (j, &conn_idx) in idxs.iter().enumerate() {
                let weight = self.weights[[i, j]];
                //println!("Spike: {} | Weight: {}", pre_spikes[conn_idx], weight);
                input[i] += pre_spikes[conn_idx] * weight;
            }
            //println!("Neuron membrane: {}v", input[i]);

            // If not in refactory update
            if self.refactory[i] == 0 {
                self.neurons[i] = self.beta * self.neurons[i] + input[i];
            }
        }

        self.repolarize_neurons();

        self.wta();

        self.neurons.clone()
    }

    fn repolarize_neurons(&mut self) {
        for i in 0..self.out_n {
            self.refactory[i] = self.refactory[i].saturating_sub(1);
        }
    }

    // Winner takes all. Selects neurons above threshold and picks top k winners. Computed before post/pre trace.
    pub fn wta(&mut self) {
        let mut neurons: Vec<(usize, f32)> =
            self.neurons.to_vec().into_iter().enumerate().collect();

        neurons.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        //let winners: Vec<(usize, f32)> = neurons.into_iter().take(self.wta_k).collect();

        let mut wns = Vec::new();
        for (i, v) in neurons.into_iter() {
            if v >= self.thresholds[i] {
                wns.push(i);
                self.refactory[i] = 3;
                self.fired += 1;
            }
            if wns.len() >= self.wta_k || v < self.thresholds[i] {
                break;
            }
        }
        assert!(wns.len() <= self.wta_k);

        self.neurons.fill(0.0);

        println!("N Winners: {}", wns.len());
        // Set winners to 1.0
        for idx in wns {
            self.neurons[idx] = 1.0
        }
    }
}
