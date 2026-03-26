use colored::Colorize;
use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

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

    pub a_plus: f32,  // How much to change weight when potentionting
    pub a_minus: f32, // How much to change weight when depressionting
    pub w_min: f32,   // Min weight
    pub w_max: f32,   // Max weight

    pub learn: bool,
    pub fired: usize,
    pub n_winners: u32,
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
    pub learn: bool,
    pub a_plus: f32,
    pub a_minus: f32,
    pub w_min: f32,
    pub beta: f32,
    pub w_max: f32,
}

impl SpikingLayer {
    pub fn new(config: LayerConfig) -> Self {
        Self {
            in_n: config.in_n,
            wta_k: config.top_k,
            out_n: config.out_n,
            a_plus: config.a_plus,
            a_minus: config.a_minus,
            learn: config.learn,
            fired: 0,
            w_min: config.w_min,
            n_winners: 0,
            w_max: config.w_max,
            num_conns: config.num_conns,
            id: config.id,
            tau_pre: config.tau_pre,
            tau_post: config.tau_post,
            beta: config.beta,
            pre_trace: Array1::zeros(config.in_n),
            post_trace: Array1::zeros(config.out_n),
            //thresholds: Array1::random(out_n, Uniform::new(0.4, 0.8).unwrap()),
            thresholds: Array1::from_elem(config.out_n, config.threshold),
            neurons: Array1::zeros(config.out_n),
            refactory: Array1::zeros(config.out_n),
            firing_rates: Array1::zeros(config.out_n),
            weights: Array2::random(
                (config.out_n, config.num_conns),
                Uniform::new(0.15, 0.5).unwrap(),
            ),
            conns: generate_connections(config.out_n, config.num_conns, config.in_n),
            //conns: Array2::random(
            //    (config.out_n, config.num_conns),
            //    Uniform::new(0, config.in_n).unwrap(),
            //),
        }
    }

    pub fn step(&mut self, pre_spikes: &Array1<f32>) -> Array1<f32> {
        // Subtract 1 from each neurons refactory state
        self.repolarize_neurons();
        self.n_winners = 0;
        assert!(
            pre_spikes.len() == self.in_n,
            "Input spikes is not correct. Expected: {} Got: {}. Layer {}",
            self.in_n,
            pre_spikes.len(),
            self.id
        );
        // Decay
        self.pre_trace *= self.tau_pre;
        self.post_trace *= self.tau_post;

        self.pre_trace += pre_spikes;

        let mut input: Array1<f32> = Array1::zeros(self.out_n);

        // Membrane update
        for i in 0..self.out_n {
            let idxs = self.conns.row(i);

            for (j, &conn_idx) in idxs.iter().enumerate() {
                let weight = self.weights[[i, j]];
                //println!("Spike: {} | Weight: {}", pre_spikes[conn_idx], weight);
                input[i] += pre_spikes[conn_idx] * weight;
            }

            //println!("ref: {}", self.refactory[i]);
            // If not in refactory update
            if self.refactory[i] == 0 {
                self.neurons[i] = self.beta * self.neurons[i] + input[i];
            }
        }
        // Converts to binary spikes, and filters by winners + thresholds
        // Returns binary spikes after wta and threshold
        let post_spikes = self.wta();

        // Reset winners
        self.inhibit_winners(&post_spikes);

        self.post_trace += &post_spikes;

        //let post_spikes = self.neurons.clone();
        if self.learn {
            self.stdp(pre_spikes, &post_spikes);
            self.homeostasis(&post_spikes);
        }

        // Post spikes
        post_spikes
    }

    fn homeostasis(&mut self, post_spikes: &Array1<f32>) {
        //let target_rate = 0.05;
        let target_rate = self.wta_k as f32 / self.out_n as f32;
        for j in 0..self.out_n {
            self.firing_rates[j] = 0.99 * self.firing_rates[j] + 0.01 * post_spikes[j];
            let error = self.firing_rates[j] - target_rate;
            //print!("Err: {}", error);
            self.thresholds[j] = (self.thresholds[j] + 0.0001 * error).clamp(0.05, 1.5);
        }
    }

    fn repolarize_neurons(&mut self) {
        //let f = self.refactory.iter().filter(|&r| *r > 0).count();
        //println!("bef: {}", f);
        self.refactory.mapv_inplace(|r| r.saturating_sub(1));
        //let af = self.refactory.iter().filter(|&r| *r > 0).count();

        //println!("af: {}", af);
    }

    fn inhibit_winners(&mut self, post: &Array1<f32>) {
        let any_winner = post.iter().any(|&s| s > 0.0);

        for (idx, &spike) in post.iter().enumerate() {
            if spike == 1.0 {
                self.neurons[idx] = 0.0;
                self.fired += 1;
                self.n_winners += 1;
                self.refactory[idx] = 3;
                //println!("won :inhibiting {}", self.refactory[idx]);
            } else if any_winner {
                self.refactory[idx] = 3;
                //println!("inhibiting: {}", idx);
            };
        }
    }

    fn stdp(&mut self, pre_spikes: &Array1<f32>, post_spikes: &Array1<f32>) {
        for i in 0..self.out_n {
            //if self.refactory[i] > 0 && post_spikes[i] == 0.0 {
            //    continue;
            //};
            let conns = self.conns.row(i);

            for (j, &conn_idx) in conns.iter().enumerate() {
                let mut dw = 0.0_f32;
                //println!("Post: {} | Pre: {}", post_spikes[i], pre_spikes[i]);
                if post_spikes[i] > 0.0 {
                    dw += self.a_plus * self.pre_trace[conn_idx];
                }

                if pre_spikes[conn_idx] > 0.0 && self.post_trace[i] > 0.0 {
                    dw -= self.a_minus * self.post_trace[i];
                }
                let decay = 0.0001 * self.weights[[i, j]];
                //println!("Update change: {}", dw);

                self.weights[[i, j]] =
                    (self.weights[[i, j]] + dw - decay).clamp(self.w_min, self.w_max);
            }
        }
    }

    // Winner takes all. Selects neurons above threshold and picks top k winners. Computed before post/pre trace.
    pub fn wta(&mut self) -> Array1<f32> {
        let mut neurons: Vec<(usize, f32)> =
            self.neurons.to_vec().into_iter().enumerate().collect();

        neurons.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        //let winners: Vec<(usize, f32)> = neurons.into_iter().take(self.wta_k).collect();

        let mut wns = Vec::new();
        for (i, v) in neurons.into_iter() {
            //if v >= self.thresholds[i] {
            //    wns.push(i);
            //    //self.refactory[i] = 3;
            //    self.fired += 1;
            //}
            //if wns.len() >= self.wta_k || v < self.thresholds[i] {
            //    break;
            //}
            if wns.len() >= self.wta_k {
                break;
            } // got enough winners
            if v < self.thresholds[i] {
                break;
            } // rest are below threshold
            wns.push(i);
        }
        assert!(wns.len() <= self.wta_k);

        let mut post = Array1::zeros(self.out_n);

        //println!("N Winners: {}", wns.len());
        // Set winners to 1.0
        for idx in wns {
            post[idx] = 1.0
        }

        post
    }
}

pub fn generate_connections(out_n: usize, num_conns: usize, in_n: usize) -> Array2<usize> {
    assert!(
        num_conns <= in_n,
        "num_conns ({}) cannot exceed in_n ({})",
        num_conns,
        in_n
    );

    let mut rng = StdRng::seed_from_u64(10);
    let mut conns = Array2::zeros((out_n, num_conns));
    let mut indices: Vec<usize> = (0..in_n).collect();

    for i in 0..out_n {
        indices.shuffle(&mut rng);
        for j in 0..num_conns {
            conns[[i, j]] = indices[j];
        }
    }

    conns
}
