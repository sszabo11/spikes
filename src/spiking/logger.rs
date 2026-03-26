use std::io::Write;

use crate::spiking::layer::SpikingLayer;

pub struct Logger {
    file: std::fs::File,
}

impl Logger {
    pub fn new(path: &str) -> Self {
        let mut file = std::fs::File::create(path).unwrap();
        writeln!(
            file,
            "epoch,image,layer,avg_weight,max_weight,avg_threshold,avg_firing_rate,active_neurons,winners"
        )
        .unwrap();
        Self { file }
    }

    pub fn log(&mut self, epoch: usize, image: usize, layer_idx: usize, layer: &SpikingLayer) {
        let avg_w = layer.weights.mean().unwrap_or(0.0);
        let max_w = layer
            .weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let avg_t = layer.thresholds.mean().unwrap_or(0.0);
        let avg_f = layer.firing_rates.mean().unwrap_or(0.0);
        let active = layer.firing_rates.iter().filter(|&&r| r > 0.001).count();
        let winners = layer.n_winners;

        writeln!(
            self.file,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{}",
            epoch, image, layer_idx, avg_w, max_w, avg_t, avg_f, active, winners
        )
        .unwrap();
    }
}
