use colored::Colorize;
use flatland::{data::get_mnist, spiking::snn::SpikingNetwork};
use ndarray::{Array2, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

const T: usize = 10;
const INPUT_DIM: usize = 784;
const EPOCHS: usize = 5;

fn main() {
    let mut net = SpikingNetwork::builder()
        .threshold(0.5)
        .input_layer(784, 78, INPUT_DIM, 34) // This is the actual input
        .layer(500, 50, 22) // First layer
        .layer(200, 14, 10)
        .layer(10, 8, 1)
        .beta(0.95)
        .timesteps(T)
        .tau_pre(0.95)
        .tau_post(0.95)
        .build();

    let mnist = get_mnist();

    for epoch in 0..EPOCHS {
        for (i, img) in mnist.iter() {}
    }

    let spike_train = Array2::random((T, 784), Uniform::new(0, 2).unwrap());

    net.run(spike_train);

    println!();

    net.print_details();

    // WTA Test
    //println!("{:?}", output_layer.iter());
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn wta1() {
        let mut net = SpikingNetwork::builder()
            .threshold(0.2)
            .input_layer(20, 2, 40, 5)
            .layer(10, 1, 5)
            .beta(0.9)
            .timesteps(1)
            .tau_pre(0.9)
            .tau_post(0.8)
            .build();

        net.layers[1].neurons = array![0.2, 0.3, 0.1, 0.6, 0.3, 0.9, 0.8, 0.6, 0.1, 0.75];

        net.layers[1].wta();
        println!("{}", net.layers[1].neurons);

        assert!(net.layers[1].neurons == array![0., 0., 0., 1., 0., 1., 1., 1., 0., 1.]);
    }
    #[test]
    fn wta2() {
        let mut net = SpikingNetwork::builder()
            .threshold(0.8)
            .input_layer(20, 2, 40, 5)
            .layer(10, 1, 5)
            .beta(0.9)
            .timesteps(1)
            .tau_pre(0.9)
            .tau_post(0.8)
            .build();

        net.layers[1].neurons = array![0.92, 0.2, 0.79, 0.78, 0.3, 0.9, 0.8, 1.2, 0.79, 0.99];
        net.layers[1].wta();

        assert!(net.layers[1].neurons == array![1., 0., 0., 0., 0., 1., 1., 1., 0., 1.]);
    }
}

//let mut net = SpikingNetwork::builder()
//    .threshold(0.7)
//    .top_k(50)
//    .input_layer(784, 12, INPUT_DIM) // This is the actual input
//    .layer(500, 10) // First layer
//    .layer(300, 5)
//    .layer(150, 4)
//    .layer(50, 3)
//    .layer(10, 2)
//    .beta(0.9)
//    .timesteps(T)
//    .tau_pre(0.95)
//    .tau_post(0.95)
//    .build();
