use flatland::{data::get_mnist, spiking::snn::SpikingNetwork};
use ndarray::{Array2, ArrayView2, Axis, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

const T: usize = 1;
const INPUT_DIM: usize = 784;

fn main() {
    let mut net = SpikingNetwork::builder()
        .threshold(0.5)
        .top_k(100)
        .input_layer(784, 12, INPUT_DIM) // This is the actual input
        .layer(500, 10) // First layer
        .layer(300, 5)
        .layer(150, 4)
        .layer(50, 2)
        .layer(10, 1)
        .beta(0.9)
        .timesteps(T)
        .tau_pre(0.9)
        .tau_post(0.8)
        .build();

    let mnist = get_mnist();

    let spike_train = Array2::random((T, 784), Uniform::new(0, 2).unwrap());

    net.run(spike_train);

    //let epochs = 200;
    //for epoch in 0..epochs {
    //    //for data in 0..100 {
    //    let input: ArrayView2<f32> = mnist.index_axis(Axis(0), 0);
    //    assert!(input.len() == 784);

    //    let flattened = input.flatten().to_owned();
    //    net.reset_all();
    //    net.set_input(flattened);
    //    net.forward();
    //    //}
    //}
    println!();
    let output_layer = net.get_output_layer();

    for (n, v) in output_layer.iter().enumerate() {
        println!("{}: {:?}%", n, v);
    }

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
            .input_layer(20, 2, 40)
            .layer(10, 1)
            .beta(0.9)
            .timesteps(10)
            .tau_pre(0.9)
            .tau_post(0.8)
            .build();

        dbg!("out: {}", net.layers[0].out_n);
        net.layers[1].neurons = array![0.2, 0.3, 0.1, 0.6, 0.3, 0.9, 0.8, 0.6, 0.1, 0.75];
        net.layers[1].wta();
        println!("{}", net.layers[1].neurons);

        assert!(net.layers[1].neurons == array![0., 0., 0., 1., 0., 1., 1., 1., 0., 1.]);
    }
    #[test]
    fn wta2() {
        let mut net = SpikingNetwork::builder()
            .threshold(0.8)
            .input_layer(20, 2, 40)
            .layer(10, 1)
            .beta(0.9)
            .timesteps(10)
            .tau_pre(0.9)
            .tau_post(0.8)
            .build();

        dbg!("out: {}", net.layers[0].out_n);
        net.layers[1].neurons = array![0.92, 0.2, 0.79, 0.78, 0.3, 0.9, 0.8, 1.2, 0.79, 0.99];
        net.layers[1].wta();
        println!("{}", net.layers[1].neurons);

        assert!(net.layers[1].neurons == array![1., 0., 0., 0., 0., 1., 1., 1., 0., 1.]);
    }
}
