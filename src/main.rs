use flatland::{data::get_mnist, spiking::snn::SpikingNetwork};
use ndarray::{ArrayView2, Axis};

fn main() {
    //let world = World::new(800, 800);
    //world.render()

    let mut net = SpikingNetwork::builder()
        .layer(784, 10)
        .layer(500, 10)
        .layer(300, 5)
        .layer(150, 4)
        .layer(50, 2)
        .layer(10, 1)
        .build();

    let mnist = get_mnist();

    let input: ArrayView2<f32> = mnist.index_axis(Axis(0), 0);

    let flattened = input.flatten().to_owned();
    println!("{}", input.len());
    net.set_input(flattened);

    net.forward();
    println!();
    let output_layer = net.get_output_layer();
    println!("{:?}", output_layer);
}
