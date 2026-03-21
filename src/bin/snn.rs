use flatland::{data::get_mnist, spiking::snn::SpikingNetwork};
use ndarray::{ArrayView2, Axis};

fn main() {
    //let world = World::new(800, 800);
    //world.render()

    let mut net = SpikingNetwork::builder()
        .input_layer(784, 12) // This is the actual input
        .layer(500, 10) // First layer
        .layer(300, 5)
        .layer(150, 4)
        .layer(50, 2)
        .layer(10, 1)
        .beta(0.9)
        .build();

    let mnist = get_mnist();

    let epochs = 200;
    for epoch in 0..epochs {
        //for data in 0..100 {
        let input: ArrayView2<f32> = mnist.index_axis(Axis(0), 0);
        assert!(input.len() == 784);

        let flattened = input.flatten().to_owned();
        net.reset_all();
        net.set_input(flattened);
        net.forward();
        //}
    }
    println!();
    let output_layer = net.get_output_layer();

    for (n, v) in output_layer.iter().enumerate() {
        println!("{}: {:?}%", n, v);
    }

    net.print_details();
    //println!("{:?}", output_layer.iter());
}
