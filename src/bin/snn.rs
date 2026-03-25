use colored::Colorize;
use flatland::{
    data::{MnistData, get_mnist, img_to_train},
    spiking::snn::SpikingNetwork,
};
use ndarray::{Array1, Array2, array};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

const T: usize = 20;
const INPUT_DIM: usize = 784;
const EPOCHS: usize = 3;

fn main() {
    let mut net = SpikingNetwork::builder()
        .tau_pre(0.95)
        .tau_post(0.95)
        .threshold(0.5)
        .beta(0.99)
        .input_layer(784, 100, INPUT_DIM, 34)
        .layer(500, 30, 22)
        .layer(10, 50, 1)
        .timesteps(T)
        .build();

    let mnist: MnistData = get_mnist();
    let MnistData {
        training_images,
        test_images,
        test_labels,
        training_labels,
    } = mnist;

    let n_classes = 10;
    let n_output = 10;
    // Digit to each output neurons number fired
    //let classify_map: HashMap<u8, Vec<u32>> = HashMap::new();
    let mut voting = vec![vec![0usize; n_classes]; n_output];

    for epoch in 0..EPOCHS {
        println!("Epoch: {}", epoch);
        println!(
            "Firing rate: {} | Thresholds: {}",
            net.layers[0].firing_rates, net.layers[0].thresholds
        );
        for i in 0..training_images.nrows() {
            net.reset();
            let img = training_images.row(i);

            let train = img_to_train(&img, T);
            net.run(train);
        }
    }

    net.learn = false;
    // Count, classify neurons to digits
    for i in 0..training_images.nrows() {
        net.reset();
        let img = training_images.row(i);
        let label = training_labels[i] as usize;

        let train = img_to_train(&img, T);
        // Spikes after WTA, thresh, ...
        let spikes = net.run(train);

        //let spikes = net.get_output_layer();
        println!("spikes: {}", spikes);

        println!("label: {}", label);
        for (neuron, &val) in spikes.iter().enumerate() {
            if val > 0.0 {
                voting[neuron][label] += val as usize;
            }
        }
    }
    println!("Voting: {:?}", voting);
    let mut o_fired = 0;
    voting
        .iter()
        .for_each(|a| a.iter().for_each(|b| o_fired += b));

    let percant = o_fired as f32 / (training_images.nrows() * EPOCHS) as f32;
    println!(
        "Outputs Fired: {} out of {} = {}%",
        o_fired,
        n_output * training_images.nrows() * EPOCHS,
        percant * 100.0
    );

    // Each output neuron will find top firing digit
    let classifier: Vec<usize> = voting
        .iter_mut()
        .map(|digits| {
            let mut d: Vec<(usize, &usize)> = digits.iter().enumerate().collect();
            d.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let top = d[0];

            // digit to corropsing score
            top.0
        })
        .collect();
    // classifier[i] corrosponds to output neuron i being classifier[i]'s digit
    // [1, 0, 6, 5, 7, 0, 0, 0, 1, 0]
    // Becomes: neuron 0: digit 1, neuron 4: digit 7
    println!("class: {:?}", classifier);

    // Evaluate

    let mut n_correct = 0;
    for i in 0..test_images.nrows() {
        net.reset();
        let img = test_images.row(i);
        let label = test_labels[i];

        let train = img_to_train(&img, T);
        let spike_counts = net.run(train);
        let active_neuron = spike_counts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        //let active_neuron = net.get_output_active_neuron();
        let prediction = classifier[active_neuron];
        let correct = label == prediction as u8;
        if correct {
            n_correct += 1;
        }
        println!(
            "{}. Expected: {} | Got: {}",
            if correct {
                "Correct".green()
            } else {
                "Incorrect".red()
            },
            label.to_string().yellow(),
            prediction.to_string().blue()
        );
        println!()
    }
    println!("Correct: {}", n_correct);
    println!(
        "Accuracy: {}%",
        n_correct as f32 / test_images.nrows() as f32 * 100.0
    );

    //let spike_train = Array2::random((T, 784), Uniform::new(0, 2).unwrap());

    //net.run(spike_train);

    println!();

    net.print_details();

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
