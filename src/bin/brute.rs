use colored::Colorize;
use flatland::{
    data::{MnistData, get_mnist, img_to_train},
    spiking::snn::SpikingNetwork,
};

const T: usize = 10;
const INPUT_DIM: usize = 784;
const EPOCHS: usize = 3;

#[derive(Debug)]
struct Best {
    acc: f32,
    conn: usize,
    beta: f32,
}

// TODO:
// Fix connections, so don't connect to duplicates
// homeostasis
fn main() {
    let conns = 10..50;

    let mut best = Best {
        acc: 0.0,
        conn: 0,
        beta: 0.0,
    };
    let trials = 50 * 40;
    let mut trial = 0;

    for conn in conns.into_iter() {
        let beta = 60..100;
        for b in beta.into_iter() {
            println!("Trial {}/{}", trial, trials);
            let bta = b as f32 / 100.0;
            let mut net = SpikingNetwork::builder()
                .threshold(0.5)
                .input_layer(784, 100, INPUT_DIM, 34)
                .layer(500, 50, 22)
                .layer(10, conn, 1)
                .beta(bta)
                .timesteps(T)
                .tau_pre(0.95)
                .tau_post(0.95)
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
            let mut voting = vec![vec![0usize; n_classes]; n_output];

            let n_out = net.layers.last().unwrap().out_n;

            for epoch in 0..EPOCHS {
                println!("Epoch: {}", epoch);
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
                //println!("spikes: {}", spikes);

                //println!("label: {}", label);
                for (neuron, &val) in spikes.iter().enumerate() {
                    if val > 0.0 {
                        voting[neuron][label] += val as usize;
                    }
                }
            }
            //println!("Voting: {:?}", voting);
            let mut o_fired = 0;
            voting
                .iter()
                .for_each(|a| a.iter().for_each(|b| o_fired += b));

            let percant = o_fired as f32 / (training_images.nrows() * EPOCHS) as f32;
            //println!(
            //    "Outputs Fired: {} out of {} = {}%",
            //    o_fired,
            //    n_output * training_images.nrows() * EPOCHS,
            //    percant * 100.0
            //);

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
            //println!("class: {:?}", classifier);

            // Evaluate

            let mut n_correct = 0;
            for i in 0..test_images.nrows() {
                net.reset();
                let img = test_images.row(i);
                let label = test_labels[i];

                let train = img_to_train(&img, T);
                let output = net.run(train);
                let active_neuron = net.get_output_active_neuron();
                let prediction = classifier[active_neuron];
                let correct = label == prediction as u8;
                if correct {
                    n_correct += 1;
                }
                //println!(
                //    "{}. Expected: {} | Got: {}",
                //    if correct {
                //        "Correct".green()
                //    } else {
                //        "Incorrect".red()
                //    },
                //    label.to_string().yellow(),
                //    prediction.to_string().blue()
                //);
                //println!()
            }
            //println!("Correct: {}", n_correct);
            let acc = n_correct as f32 / test_images.nrows() as f32 * 100.0;

            println!("Accuracy: {}%", acc);
            println!("Conns {} | beta: {}", conn, bta);

            if acc > best.acc {
                best.acc = acc;
                best.conn = conn;
                best.beta = bta;
            }

            println!();

            //net.print_details();
            trial += 1;
        }
    }
    println!("Best: {:?}", best);

    println!("Done");
}
