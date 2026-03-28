use mnist::{Mnist, MnistBuilder};
use ndarray::{Array1, Array2, Array3, ArrayView1, Axis, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};

pub struct MnistData {
    pub training_images: Array2<f32>,
    pub training_labels: Vec<u8>,
    pub test_images: Array2<f32>,
    pub test_labels: Vec<u8>,
}

pub fn get_mnist() -> MnistData {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(10000)
        .validation_set_length(1_000)
        .test_set_length(1000)
        .finalize();

    let train_data = Array2::from_shape_vec((10000, 784), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    let mut data = Array2::zeros((500, 784));
    for (i, &d) in trn_lbl.iter().enumerate() {
        if d == 5 {
            let _ = data.push_row(train_data.row(i));
        }
    }

    let test_data = Array2::from_shape_vec((1000, 784), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    println!("Got data");
    MnistData {
        training_images: data,
        training_labels: trn_lbl,
        test_images: test_data,
        test_labels: tst_lbl,
    }
}

pub fn img_to_train(img: &ArrayView1<f32>, timesteps: usize) -> Array2<usize> {
    let mut train = Array2::zeros((timesteps, img.len()));

    let mut rng = rand::rng();

    for t in 0..timesteps {
        for (j, &p) in img.iter().enumerate() {
            train[[t, j]] = if rng.random::<f32>() < p { 1 } else { 0 };
        }
    }

    train
}

pub fn encode_deterministic2(img: &ArrayView1<f32>, t: usize, T: usize) -> Array1<f32> {
    img.mapv(|p| {
        let n_spikes = (p * T as f32).round() as usize;
        if n_spikes == 0 {
            return 0.0;
        }
        let interval = T as f32 / n_spikes as f32;
        let phase = t as f32 % interval;
        if phase < 1.0 { 1.0 } else { 0.0 }
    })
}
pub fn encode_deterministic(img: &ArrayView1<f32>, t: usize, T: usize) -> Array1<f32> {
    img.mapv(|p| {
        let n_spikes = (p * T as f32).round() as usize;
        if n_spikes == 0 {
            return 0.0;
        }
        // Evenly space n_spikes across T timesteps
        // Spike at timestep t if: (t * n_spikes) / T differs from ((t-1) * n_spikes) / T
        let spikes_so_far = (t * n_spikes) / T;
        let spikes_before = if t == 0 { 0 } else { ((t - 1) * n_spikes) / T };
        if spikes_so_far > spikes_before {
            1.0
        } else {
            0.0
        }
    })
}
