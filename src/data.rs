use mnist::{Mnist, MnistBuilder};
use ndarray::{Array1, Array2, Array3, ArrayView1, s};
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
        .training_set_length(10_000)
        .validation_set_length(1_000)
        .test_set_length(1000)
        .finalize();

    let train_data = Array2::from_shape_vec((10_000, 784), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    let test_data = Array2::from_shape_vec((1000, 784), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    MnistData {
        training_images: train_data,
        training_labels: trn_lbl,
        test_images: test_data,
        test_labels: tst_lbl,
    }
}

pub fn img_to_train(img: &ArrayView1<f32>, timesteps: usize) -> Array2<usize> {
    let mut train = Array2::zeros((timesteps, img.len()));

    let mut rng = StdRng::seed_from_u64(10);

    for t in 0..timesteps {
        for (j, &p) in img.iter().enumerate() {
            train[[t, j]] = if rng.random::<f32>() < p { 1 } else { 0 };
        }
    }

    train
}

pub fn encode_deterministic(img: &ArrayView1<f32>, t: usize, T: usize) -> Array1<f32> {
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
