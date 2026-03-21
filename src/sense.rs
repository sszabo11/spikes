use crate::{brain::Brain, world::World};
use ndarray::Array2;

pub trait Sense {
    fn extract_from(self, world: World, brain: Brain) -> Array2<f32>;
}
