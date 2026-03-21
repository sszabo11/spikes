use crate::{brain::Brain, world::World};
use ndarray::Array2;

pub trait Action {
    fn parse_from(self, world: World, brain: Brain);
}
