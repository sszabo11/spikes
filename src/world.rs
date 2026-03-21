pub struct World {
    width: usize,
    height: usize,
}

impl World {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub fn render(&mut self) {
        println!("Rendering...")
    }
}
