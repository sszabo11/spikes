pub enum Direction {
    North,
    South,
    East,
    West,
}

pub struct Being {
    x: u32,
    y: u32,
    angle: u32,
    dir: Direction,
    health: usize,
    age: usize,
    width: u32,
    fov_dist: u32,
}

impl Being {
    pub fn new() -> Self {
        Self {
            x: 400,
            y: 400,
            angle: 0,
            dir: Direction::South,
            health: 100,
            age: 0,
            width: 46,
            fov_dist: 200,
        }
    }
}
