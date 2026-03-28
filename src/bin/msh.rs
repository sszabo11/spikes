use flatland::mesh::{builder::Builder, mesh::Mesh};

fn main() {
    let mut mesh = Mesh::builder()
        .threshold(0.3)
        .tau_pre(0.4)
        .tau_post(0.4)
        .beta(0.9)
        .timesteps(10)
        .build();

    mesh.run();
}
