use crate::mesh::mesh::Mesh;

#[allow(non_snake_case)]
pub struct MeshBuilder {
    n_layers: usize,
    beta: f32,
    n_conns: usize,
    n_neurons: usize,

    tau_post: f32,
    tau_pre: f32,
    threshold: f32,
    learn: bool,
    a_plus: f32,
    a_minus: f32,

    pub T: usize, // Timesteps
}

impl MeshBuilder {
    pub fn input_layer(
        mut self,
        num_neurons: usize,
        num_conns: usize,
        input_len: usize,
        k: usize,
    ) -> Self {
        assert!(self.threshold != -10.0, "Please specify threshold");
        assert!(self.tau_pre != 0.0, "Please specify `tau_pre`");
        assert!(self.tau_post != 0.0, "Please specify `tau_post`");

        self
    }
    #[allow(non_snake_case)]
    pub fn timesteps(mut self, T: usize) -> Self {
        self.T = T;
        self
    }

    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn tau_pre(mut self, tau_pre: f32) -> Self {
        self.tau_pre = tau_pre;
        self
    }
    pub fn tau_post(mut self, tau_post: f32) -> Self {
        self.tau_post = tau_post;
        self
    }

    pub fn build(self) -> Mesh {
        assert!(
            self.T > 0,
            "Please specify number of timesteps. Example: `builder.timesteps(10)`"
        );

        assert!(
            self.tau_pre > 0.0,
            "`tay_pre` controls decay rate for pre_trace and must be positive"
        );
        assert!(
            self.tau_post > 0.0,
            "`tay_post` controls decay rate for post_trace and must be positive"
        );
        assert!(self.n_neurons > 0);
        assert!(self.n_layers > 0);
        assert!(self.n_conns > 0);
        Mesh {
            tau_pre: 0.0,
            tau_post: 0.0,
            T: self.T,
            learn: self.learn,
            w_max: 1.0,
            w_min: 0.0,
            n_conns_neuron: self.n_conns,
            n_neurons: self.n_neurons,
            fired: 0,
            beta: self.beta,
            ..Default::default()
        }
    }
}

pub trait Builder {
    fn builder() -> MeshBuilder {
        MeshBuilder {
            n_layers: 0,
            threshold: -10.0,
            beta: 0.9,
            learn: true,
            tau_post: 0.0,
            a_plus: 0.01,
            a_minus: 0.01,
            tau_pre: 0.0,
            n_conns: 0,
            T: 0,
            n_neurons: 0,
        }
    }
}
