use tch::nn::{self, Adam, Module, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{Kind, Tensor};

use crate::{
    environment::{Action, State},
    experience::*,
};

#[derive(Debug)]
pub struct NoisyLinear {
    mu_weight: Tensor,
    mu_bias: Tensor,
    sigma_weight: Tensor,
    sigma_bias: Tensor,
    in_features: i64,
    out_features: i64,

    // Factorized noise vectors
    eps_in: Tensor,
    eps_out: Tensor,
}
impl NoisyLinear {
    pub fn new(vs: nn::Path, in_features: i64, out_features: i64, init_std: f64) -> Self {
        let mu_weight = vs.var(
            "mu_weight",
            &[out_features, in_features],
            nn::Init::Uniform {
                lo: -1.0 / (in_features as f64).sqrt(),
                up: 1.0 / (in_features as f64).sqrt(),
            },
        );
        let mu_bias = vs.var("mu_bias", &[out_features], nn::Init::Const(0.0));

        // Initialize sigma parameters to a small value
        let sigma_weight = vs.var(
            "sigma_weight",
            &[out_features, in_features],
            nn::Init::Const(init_std as f64),
        );
        let sigma_bias = vs.var(
            "sigma_bias",
            &[out_features],
            nn::Init::Const(init_std as f64),
        );

        let eps_in = Tensor::zeros(&[in_features], (Kind::Float, vs.device()));
        let eps_out = Tensor::zeros(&[out_features], (Kind::Float, vs.device()));

        let mut layer = NoisyLinear {
            mu_weight,
            mu_bias,
            sigma_weight,
            sigma_bias,
            in_features,
            out_features,
            eps_in,
            eps_out,
        };
        layer.reset_noise();
        layer
    }

    /// Sample noise from a factorized Gaussian distribution
    fn factorized_noise(dim: i64, device: tch::Device) -> Tensor {
        let x = Tensor::randn(&[dim], (Kind::Float, device));
        x.sign() * x.abs().sqrt()
    }

    pub fn reset_noise(&mut self) {
        tch::no_grad(|| {
            self.eps_in = Self::factorized_noise(self.in_features, self.mu_weight.device());
            self.eps_out = Self::factorized_noise(self.out_features, self.mu_weight.device());
        });
    }
}
impl Module for NoisyLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let eps_w = self.eps_out.unsqueeze(1).mm(&self.eps_in.unsqueeze(0));
        let eps_b = &self.eps_out;

        let w = &self.mu_weight + &self.sigma_weight * &eps_w;
        let b = &self.mu_bias + &self.sigma_bias * eps_b;

        xs.mm(&w.transpose(0, 1)) + b
    }
}

/// A neural network for Dueling Deep Q-Network (Dueling DQN).
///
/// This architecture splits the Q-value estimation into two streams:
/// 1. **Value stream**: Estimates the value of being in a given state, independent of the action taken.
/// 2. **Advantage stream**: Estimates the relative benefit (advantage) of each action in the current state.
///
/// The Q-value for each action is then calculated as:
/// Q(s, a) = V(s) + A(s, a) - mean(A(s, a')),
/// where mean(A(s, a')) normalizes the advantage across all actions.
///
/// This approach improves stability and allows the network to more effectively learn which states are valuable,
/// even if the actions available have similar advantages.
#[derive(Debug)]
pub struct DeepQNet {
    /// Shared layers for extracting features from the input state.
    shared_layer: Sequential,
    /// Stream responsible for estimating the value of the state (V(s)).
    value_stream: Sequential,
    /// Stream responsible for estimating the advantage of actions (A(s, a)).
    advantage_stream: Sequential,
}
impl DeepQNet {
    pub fn new(vs: &VarStore) -> Self {
        let input_size = State::SIZE as i64;
        let output_size = Action::SIZE as i64;

        Self {
            shared_layer: nn::seq()
                .add(nn::linear(
                    &vs.root() / "shared_fc1",
                    input_size,
                    128,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    &vs.root() / "shared_fc2",
                    128,
                    128,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu()),

            value_stream: nn::seq()
                .add(nn::linear(
                    &vs.root() / "value_fc1",
                    128,
                    128,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    &vs.root() / "value_fc2",
                    128,
                    1,
                    Default::default(),
                )),

            advantage_stream: nn::seq()
                .add(nn::linear(
                    &vs.root() / "advantage_fc1",
                    128,
                    128,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    &vs.root() / "advantage_fc2",
                    128,
                    output_size,
                    Default::default(),
                )),
        }
    }
}
impl Module for DeepQNet {
    /// # Returns
    /// A tensor representing the Q-values for each action in the input state(s).
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Shared layers: Extract state features.
        let x = self.shared_layer.forward(xs);

        // Value stream: Estimate the state value V(s).
        let value = self.value_stream.forward(&x);

        // Advantage stream: Estimate the advantage of each action A(s, a).
        let advantage = self.advantage_stream.forward(&x);

        // Normalize the advantage by subtracting its mean across actions.
        let advantage_mean = if xs.size().len() == 1 {
            advantage.mean(Kind::Float)
        } else {
            advantage.mean_dim(1, true, Kind::Float)
        };

        // Combine Value and Advantage to compute Q(s, a).
        value + advantage - advantage_mean
    }
}

/// Represents a trainer for Double Deep Q-Network (D-DQN) algorithms with Dueling Architecture.
///
/// This trainer uses the Dueling Deep Q-Network (`DeepQNet`) model, which separates the Q-value
/// calculation into two streams: Value (V(s)) and Advantage (A(s, a)). This allows the network
/// to better distinguish between state values and action advantages.
///
/// The trainer includes two networks:
/// 1. **Online Q-Network**: Trained to approximate Q-values for current states and actions.
/// 2. **Target Q-Network**: Provides stable Q-value estimates for training, reducing overestimation.
///
/// This trainer maintains all functionalities of Double DQN, including:
/// - Action selection based on the online Q-network.
/// - Action evaluation based on the target Q-network.
/// - Soft updates for the target network's parameters.
pub struct DDqnTrainer {
    // The online Q-network trained to estimate action-values
    online_q_network: DeepQNet,
    // Optimizer for updating the online_q_network's parameters
    online_q_optimizer: Optimizer,
    // Variable store holding the online_q_network's parameters
    online_q_vs: VarStore,

    // The target Q-network providing stable targets
    target_q_network: DeepQNet,
    // Variable store holding the target_q_network's parameters
    target_q_vs: VarStore,

    // Discount factor for future rewards
    gamma: f32,
    // Soft update parameter for the target network
    tau: f64,
}

impl DDqnTrainer {
    pub fn new(q_lr: f64, gamma: f32, tau: f64) -> Self {
        let q_vs = VarStore::new(DEVICE);
        let target_q_vs = VarStore::new(DEVICE);
        Self {
            online_q_network: DeepQNet::new(&q_vs),
            online_q_optimizer: Adam::default().build(&q_vs, q_lr).unwrap(),
            online_q_vs: q_vs,

            target_q_network: DeepQNet::new(&target_q_vs),
            target_q_vs,

            gamma,
            tau,
        }
    }

    /// Loads data from path into online_q_network var store.
    pub fn load_in_q_network<T: AsRef<std::path::Path>>(&mut self, path: T) {
        self.online_q_vs.load(path).unwrap();
    }

    /// Copies the data from online_q_network var store to target_q_network.
    pub fn fill_online_network_in_target(&mut self) {
        self.target_q_vs.copy(&self.online_q_vs).unwrap();
    }

    /// Saves the online_q_network var store in path.
    pub fn save_online_q_network<T: AsRef<std::path::Path>>(&self, path: T) {
        self.online_q_vs.save(path).unwrap();
    }

    /// Uses online_q_network to predict values, xs must have [`State::SIZE`] values in a single dimension.
    pub fn online_q_forward(&self, xs: &Tensor) -> Tensor {
        self.online_q_network.forward(&xs.to(DEVICE))
    }

    /// Calculates the loss and TD errors.
    ///
    /// # Returns
    /// - `loss`: Tensor of shape `[]` representing the mean loss.
    /// - `td_errors`: Tensor of shape `[batch_size, 1]` representing the TD errors for each sample.
    pub fn compute_loss(&self, experiences: &PrioritizedExperiences) -> (Tensor, Tensor) {
        // Unpack the mini-batch of experience tuples
        let (states, actions, rewards, next_states, done_values) = experiences.experiences.unpack();

        // Action Selection: Get the best actions for next_states from the online network
        let next_q_values_online = self.online_q_network.forward(&next_states);
        let (_, best_actions) = next_q_values_online.max_dim(1, true); // Indices of best actions

        // Action Evaluation: Get Q-values from the target network for the best actions
        let next_q_values_target = self.target_q_network.forward(&next_states);
        let selected_q_values = next_q_values_target.gather(1, &best_actions, false);

        // Compute y_targets using Double DQN formula
        let y_targets = rewards + (self.gamma * selected_q_values * (1.0 - done_values));

        // Get the Q-values for the actions actually taken
        let q_values = self.online_q_network.forward(&states);
        let q_values = q_values.gather(1, &actions.to_kind(Kind::Int64), false);

        // Compute TD errors: δ = y_targets - Q(s,a)
        let td_errors = y_targets - q_values;

        // Apply importance-sampling weights
        let weights = &experiences.weights;
        let mut weighted_td_errors: Tensor = weights * &td_errors;

        // Compute the loss: Mean squared error with importance-sampling weights
        let loss = weighted_td_errors.pow_(2).mean(Kind::Float);

        (loss, td_errors)
    }

    /// Updates the weights of the Q networks.
    pub fn agent_learn(&mut self, loss: Tensor) {
        self.online_q_optimizer.zero_grad();
        // Compute gradients of the loss with respect to the weights
        loss.backward();
        // Update the weights of the online_q_network.
        self.online_q_optimizer.step();

        // Update the weights of target_q_network using soft update.
        tch::no_grad(|| {
            for (target_params, q_net_params) in self
                .target_q_vs
                .trainable_variables()
                .iter_mut()
                .zip(self.online_q_vs.trainable_variables().iter())
            {
                let new_target_params =
                    self.tau * q_net_params.data() + (1.0 - self.tau) * target_params.data();

                target_params.data().copy_(&new_target_params);
            }
        });
    }
}
