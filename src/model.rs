use tch::nn::{self, Adam, Linear, Module, Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor};

use crate::{
    environment::{Action, State},
    experience::*,
};

const DEVICE: Device = Device::Cpu;

/// A neural network model designed for Deep Q-Network (DQN) algorithms.
///
/// The network is used to approximate the Q-value function, which predicts the expected cumulative reward
/// for taking a given action in a given state. The goal of this network is to map input states to Q-values
/// for each possible action, guiding the agent's decision-making process.
///
/// The network's output corresponds to the Q-values for all actions in the action space,
/// and during training, the network minimizes the mean squared error between the predicted Q-values and
/// the target Q-values derived from the Bellman equation. This process helps the agent learn which actions
/// lead to the highest long-term rewards.
///
/// The network is trained using experience tuples collected from the environment, with the aim of improving
/// the agent's policy over time by learning the optimal Q-value function.
#[derive(Debug)]
pub struct DeepQNet {
    /// This layer receives the input data, with the input size, and output values with next the hidden layer size.
    input: Linear,
    /// This layer receives the output data from the previous layer, and output values with the next layer size.
    hidden1: Linear,
    /// This layer receives the output data from the previous layer, and output values with the output size.
    output: Linear,
}
impl DeepQNet {
    pub fn new(vs: &VarStore) -> Self {
        let input_size = State::SIZE as i64;
        let output_size = Action::SIZE as i64;
        let input = nn::linear(&vs.root() / "input", input_size, 64, Default::default());
        let hidden1 = nn::linear(&vs.root() / "hidden1", 64, 64, Default::default());
        let output = nn::linear(&vs.root() / "output", 64, output_size, Default::default());

        Self {
            input,
            hidden1,
            output,
        }
    }
}
impl Module for DeepQNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.input)
            .relu()
            .apply(&self.hidden1)
            .relu()
            .apply(&self.output)
    }
}

/// Represents a trainer for Deep Q-Network (DQN) algorithms.
///
/// It utilizes two separate neural networks: the primary Q-network (q_network)
/// and the target Q-network (target_q_network).
///
/// The use of two networks helps stabilize training by providing a stable
/// target for Q-value estimates.
pub struct QTrainer {
    // The primary Q-network trained to estimate action-values
    q_network: DeepQNet,
    // Optimizer for updating the q_network's parameters
    q_optimizer: Optimizer,
    // Variable store holding the q_network's parameters
    q_vs: VarStore,

    // The target Q-network providing stable targets
    target_q_network: DeepQNet,
    // Variable store holding the target_q_network's parameters
    target_q_vs: VarStore,

    // Discount factor for future rewards
    gamma: f32,
    // Soft update parameter for the target network
    tau: f64,
}

impl QTrainer {
    pub fn new(q_lr: f64, gamma: f32, tau: f64) -> Self {
        let q_vs = VarStore::new(DEVICE);
        let target_q_vs = VarStore::new(DEVICE);
        Self {
            q_network: DeepQNet::new(&q_vs),
            q_optimizer: Adam::default().build(&q_vs, q_lr).unwrap(),
            q_vs,

            target_q_network: DeepQNet::new(&target_q_vs),
            target_q_vs,

            gamma,
            tau,
        }
    }

    /// Loads data from path into q_network var store.
    pub fn load_in_q_network<T: AsRef<std::path::Path>>(&mut self, path: T) {
        self.q_vs.load(path).unwrap();
    }

    /// Copies the data from q_network var store to target_q_network.
    pub fn fill_q_network_in_target(&mut self) {
        self.target_q_vs.copy(&self.q_vs).unwrap();
    }

    /// Saves the q_network var store in path.
    pub fn save_q_network<T: AsRef<std::path::Path>>(&self, path: T) {
        self.q_vs.save(path).unwrap();
    }

    /// Uses q_network to predict values, xs should have [`State::SIZE`] values in a single dimension.
    pub fn q_forward(&self, xs: &Tensor) -> Tensor {
        self.q_network.forward(xs)
    }

    /// Calculates the loss.
    ///
    /// # Returns
    /// loss: Tensor(size=[]) the Mean-Squared Error between
    /// the y targets and the Q(s,a) values.
    fn compute_loss(&self, experiences: &Experiences) -> Tensor {
        // Unpack the mini-batch of experience tuples
        let (states, actions, rewards, next_states, done_values) = experiences.unpack();

        // Compute max Q^(s,a)
        let (max_qsa, _) = self.target_q_network.forward(&next_states).max_dim(1, true);

        // Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
        let y_targets: Tensor = rewards + (self.gamma * max_qsa * (1.0 - done_values));

        // Get the q_values
        let q_values = self.q_network.forward(&states);
        let q_values = q_values.gather(1, &actions.to_kind(Kind::Int64), false);

        // Compute the loss
        let loss = q_values.mse_loss(&y_targets, tch::Reduction::Mean);

        loss
    }

    /// Updates the weights of the Q networks.
    pub fn agent_learn(&mut self, experiences: &Experiences) {
        // Calculate the loss
        let loss = self.compute_loss(&experiences);

        self.q_optimizer.zero_grad();
        // Compute gradients of the loss with respect to the weights
        loss.backward();
        // Update the weights of the q_network.
        self.q_optimizer.step();

        // Update the weights of target q_network using soft update.
        tch::no_grad(|| {
            for (target_params, q_net_params) in self
                .target_q_vs
                .trainable_variables()
                .iter_mut()
                .zip(self.q_vs.trainable_variables().iter())
            {
                let new_target_params =
                    self.tau * q_net_params.data() + (1.0 - self.tau) * target_params.data();

                target_params.data().copy_(&new_target_params);
            }
        });
    }
}
