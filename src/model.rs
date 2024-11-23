use tch::nn::{self, Adam, Linear, Module, Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor};

use crate::{State, StepActionEvent};

const DEVICE: Device = Device::Cpu;
type DType = f32;

/// Represents an experience tuple (𝑆𝑡, 𝐴𝑡, 𝑅𝑡, 𝑆𝑡+1) used in Deep Q-Network (DQN) algorithms.
///
/// This struct captures the agent's interaction with the environment in reinforcement learning.
/// It stores the current state (`state`), the action taken (`action`), the reward received (`reward`),
/// and the next state (`next_state`) after the action is applied. Additionally, the `done` flag indicates
/// whether the episode has terminated.
#[derive(Debug)]
pub struct Experience {
    /// The state 𝑆𝑡 before the action is taken.
    pub state: State,
    /// The action 𝐴𝑡 performed by the agent.
    pub action: StepActionEvent,
    /// The reward 𝑅𝑡 received after taking the action.
    pub reward: DType,
    /// The resulting state 𝑆𝑡+1 after the action is applied.
    pub next_state: State,
    /// A boolean flag that indicates if the episode has terminated.
    pub done: bool,
}

pub struct ExperienceConcat {
    pub state: Vec<DType>,
    pub action: Vec<DType>,
    pub reward: Vec<DType>,
    pub next_state: Vec<DType>,
    pub done: Vec<DType>,
    target_size: usize,
}
impl ExperienceConcat {
    pub fn building(target_size: usize) -> Self {
        Self {
            state: Vec::with_capacity(target_size * State::SIZE),
            action: Vec::with_capacity(target_size * StepActionEvent::SIZE),
            reward: Vec::with_capacity(target_size),
            next_state: Vec::with_capacity(target_size * State::SIZE),
            done: Vec::with_capacity(target_size),
            target_size,
        }
    }

    pub fn push(&mut self, experience: &Experience) {
        if self.is_built() {
            panic!("ExperienceConcat already built (full).");
        }
        self.state.extend_from_slice(&experience.state.0);
        self.action.push(experience.action.to_index() as DType);
        self.reward.push(experience.reward);
        self.next_state.extend_from_slice(&experience.next_state.0);
        self.done.push(if experience.done { 1.0 } else { 0.0 });
    }

    pub fn is_built(&self) -> bool {
        self.done.len() == self.target_size
    }

    pub fn check_built(&self) {
        if !self.is_built() {
            panic!(
                "ExperienceConcat is not built, filled: {:?}, target size: {:?}",
                self.done.len(),
                self.target_size
            );
        }
    }
}

/// Experiences represented by tensors.
pub struct Experiences {
    states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    done_values: Tensor,
}
impl Experiences {
    pub fn from_concat(snapshots: &ExperienceConcat) -> Self {
        snapshots.check_built();

        let len = snapshots.target_size as i64;
        let state_view = [len, State::SIZE as i64];
        let action_view = [len, 1];
        let reward_view = [len, 1];
        let done_view = [len, 1];

        let states = Tensor::from_slice(&snapshots.state).view(state_view);
        let actions = Tensor::from_slice(&snapshots.action).view(action_view);
        let rewards = Tensor::from_slice(&snapshots.reward).view(reward_view);
        let next_states = Tensor::from_slice(&snapshots.next_state).view(state_view);
        let done_values = Tensor::from_slice(&snapshots.done).view(done_view);

        Self {
            states,
            actions,
            rewards,
            next_states,
            done_values,
        }
    }

    fn unpack(&self) -> (&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) {
        (
            &self.states,
            &self.actions,
            &self.rewards,
            &self.next_states,
            &self.done_values,
        )
    }
}

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
        let output_size = StepActionEvent::SIZE as i64;
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
