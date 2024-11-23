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
