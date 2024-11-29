use rand::seq::SliceRandom;
use tch::{Device, Kind, Tensor};

use crate::environment::{Action, State};

pub const DEVICE: Device = Device::Cpu;
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
    pub action: Action,
    /// The reward 𝑅𝑡 received after taking the action.
    pub reward: DType,
    /// The resulting state 𝑆𝑡+1 after the action is applied.
    pub next_state: State,
    /// A boolean flag that indicates if the episode has terminated.
    pub done: bool,
}
pub struct ExperienceReplayBuffer {
    /// Shape: (capacity, State::SIZE)
    states: Tensor,
    /// Shape: (capacity, 1)
    actions: Tensor,
    /// Shape: (capacity, 1)
    rewards: Tensor,
    /// Shape: (capacity, State::SIZE)
    next_states: Tensor,
    /// Shape: (capacity, 1)
    done_values: Tensor,

    capacity: usize,
    position: usize,
    size: usize,
}

impl ExperienceReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        let state_size_i64 = State::SIZE as i64;
        let capacity_i64 = capacity as i64;
        Self {
            states: Tensor::zeros(&[capacity_i64, state_size_i64], (Kind::Float, DEVICE)),
            actions: Tensor::zeros(&[capacity_i64, 1], (Kind::Int64, DEVICE)),
            rewards: Tensor::zeros(&[capacity_i64, 1], (Kind::Float, DEVICE)),
            next_states: Tensor::zeros(&[capacity_i64, state_size_i64], (Kind::Float, DEVICE)),
            done_values: Tensor::zeros(&[capacity_i64, 1], (Kind::Float, DEVICE)),
            capacity,
            position: 0,
            size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Appends an experience to the buffer
    ///
    /// If the buffer is full, replace other experience by the new
    /// experience (rotational position).
    pub fn push(&mut self, experience: &Experience) {
        let idx = (self.position % self.capacity) as i64;

        // Store each component in its respective tensor
        self.states
            .get(idx)
            .copy_(&Tensor::from_slice(&experience.state.0));
        self.actions
            .get(idx)
            .copy_(&Tensor::from(experience.action.to_index() as i64));
        self.rewards
            .get(idx)
            .copy_(&Tensor::from(experience.reward));
        self.next_states
            .get(idx)
            .copy_(&Tensor::from_slice(&experience.next_state.0));
        self.done_values
            .get(idx)
            .copy_(&Tensor::from(if experience.done { 1.0 } else { 0.0 }));

        self.position = (self.position + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Samples experiences from this buffer with a desired size.
    ///
    /// # Panics
    /// If the buffer has less elements than `batch_size`
    pub fn sample(&self, batch_size: usize) -> Experiences {
        if self.size < batch_size {
            panic!(
                "Not enough experiences in the buffer to sample a batch with size {batch_size}."
            );
        }

        let mut rng = rand::thread_rng();
        let indices = Tensor::from_slice(
            &(0..self.size as i64)
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect::<Vec<_>>(),
        )
        .to_kind(Kind::Int64);

        Experiences {
            states: self.states.index_select(0, &indices),
            actions: self.actions.index_select(0, &indices),
            rewards: self.rewards.index_select(0, &indices),
            next_states: self.next_states.index_select(0, &indices),
            done_values: self.done_values.index_select(0, &indices),
        }
    }
}

/// Experiences represented by tensors.
pub struct Experiences {
    /// State batch.
    states: Tensor,
    /// Action batch.
    actions: Tensor,
    /// Reward batch.
    rewards: Tensor,
    /// Next states batch.
    next_states: Tensor,
    /// Done batch.
    done_values: Tensor,
}
impl Experiences {
    /// # Returns
    ///
    /// - State batch.
    /// - Action batch.
    /// - Reward batch.
    /// - Next states batch.
    /// - Done batch.
    pub fn unpack(&self) -> (&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) {
        (
            &self.states,
            &self.actions,
            &self.rewards,
            &self.next_states,
            &self.done_values,
        )
    }
}
