use rand::Rng;
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

/// Prioritized experience replay buffer.
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

    // Prioritized Experience Replay parameters
    alpha: f32,
    max_priority: f32,
    priority_sum: Vec<f32>,
    priority_min: Vec<f32>,
}

impl ExperienceReplayBuffer {
    pub fn new(capacity: usize, alpha: f32) -> Self {
        let state_size_i64 = State::SIZE as i64;
        let capacity_i64 = capacity as i64;

        // Tree size for binary segment tree: 2 * capacity - 1
        let tree_size = 2 * capacity - 1;

        Self {
            states: Tensor::zeros(&[capacity_i64, state_size_i64], (Kind::Float, DEVICE)),
            actions: Tensor::zeros(&[capacity_i64, 1], (Kind::Int64, DEVICE)),
            rewards: Tensor::zeros(&[capacity_i64, 1], (Kind::Float, DEVICE)),
            next_states: Tensor::zeros(&[capacity_i64, state_size_i64], (Kind::Float, DEVICE)),
            done_values: Tensor::zeros(&[capacity_i64, 1], (Kind::Float, DEVICE)),
            capacity,
            position: 0,
            size: 0,
            alpha,
            max_priority: 1.0,
            priority_sum: vec![0.0; tree_size],
            priority_min: vec![f32::INFINITY; tree_size],
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Appends an experience to the buffer with maximum priority.
    ///
    /// If the buffer is full, replace an old experience (cyclically).
    pub fn push(&mut self, experience: &Experience) {
        let idx = self.position;

        // Store each component in its respective tensor
        self.states
            .get(idx as i64)
            .copy_(&Tensor::from_slice(&experience.state.0).to(DEVICE));
        self.actions
            .get(idx as i64)
            .copy_(&Tensor::from(experience.action.to_index() as i64).to(DEVICE));
        self.rewards
            .get(idx as i64)
            .copy_(&Tensor::from(experience.reward).to(DEVICE));
        self.next_states
            .get(idx as i64)
            .copy_(&Tensor::from_slice(&experience.next_state.0).to(DEVICE));
        self.done_values
            .get(idx as i64)
            .copy_(&Tensor::from(if experience.done { 1.0 } else { 0.0 }).to(DEVICE));

        // Set priority for the new experience
        let priority_alpha = self.max_priority.powf(self.alpha);
        self.set_priority(idx, priority_alpha);

        self.position = (self.position + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Updates the priority for a given index.
    fn set_priority(&mut self, idx: usize, priority: f32) {
        self.set_priority_min(idx, priority);
        self.set_priority_sum(idx, priority);
    }

    /// Updates the priority in the sum tree.
    fn set_priority_sum(&mut self, idx: usize, priority: f32) {
        let mut idx = idx + self.capacity - 1;
        self.priority_sum[idx] = priority;

        while idx > 0 {
            idx = (idx - 1) / 2;
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;

            self.priority_sum[idx] = self.priority_sum[left] + self.priority_sum[right];
        }
    }

    /// Updates the priority in the min tree.
    fn set_priority_min(&mut self, idx: usize, priority: f32) {
        let mut idx = idx + self.capacity - 1;
        self.priority_min[idx] = priority;

        while idx > 0 {
            idx = (idx - 1) / 2;
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;

            self.priority_min[idx] = self.priority_min[left].min(self.priority_min[right]);
        }
    }

    /// Returns the total sum of priorities.
    fn sum(&self) -> f32 {
        self.priority_sum[0]
    }

    /// Returns the minimum priority.
    fn min(&self) -> f32 {
        self.priority_min[0]
    }

    /// Finds the index corresponding to a given cumulative priority.
    fn find_prefix_sum_idx(&self, mut prefix_sum: f32) -> usize {
        let mut idx = 0;

        while idx < self.capacity - 1 {
            let left = 2 * idx + 1;
            let right = left + 1;

            if self.priority_sum[left] > prefix_sum {
                idx = left;
            } else {
                prefix_sum -= self.priority_sum[left];
                idx = right;
            }
        }

        // Return the leaf index
        idx - (self.capacity - 1)
    }

    /// Samples experiences based on their priorities.
    ///
    /// Returns the sampled experiences along with importance-sampling weights and indices.
    pub fn sample(&self, batch_size: usize, beta: f32) -> PrioritizedExperiences {
        if self.size < batch_size {
            panic!(
                "Not enough experiences in the buffer to sample a batch with size {}.",
                batch_size
            );
        }

        let mut rng = rand::thread_rng();
        let mut indices = Vec::with_capacity(batch_size);

        let total_priority = self.sum();

        for _ in 0..batch_size {
            let p = rng.gen_range(0.0..total_priority);
            let idx = self.find_prefix_sum_idx(p);
            indices.push(idx);
        }

        // Compute importance-sampling weights
        let prob_min = self.min() / total_priority;
        let max_weight = (prob_min * self.size as f32).powf(-beta);

        let mut weights_vec = Vec::with_capacity(batch_size);

        for &idx in &indices {
            let idx_in_tree = idx + self.capacity - 1;
            let priority = self.priority_sum[idx_in_tree];
            let prob = priority / total_priority;
            let weight = (prob * self.size as f32).powf(-beta);
            weights_vec.push(weight / max_weight);
        }

        let weights = Tensor::from_slice(&weights_vec).unsqueeze(1).to(DEVICE);

        let indices_tensor =
            Tensor::from_slice(&indices.iter().map(|&i| i as i64).collect::<Vec<_>>()).to(DEVICE);

        let experiences = Experiences {
            states: self.states.index_select(0, &indices_tensor),
            actions: self.actions.index_select(0, &indices_tensor),
            rewards: self.rewards.index_select(0, &indices_tensor),
            next_states: self.next_states.index_select(0, &indices_tensor),
            done_values: self.done_values.index_select(0, &indices_tensor),
        };

        PrioritizedExperiences {
            experiences,
            weights,
            indices,
        }
    }

    /// Updates the priorities of sampled experiences.
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f32]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            self.max_priority = self.max_priority.max(priority);
            let priority_alpha = priority.powf(self.alpha);
            self.set_priority(idx, priority_alpha);
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

/// Experiences with priorities and importance-sampling weights.
pub struct PrioritizedExperiences {
    /// Experiences batch.
    pub experiences: Experiences,
    /// Importance-sampling weights.
    pub weights: Tensor,
    /// Indices of sampled experiences.
    pub indices: Vec<usize>,
}
