use std::path::Path;

use rand::{thread_rng, Rng};
use tch::Tensor;

use crate::{model::*, util::FixedVecDeque, State, StepActionEvent};

/// Size of memory buffer.
const MEMORY_SIZE: usize = 100_000;
/// Discount factor.
const GAMMA: f32 = 0.995;
/// Learning Rate.
const ALPHA: f64 = 0.001;
/// Perform a leaning update every C time steps
const NUM_STEPS_FOR_UPDATE: usize = 4;

/// Mini-batch size.
const MINI_BATCH_SIZE: usize = 64;
/// Soft update parameter.
const TAU: f64 = 0.001;
/// ε-decay rate for the ε-greedy policy.
const E_DECAY: f64 = 0.9995;
/// Minimum ε value for the ε-greedy policy.
const E_MIN: f64 = 0.01;
