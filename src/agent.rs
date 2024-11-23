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

pub struct Agent {
    n_games: usize,
    memory_buffer: FixedVecDeque<Experience>,
    trainer: QTrainer,
    epsilon: f64,
}

impl Agent {
    pub fn load_if_exists(file_name: &str) -> Self {
        let mut exit = Self {
            n_games: 0,
            memory_buffer: FixedVecDeque::new(MEMORY_SIZE),
            trainer: QTrainer::new(ALPHA, GAMMA, TAU),
            epsilon: 1.0,
        };

        let file_name = Path::new("./model").join(file_name);
        if file_name.exists() {
            exit.trainer.load_in_q_network(file_name);
        }
        exit.trainer.fill_q_network_in_target();

        exit
    }

    pub fn save(&self, file_name: &str) {
        let model_folder_path = Path::new("./model");
        if !model_folder_path.exists() {
            std::fs::create_dir(model_folder_path).unwrap();
        }

        let file_name = model_folder_path.join(file_name);
        self.trainer.save_q_network(file_name);
    }
