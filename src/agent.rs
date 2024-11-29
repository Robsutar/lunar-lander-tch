use std::path::Path;

use rand::{thread_rng, Rng};
use tch::Tensor;

use crate::{
    environment::{Action, State},
    experience::*,
    model::*,
    util::FixedVecDeque,
};

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
    memory_buffer: FixedVecDeque<Experience>,
    trainer: QTrainer,
    epsilon: f64,
}

impl Agent {
    pub fn load_if_exists(file_name: &str) -> Self {
        let mut exit = Self {
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

    pub fn append_experience(&mut self, experience: Experience) {
        self.memory_buffer.push(experience);
    }

    pub fn get_experiences(&self) -> Experiences {
        if self.memory_buffer.len() < MINI_BATCH_SIZE {
            panic!("There is no sufficient experiences to fill the MINI_BATCH_SIZE");
        }

        let mut mini_sample = ExperienceConcat::building(MINI_BATCH_SIZE);
        let mut rng = thread_rng();
        for index in rand::seq::index::sample(&mut rng, self.memory_buffer.len(), MINI_BATCH_SIZE)
            .into_iter()
        {
            mini_sample.push(&self.memory_buffer.as_deque()[index]);
        }

        Experiences::from_concat(&mini_sample)
    }

    pub fn get_action(&self, state: &State) -> Action {
        let mut rng = thread_rng();

        let final_move = if rng.gen_range(0.0..1.0) > self.epsilon {
            let state = Tensor::from_slice(&state.0);
            let prediction = self.trainer.q_forward(&state);
            let target_move = prediction.argmax(0, false).int64_value(&[]);
            Action::from_index(target_move as u8)
        } else {
            let target_move = rng.gen_range(0..Action::SIZE);
            Action::from_index(target_move as u8)
        };

        return final_move;
    }

    pub fn check_update_conditions(&self, time_step: usize) -> bool {
        (time_step + 1) % NUM_STEPS_FOR_UPDATE == 0 && self.memory_buffer.len() > MINI_BATCH_SIZE
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = E_MIN.max(E_DECAY * self.epsilon);
    }

    pub fn learn(&mut self, experiences: &Experiences) {
        self.trainer.agent_learn(experiences);
    }
}
