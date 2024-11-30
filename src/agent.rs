use std::path::Path;

use rand::{thread_rng, Rng};
use tch::Tensor;

use crate::{
    environment::{Action, State},
    experience::*,
    model::*,
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
const E_DECAY: f64 = 0.995;
/// Minimum ε value for the ε-greedy policy.
const E_MIN: f64 = 0.01;

pub struct Agent {
    memory_buffer: ExperienceReplayBuffer,
    trainer: QTrainer,
    epsilon: f64,
}

impl Agent {
    pub fn load_if_exists(name: &str) -> Self {
        let mut exit = Self {
            memory_buffer: ExperienceReplayBuffer::new(MEMORY_SIZE),
            trainer: QTrainer::new(ALPHA, GAMMA, TAU),
            epsilon: 1.0,
        };

        let model_file = Path::new("./model").join(name.to_owned() + ".ot");
        if model_file.exists() {
            exit.trainer.load_in_q_network(model_file);
        }
        exit.trainer.fill_q_network_in_target();

        let agent_file = Path::new("./model").join(name.to_owned() + ".json");
        if agent_file.exists() {
            let data: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(agent_file).unwrap()).unwrap();

            exit.epsilon = data["epsilon"].as_f64().unwrap();
        }

        exit
    }

    pub fn save(&self, name: &str) {
        let folder_path = Path::new("./model");
        if !folder_path.exists() {
            std::fs::create_dir(folder_path).unwrap();
        }

        let model_file = folder_path.join(name.to_owned() + ".ot");
        self.trainer.save_q_network(model_file);

        let agent_file = folder_path.join(name.to_owned() + ".json");
        let mut json = serde_json::Value::Object(serde_json::Map::new());
        let json_map = json.as_object_mut().unwrap();
        json_map.insert(
            "epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.epsilon).unwrap()),
        );
        std::fs::write(agent_file, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    pub fn append_experience(&mut self, experience: &Experience) {
        self.memory_buffer.push(experience);
    }

    pub fn get_experiences(&self) -> Experiences {
        self.memory_buffer.sample(MINI_BATCH_SIZE)
    }

    pub fn get_action(&self, state: &State) -> Action {
        let mut rng = thread_rng();

        let final_move = if rng.gen_range(0.0..1.0) > self.epsilon {
            let state = Tensor::from_slice(&state.0);
            let prediction = self.trainer.online_q_forward(&state);
            let target_move = prediction.argmax(0, false).int64_value(&[]);
            Action::from_index(target_move as u8)
        } else {
            let target_move = rng.gen_range(0..Action::SIZE);
            Action::from_index(target_move as u8)
        };

        return final_move;
    }

    pub fn check_update_conditions(&self, time_step: usize) -> bool {
        (time_step + 1) % NUM_STEPS_FOR_UPDATE == 0 && self.memory_buffer.size() > MINI_BATCH_SIZE
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = E_MIN.max(E_DECAY * self.epsilon);
    }

    pub fn learn(&mut self, experiences: &Experiences) {
        self.trainer.agent_learn(experiences);
    }
}
