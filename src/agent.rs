use std::path::Path;

use rand::{thread_rng, Rng};
use tch::{Device, Tensor};

use crate::{
    environment::{Action, Environment, State},
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
/// Initial ε value for the ε-greedy policy.
const E_START: f64 = 1.0;

/// Used to train the model trainer (DDqnTrainer), with experience replay and ε-greedy policy.
pub struct Agent {
    replay_buffer: ReplayBuffer,
    trainer: DDqnTrainer,
    number_of_episodes: i32,
    number_of_trainings: i32,
}

impl Agent {
    /// Loads the online_q_network (then clone it to target_q_network) if the file "model/`name`.ot"
    /// exists.
    ///
    /// Loads the agent properties, like the epsilon, if the file "model/`name`.json" exists.
    pub fn load_if_exists(name: &str) -> Self {
        let mut exit = Self {
            replay_buffer: ReplayBuffer::new(MEMORY_SIZE, 0.6),
            trainer: DDqnTrainer::new(ALPHA, GAMMA, TAU),
            number_of_episodes: 0,
            number_of_trainings: 0,
        };

        let model_file = Path::new("./model").join(name.to_owned() + ".ot");
        if model_file.exists() {
            exit.trainer.load_in_q_network(model_file);
        }
        exit.trainer.fill_online_network_in_target();

        let agent_file = Path::new("./model").join(name.to_owned() + ".json");
        if agent_file.exists() {
            let data: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(agent_file).unwrap()).unwrap();

            exit.number_of_episodes = data["number_of_episodes"].as_f64().unwrap() as i32;
            exit.number_of_trainings = data["number_of_trainings"].as_f64().unwrap() as i32;
        }

        exit
    }

    /// Saves the online_q_network model in "model/`name`.ot", and the agent properties. like the
    /// epsilon in "model/`name`.json". Then [`Agent::load_if_exists`] can be used to load both
    /// files using the same `name` in function.
    pub fn save(&self, name: &str) {
        let folder_path = Path::new("./model");
        if !folder_path.exists() {
            std::fs::create_dir(folder_path).unwrap();
        }

        let model_file = folder_path.join(name.to_owned() + ".ot");
        self.trainer.save_online_q_network(model_file);

        let agent_file = folder_path.join(name.to_owned() + ".json");
        let mut json = serde_json::Value::Object(serde_json::Map::new());
        let json_map = json.as_object_mut().unwrap();
        json_map.insert(
            "number_of_episodes".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.number_of_episodes as f64).unwrap(),
            ),
        );
        json_map.insert(
            "number_of_trainings".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.number_of_trainings as f64).unwrap(),
            ),
        );
        std::fs::write(agent_file, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    /// Inserts an experience in the memory buffer used for experience replay.
    ///
    /// See [`ExperienceReplayBuffer::push`].
    pub fn append_experience(&mut self, experience: &Experience) {
        self.replay_buffer.push(experience);
    }

    pub fn append_done_env(&mut self, _env: &Environment) {
        self.number_of_episodes += 1;
    }

    /// Decide whether to take a random action or use the online_q_network to find the best action.
    ///
    /// The higher the epsilon of this agent, the greater the chance of a random action being chosen,
    /// this is also called as exploration/exploitation policy.
    pub fn get_action(&self, state: &State) -> Action {
        let mut rng = thread_rng();

        let final_move = if rng.gen_range(0.0..1.0) > self.compute_epsilon() {
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

    /// Returns if good conditions to use [`Agent::learn`] with the values of [`Agent::get_experiences`]
    /// are available.
    pub fn check_update_conditions(&self, time_step: usize) -> bool {
        (time_step + 1) % NUM_STEPS_FOR_UPDATE == 0 && self.replay_buffer.size() > MINI_BATCH_SIZE
    }

    /// Epsilon is used to chose between exploration and exploitation, see [`Agent::get_action`].
    ///
    /// # Returns
    /// Epsilon based on [`Agent#number_of_episodes`].
    fn compute_epsilon(&self) -> f64 {
        let epsilon_n = E_DECAY.powi(self.number_of_episodes) * E_START;
        epsilon_n.max(E_MIN)
    }

    /// Uses `experiences` to adjust the model network parameters, computing loss them use backwards
    /// propagation. Then, the target_q_network is updated with soft updates.
    pub fn learn(&mut self) {
        // Sample random mini-batch of experience tuples (S,A,R,S') from D
        let experiences = self.replay_buffer.sample(MINI_BATCH_SIZE, 0.4);

        // Calculate the loss and TD errors
        let (loss, td_errors) = self.trainer.compute_loss(&experiences);

        // Set the y targets, perform a gradient descent step,
        // and update the network weights.
        self.trainer.agent_learn(loss);

        // Update priorities in the replay buffer
        let td_errors_abs = td_errors.detach().abs().squeeze().to(Device::Cpu);
        let td_errors_vec = Vec::<f32>::try_from(&td_errors_abs).unwrap();

        // Update priorities in the buffer
        self.replay_buffer
            .update_priorities(&experiences.indices, &td_errors_vec);

        self.number_of_trainings += 1;
    }
}
