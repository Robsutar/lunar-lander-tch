use std::sync::Mutex;

use crate::{
    agent::{self, *},
    game::*,
    model::*,
};
use bevy::prelude::*;

#[derive(Resource)]
pub struct GameHolder {
    action: Action,
    state: State,
    total_points: f32,

    agent: Mutex<Agent>,
}
impl GameHolder {
    fn agent(&self) -> std::sync::MutexGuard<'_, Agent> {
        self.agent.lock().unwrap()
    }
}

pub fn game_post_reset(
    mut commands: Commands,
    holder: Option<ResMut<GameHolder>>,
    mut ev_reset: ResMut<Events<GameResetEvent>>,
) {
    let state = ev_reset.drain().next().unwrap().initial_state;

    match holder {
        Some(mut holder) => {
            holder.action = Action::Nothing;
            holder.state = state;
        }
        None => {
            commands.insert_resource(GameHolder {
                action: Action::Nothing,
                state,
                total_points: 0.0,

                agent: Mutex::new(Agent::load_if_exists("model.ot")),
            });
        }
    }
}

pub fn game_pre_step(
    mut commands: Commands,
    mut holder: ResMut<GameHolder>,
    mut ev_step_action: EventWriter<Action>,
) {
    // From the current state S choose an action A using an ε-greedy policy
    let action = holder.agent().get_action(&holder.state);
    holder.action = action.clone();

    Game::play_step(&mut commands, &mut ev_step_action, action);
}

pub fn game_post_step(
    mut commands: Commands,
    mut holder: ResMut<GameHolder>,
    mut ev_step_result: ResMut<Events<StepResultEvent>>,
    q_game: Query<&Game>,
) {
    let game = q_game.single();

    // Take action A and receive reward R and the next state S'
    let (next_state, reward, done) = ev_step_result.drain().next().unwrap().unpack();

    // Store experience tuple (S,A,R,S') in the memory buffer.
    // We store the done variable as well for convenience.
    holder.agent().append_experience(Experience {
        state: holder.state.clone(),
        action: holder.action.clone(),
        reward,
        next_state: next_state.clone(),
        done,
    });

    // Only update the network every NUM_STEPS_FOR_UPDATE time steps.
    let update = holder.agent().check_update_conditions(game.frame());

    if update {
        // Sample random mini-batch of experience tuples (S,A,R,S') from D
        let experiences = holder.agent().get_experiences();

        // Set the y targets, perform a gradient descent step,
        // and update the network weights.
        holder.agent().learn(&experiences);
    }

    holder.state = next_state;
    holder.total_points += reward;

    if done {
        Game::reset(&mut commands);
    }
}
