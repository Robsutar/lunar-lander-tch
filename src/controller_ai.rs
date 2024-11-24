use std::sync::Mutex;

use crate::{agent::Agent, game::*};
use bevy::prelude::*;

#[derive(Resource)]
pub struct GameHolder {
    action: StepActionEvent,
    state: State,
    total_points: f32,

    agent: Mutex<Agent>,
}

pub fn game_post_reset(mut commands: Commands, mut ev_reset: ResMut<Events<GameResetEvent>>) {
    let state = ev_reset.drain().next().unwrap().initial_state;

    commands.insert_resource(GameHolder {
        action: StepActionEvent::Nothing,
        state,
        total_points: 0.0,

        agent: Mutex::new(Agent::load_if_exists("model.ot")),
    });
}

pub fn game_pre_step(
    mut commands: Commands,
    mut holder: ResMut<GameHolder>,
    mut ev_step_action: EventWriter<StepActionEvent>,
) {
    // From the current state S choose an action A using an ε-greedy policy
    let action = holder.agent.lock().unwrap().get_action(&holder.state);
    holder.action = action.clone();

    Game::play_step(&mut commands, &mut ev_step_action, action);
}

pub fn game_post_step(
    mut commands: Commands,
    mut holder: ResMut<GameHolder>,
    mut ev_step_result: ResMut<Events<StepResultEvent>>,
) {
    let (next_state, reward, done) = ev_step_result.drain().next().unwrap().unpack();

    // TODO: train model, update buffer...

    holder.state = next_state.clone();
    holder.total_points += reward;

    if done {
        Game::reset(&mut commands);
    }
}
