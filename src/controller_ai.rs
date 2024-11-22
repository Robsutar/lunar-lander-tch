use crate::game::*;
use bevy::prelude::*;

#[derive(Resource)]
pub struct GameHolder {
    state: State,
    total_points: f32,
}

pub fn game_post_reset(mut commands: Commands, mut ev_reset: ResMut<Events<GameResetEvent>>) {
    let state = ev_reset.drain().next().unwrap().initial_state;

    commands.insert_resource(GameHolder {
        state,
        total_points: 0.0,
    });
}

pub fn game_pre_step(
    mut commands: Commands,
    mut holder: ResMut<GameHolder>,
    mut ev_step_action: EventWriter<StepActionEvent>,
) {
    // TODO: use holder.state and the model to calculate the better action

    Game::play_step(
        &mut commands,
        &mut ev_step_action,
        StepActionEvent::ThrusterLeft,
    );
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
