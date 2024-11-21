use crate::game;
use bevy::prelude::*;

#[derive(Component)]
pub struct GameHolder {
    state: game::State,
    total_points: f32,
}

pub fn init_game(mut commands: Commands, mut ev_init: ResMut<Events<game::GameInitEvent>>) {
    let state = ev_init.drain().next().unwrap().initial_state;

    commands.spawn(GameHolder {
        state,
        total_points: 0.0,
    });
}

pub fn game_post_reset(
    mut q_holder: Query<&mut GameHolder>,
    mut ev_reset: ResMut<Events<game::GameResetEvent>>,
) {
    let mut holder = q_holder.single_mut();

    holder.state = ev_reset.drain().next().unwrap().initial_state;
    holder.total_points = 0.0;
}

pub fn game_pre_step(
    mut commands: Commands,
    mut q_holder: Query<&mut GameHolder>,
    mut ev_step_action: EventWriter<game::StepActionEvent>,
) {
    let mut holder = q_holder.single_mut();

    // TODO: use holder.state and the model to calculate the better action

    game::Game::play_step(
        &mut commands,
        &mut ev_step_action,
        game::StepActionEvent::ThrusterLeft,
    );
}

pub fn game_post_step(
    mut commands: Commands,
    mut q_holder: Query<&mut GameHolder>,
    mut ev_step_result: ResMut<Events<game::StepResultEvent>>,
) {
    let mut holder = q_holder.single_mut();
    let (next_state, reward, done) = ev_step_result.drain().next().unwrap().unpack();

    // TODO: train model, update buffer...

    holder.state = next_state.clone();
    holder.total_points += reward;

    if done {
        game::Game::reset(&mut commands);
    }
}
