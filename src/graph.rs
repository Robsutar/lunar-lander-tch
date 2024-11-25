use crate::game::*;
use bevy::prelude::*;

pub struct GraphPlugin;

impl Plugin for GraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostGameResetSchedule, game_post_reset);
        app.add_systems(AvailableUpdateSchedule, game_pre_step);
        app.add_systems(PostGameStepSchedule, game_post_step);
    }
}

pub fn game_post_reset(mut ev_reset: EventReader<GameResetEvent>) {
    let state: State = ev_reset.read().next().unwrap().initial_state.clone();

    println!("Post reset!");
}

pub fn game_pre_step() {
    println!("Pre step!");
}

pub fn game_post_step(mut ev_step_result: EventReader<StepResultEvent>) {
    let (next_state, reward, done) = ev_step_result.read().next().unwrap().clone().unpack();

    println!("Post step!");
}
