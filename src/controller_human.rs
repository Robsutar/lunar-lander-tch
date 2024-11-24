use crate::game::*;
use bevy::prelude::*;
use rand::Rng;

pub fn game_post_reset(mut ev_reset: ResMut<Events<GameResetEvent>>) {
    let _state: State = ev_reset.drain().next().unwrap().initial_state;
}

pub fn game_pre_step(
    mut commands: Commands,
    mut ev_step_action: EventWriter<Action>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let mut possible_actions = Vec::new();
    if keys.pressed(KeyCode::ArrowLeft) {
        possible_actions.push(Action::ThrusterRight);
    }
    if keys.pressed(KeyCode::ArrowRight) {
        possible_actions.push(Action::ThrusterLeft);
    }
    if keys.pressed(KeyCode::Space) {
        possible_actions.push(Action::ThrusterMain);
    }

    let action = {
        if !possible_actions.is_empty() {
            let mut rng = rand::thread_rng();
            let i = rng.gen_range(0..possible_actions.len());

            possible_actions.remove(i)
        } else {
            Action::Nothing
        }
    };

    Game::play_step(&mut commands, &mut ev_step_action, action);
}

pub fn game_post_step(mut commands: Commands, mut ev_step_result: ResMut<Events<StepResultEvent>>) {
    let (_next_state, _reward, done) = ev_step_result.drain().next().unwrap().unpack();

    if done {
        Game::reset(&mut commands);
    }
}
