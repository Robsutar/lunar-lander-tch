use crate::game::*;
use bevy::prelude::*;
use rand::Rng;

pub fn init_game(mut ev_init: ResMut<Events<GameInitEvent>>) {
    let _state = ev_init.drain().next().unwrap().initial_state;
}

pub fn game_post_reset(mut ev_reset: ResMut<Events<GameResetEvent>>) {
    let _state = ev_reset.drain().next().unwrap().initial_state;
}

pub fn game_pre_step(
    mut commands: Commands,
    mut ev_step_action: EventWriter<StepActionEvent>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let mut possible_actions = Vec::new();
    if keys.pressed(KeyCode::ArrowLeft) {
        possible_actions.push(StepActionEvent::ThrusterRight);
    }
    if keys.pressed(KeyCode::ArrowRight) {
        possible_actions.push(StepActionEvent::ThrusterLeft);
    }
    if keys.pressed(KeyCode::Space) {
        possible_actions.push(StepActionEvent::ThrusterMain);
    }

    let action = {
        if !possible_actions.is_empty() {
            let mut rng = rand::thread_rng();
            let i = rng.gen_range(0..possible_actions.len());

            possible_actions.remove(i)
        } else {
            StepActionEvent::Nothing
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