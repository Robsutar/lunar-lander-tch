use std::sync::Mutex;

use crate::{agent::*, environment::*, experience::*};
use bevy::prelude::*;

const MAX_NUM_TIME_STEPS: usize = 1000;

#[derive(Resource)]
pub struct EnvHolder {
    action: Action,
    state: State,

    agent: Mutex<Agent>,
}
impl EnvHolder {
    fn agent(&self) -> std::sync::MutexGuard<'_, Agent> {
        self.agent.lock().unwrap()
    }
}

pub fn env_post_reset(
    mut commands: Commands,
    holder: Option<ResMut<EnvHolder>>,
    mut ev_reset: ResMut<Events<EnvResetEvent>>,
) {
    let state = ev_reset.drain().next().unwrap().initial_state;

    match holder {
        Some(mut holder) => {
            holder.action = Action::Nothing;
            holder.state = state;

            holder.agent().save("model");
        }
        None => {
            commands.insert_resource(EnvHolder {
                action: Action::Nothing,
                state,

                agent: Mutex::new(Agent::load_if_exists("model")),
            });
        }
    }
}

pub fn env_pre_step(
    mut commands: Commands,
    mut holder: ResMut<EnvHolder>,
    mut ev_step_action: EventWriter<Action>,
) {
    // From the current state S choose an action A using an ε-greedy policy
    let action = holder.agent().get_action(&holder.state);
    holder.action = action.clone();

    Environment::play_step(&mut commands, &mut ev_step_action, action);
}

pub fn env_post_step(
    mut commands: Commands,
    mut holder: ResMut<EnvHolder>,
    mut ev_step_result: ResMut<Events<StepResultEvent>>,
    q_env: Query<&Environment>,
) {
    let env = q_env.single();

    // Take action A and receive reward R and the next state S'
    let (next_state, reward, done) = ev_step_result.drain().next().unwrap().unpack();

    // Store experience tuple (S,A,R,S') in the memory buffer.
    // We store the done variable as well for convenience.
    holder.agent().append_experience(&Experience {
        state: holder.state.clone(),
        action: holder.action.clone(),
        reward,
        next_state: next_state.clone(),
        done,
    });

    // Only update the network every NUM_STEPS_FOR_UPDATE time steps.
    let update = holder.agent().check_update_conditions(env.frame());

    if update {
        holder.agent().learn();
    }

    holder.state = next_state;

    if done || env.frame() >= MAX_NUM_TIME_STEPS {
        holder.agent().append_done_env(env);
        Environment::reset(&mut commands);
    }
}
