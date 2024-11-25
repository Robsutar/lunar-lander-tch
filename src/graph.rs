use crate::{controller_ai, controller_human, game::*};
use bevy::prelude::*;
use bevy_egui::{
    egui::{self},
    EguiContexts, EguiPlugin,
};

struct GameGraph {
    state: State,

    total_points: f32,
}

#[derive(Resource)]
struct Graph {
    num_games: usize,

    actual_game: GameGraph,
}

pub struct GraphPlugin;

impl Plugin for GraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin);

        app.add_systems(
            PostGameResetSchedule,
            game_post_reset
                .before(controller_ai::game_post_reset)
                .before(controller_human::game_post_reset),
        );
        app.add_systems(
            AvailableUpdateSchedule,
            game_pre_step
                .before(controller_ai::game_pre_step)
                .before(controller_human::game_pre_step),
        );
        app.add_systems(
            PostGameStepSchedule,
            game_post_step
                .before(controller_ai::game_post_step)
                .before(controller_human::game_post_step),
        );

        app.add_systems(Update, ui_update);
    }
}

fn game_post_reset(
    mut commands: Commands,
    graph: Option<ResMut<Graph>>,
    mut ev_reset: EventReader<GameResetEvent>,
) {
    let state: State = ev_reset.read().next().unwrap().initial_state.clone();

    match graph {
        Some(mut graph) => {
            graph.num_games += 1;

            graph.actual_game = GameGraph {
                state,
                total_points: 0.0,
            };
        }
        None => {
            commands.insert_resource(Graph {
                num_games: 0,
                actual_game: GameGraph {
                    state,
                    total_points: 0.0,
                },
            });
        }
    }
}

fn game_pre_step() {}

fn game_post_step(mut graph: ResMut<Graph>, mut ev_step_result: EventReader<StepResultEvent>) {
    let (next_state, reward, done) = ev_step_result.read().next().unwrap().clone().unpack();

    graph.actual_game.state = next_state;

    graph.actual_game.total_points += reward;
}

fn ui_update(graph: Res<Graph>, mut egui_context: EguiContexts) {
    egui::Window::new("State")
        .resizable(false)
        .show(egui_context.ctx_mut(), |ui| {
            let state = &graph.actual_game.state;

            ui.label(format!("position_x: {:.5}", state.position_x()));
            ui.label(format!("position_y: {:.5}", state.position_y()));
            ui.label(format!("velocity_x: {:.5}", state.velocity_x()));
            ui.label(format!("velocity_y: {:.5}", state.velocity_y()));
            ui.label(format!(
                "orientation_angle: {:.5}",
                state.orientation_angle()
            ));
            ui.label(format!("angular_velocity: {:.5}", state.angular_velocity()));

            ui.label(format!(
                "is_left_leg_contact: {:?}",
                state.is_left_leg_contact()
            ));
            ui.label(format!(
                "is_right_leg_contact: {:?}",
                state.is_right_leg_contact()
            ));
        });
}
