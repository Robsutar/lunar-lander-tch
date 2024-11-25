use crate::{controller_ai, controller_human, game::*};
use bevy::prelude::*;
use bevy_egui::{
    egui::{self},
    EguiContexts, EguiPlugin,
};
use egui_plot::{AxisHints, Legend, Line, Plot, PlotPoints};

struct GameGraph {
    state: State,

    total_points: f32,
    frames: usize,
}

#[derive(Resource)]
struct Graph {
    done_games: Vec<GameGraph>,

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

    let new_game_graph = GameGraph {
        state,
        total_points: 0.0,
        frames: 0,
    };

    match graph {
        Some(mut graph) => {
            let past_game = std::mem::replace(&mut graph.actual_game, new_game_graph);

            graph.done_games.push(past_game);
        }
        None => {
            commands.insert_resource(Graph {
                done_games: Vec::new(),

                actual_game: new_game_graph,
            });
        }
    }
}

fn game_pre_step() {}

fn game_post_step(mut graph: ResMut<Graph>, mut ev_step_result: EventReader<StepResultEvent>) {
    let (next_state, reward, done) = ev_step_result.read().next().unwrap().clone().unpack();

    graph.actual_game.state = next_state;

    graph.actual_game.total_points += reward;
    graph.actual_game.frames += 1;
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

    egui::Window::new("Game History").show(egui_context.ctx_mut(), |ui| {
        Plot::new("game_history")
            .show_background(false)
            .show_grid(false)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_x(false)
            .show_y(false)
            .legend(Legend::default().position(egui_plot::Corner::LeftTop))
            .custom_x_axes(vec![AxisHints::new_x().label("Number of Games")])
            .show(ui, |plot_ui| {
                let actual_game = &graph.actual_game;
                let done_games = &graph.done_games;

                if done_games.is_empty() {
                    return;
                }

                let mut total_points: Vec<[f64; 2]> = graph
                    .done_games
                    .iter()
                    .enumerate()
                    .map(|(i, game)| [i as f64, game.total_points as f64])
                    .collect();
                total_points.push([done_games.len() as f64, actual_game.total_points as f64]);
                plot_ui.line(Line::new(PlotPoints::new(total_points)).name("Total Points"));

                let mut points_mean: Vec<[f64; 2]> = graph
                    .done_games
                    .iter()
                    .enumerate()
                    .map(|(i, game)| {
                        [
                            i as f64,
                            game.total_points as f64 / (game.frames + 1) as f64,
                        ]
                    })
                    .collect();
                points_mean.push([
                    done_games.len() as f64,
                    actual_game.total_points as f64 / (actual_game.frames + 1) as f64,
                ]);
                plot_ui.line(Line::new(PlotPoints::new(points_mean)).name("Points Mean"));
            });
    });
}
