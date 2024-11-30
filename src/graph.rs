use crate::{controller_ai, controller_human, environment::*};
use bevy::prelude::*;
use bevy_egui::{
    egui::{self},
    EguiContexts, EguiPlugin,
};
use egui_plot::{AxisHints, Legend, Line, Plot, PlotPoints};

struct EnvGraph {
    state: State,

    total_points: f32,
    frame: usize,
}

#[derive(Resource)]
struct Graph {
    done_envs: Vec<EnvGraph>,

    actual_env: EnvGraph,
}

pub struct GraphPlugin;

impl Plugin for GraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin);

        app.add_systems(
            PostEnvResetSchedule,
            env_post_reset
                .before(controller_ai::env_post_reset)
                .before(controller_human::env_post_reset),
        );
        app.add_systems(
            AvailableUpdateSchedule,
            env_pre_step
                .before(controller_ai::env_pre_step)
                .before(controller_human::env_pre_step),
        );
        app.add_systems(
            PostEnvStepSchedule,
            env_post_step
                .before(controller_ai::env_post_step)
                .before(controller_human::env_post_step),
        );

        app.add_systems(Update, ui_update);
    }
}

fn env_post_reset(
    mut commands: Commands,
    graph: Option<ResMut<Graph>>,
    mut ev_reset: EventReader<EnvResetEvent>,
) {
    let state: State = ev_reset.read().next().unwrap().initial_state.clone();

    let new_env_graph = EnvGraph {
        state,
        total_points: 0.0,
        frame: 0,
    };

    match graph {
        Some(mut graph) => {
            let past_env = std::mem::replace(&mut graph.actual_env, new_env_graph);

            graph.done_envs.push(past_env);
        }
        None => {
            commands.insert_resource(Graph {
                done_envs: Vec::new(),

                actual_env: new_env_graph,
            });
        }
    }
}

fn env_pre_step() {}

fn env_post_step(mut graph: ResMut<Graph>, mut ev_step_result: EventReader<StepResultEvent>) {
    let (next_state, reward, _done) = ev_step_result.read().next().unwrap().clone().unpack();

    graph.actual_env.state = next_state;

    graph.actual_env.total_points += reward;
    graph.actual_env.frame += 1;
}

fn ui_update(graph: Res<Graph>, mut egui_context: EguiContexts) {
    egui::Window::new("State")
        .resizable(false)
        .show(egui_context.ctx_mut(), |ui| {
            let state = &graph.actual_env.state;

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

            ui.label("");
            ui.label(format!("Frame: {:?}", graph.actual_env.frame));
        });

    egui::Window::new("Environment History").show(egui_context.ctx_mut(), |ui| {
        Plot::new("env_history")
            .show_background(false)
            .show_grid(false)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_x(false)
            .show_y(false)
            .legend(Legend::default().position(egui_plot::Corner::LeftTop))
            .custom_x_axes(vec![AxisHints::new_x().label("Number of Episodes")])
            .show(ui, |plot_ui| {
                let actual_env = &graph.actual_env;
                let done_envs = &graph.done_envs;

                if done_envs.is_empty() {
                    return;
                }

                let mut total_points: Vec<[f64; 2]> = graph
                    .done_envs
                    .iter()
                    .enumerate()
                    .map(|(i, env)| [i as f64, env.total_points as f64])
                    .collect();
                total_points.push([done_envs.len() as f64, actual_env.total_points as f64]);
                plot_ui.line(Line::new(PlotPoints::new(total_points)).name("Total Points"));

                let mut points_mean: Vec<[f64; 2]> = graph
                    .done_envs
                    .iter()
                    .enumerate()
                    .map(|(i, env)| [i as f64, env.total_points as f64 / (env.frame + 1) as f64])
                    .collect();
                points_mean.push([
                    done_envs.len() as f64,
                    actual_env.total_points as f64 / (actual_env.frame + 1) as f64,
                ]);
                plot_ui.line(Line::new(PlotPoints::new(points_mean)).name("Points Mean"));
            });
    });
}
