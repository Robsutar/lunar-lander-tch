mod agent;
mod controller_ai;
mod controller_human;
mod environment;
mod experience;
mod graph;
mod model;
mod particle;
mod util;

use bevy::{prelude::*, window::WindowResolution};
use environment::*;
use graph::GraphPlugin;

const WINDOW_ZOOM: f32 = 2.0; // Affects only visually the scale of the window, adding zoom to camera.

fn main() {
    let human_controller = false;

    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            resolution: WindowResolution::new(VIEWPORT_W * WINDOW_ZOOM, VIEWPORT_H * WINDOW_ZOOM),
            ..Default::default()
        }),
        ..Default::default()
    }));

    app.add_plugins(EnvironmentPlugin::default());
    app.add_plugins(GraphPlugin);

    if human_controller {
        app.add_systems(PostEnvResetSchedule, controller_human::env_post_reset);
        app.add_systems(AvailableUpdateSchedule, controller_human::env_pre_step);
        app.add_systems(PostEnvStepSchedule, controller_human::env_post_step);
    } else {
        app.add_systems(PostEnvResetSchedule, controller_ai::env_post_reset);
        app.add_systems(AvailableUpdateSchedule, controller_ai::env_pre_step);
        app.add_systems(PostEnvStepSchedule, controller_ai::env_post_step);
    }

    app.add_systems(PostStartup, common_init);

    app.run();
}

fn common_init(mut commands: Commands) {
    // Create camera
    commands.spawn(Camera2dBundle {
        transform: Transform::from_scale(Vec3::new(
            1.0 / SCALE / WINDOW_ZOOM,
            1.0 / SCALE / WINDOW_ZOOM,
            1.0 / SCALE / WINDOW_ZOOM,
        )),
        ..Default::default()
    });
}
