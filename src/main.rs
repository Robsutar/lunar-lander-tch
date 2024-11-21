mod controller_ai;
mod controller_human;
mod game;
mod util;

use bevy::{prelude::*, window::WindowResolution};

pub const WINDOW_ZOOM: f32 = 2.0; // Affects only visually the scale of the window, adding zoom to camera.

fn main() {
    let human_controller = true;

    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            resolution: WindowResolution::new(
                game::VIEWPORT_W * WINDOW_ZOOM,
                game::VIEWPORT_H * WINDOW_ZOOM,
            ),
            ..Default::default()
        }),
        ..Default::default()
    }));

    app.add_plugins(game::GamePlugin::default());

    if human_controller {
        app.add_systems(game::PostGameInitSchedule, controller_human::init_game);
        app.add_systems(
            game::PostGameResetSchedule,
            controller_human::game_post_reset,
        );
        app.add_systems(Update, controller_human::game_pre_step);
        app.add_systems(game::PostGameStepSchedule, controller_human::game_post_step);
    } else {
        app.add_systems(game::PostGameInitSchedule, controller_ai::init_game);
        app.add_systems(game::PostGameResetSchedule, controller_ai::game_post_reset);
        app.add_systems(Update, controller_ai::game_pre_step);
        app.add_systems(game::PostGameStepSchedule, controller_ai::game_post_step);
    }

    app.add_systems(PostStartup, common_init);

    app.run();
}

fn common_init(mut commands: Commands) {
    // Create camera
    commands.spawn(Camera2dBundle {
        transform: Transform::from_scale(Vec3::new(
            1.0 / game::SCALE / WINDOW_ZOOM,
            1.0 / game::SCALE / WINDOW_ZOOM,
            1.0 / game::SCALE / WINDOW_ZOOM,
        )),
        ..Default::default()
    });
}
