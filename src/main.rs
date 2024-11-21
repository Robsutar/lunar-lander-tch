mod game;
mod util;

use bevy::{prelude::*, window::WindowResolution};

pub const WINDOW_ZOOM: f32 = 2.0; // Affects only visually the scale of the window, adding zoom to camera.

#[derive(Component)]
struct GameHolder {
    state: game::State,
    total_points: f32,
}

fn main() {
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

    app.add_systems(game::PostGameInitSchedule, init_game);
    app.add_systems(game::PostGameResetSchedule, game_post_reset);
    app.add_systems(Update, game_pre_step);
    app.add_systems(game::PostGameStepSchedule, game_post_step);

    app.run();
}
