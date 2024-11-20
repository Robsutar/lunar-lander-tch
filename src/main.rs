mod game;

use bevy::{prelude::*, window::WindowResolution};
use game::GamePlugin;

fn main() {
    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            resolution: WindowResolution::new(
                game::VIEWPORT_W * game::WINDOW_ZOOM,
                game::VIEWPORT_H * game::WINDOW_ZOOM,
            ),
            ..Default::default()
        }),
        ..Default::default()
    }));
    app.add_plugins(GamePlugin::default());
    app.run();
}
