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

fn init_game(mut commands: Commands, mut ev_init: EventReader<game::GameInitEvent>) {
    let state = ev_init.read().next().unwrap().initial_state.clone();

    commands.spawn(GameHolder {
        state,
        total_points: 0.0,
    });

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

fn game_post_reset(
    mut q_holder: Query<&mut GameHolder>,
    mut ev_reset: EventReader<game::GameResetEvent>,
) {
    let mut holder = q_holder.single_mut();

    holder.state = ev_reset.read().next().unwrap().initial_state.clone();
    holder.total_points = 0.0;
}

fn game_pre_step(
    mut commands: Commands,
    mut q_holder: Query<&mut GameHolder>,
    mut ev_step_action: EventWriter<game::StepActionEvent>,
) {
    let mut holder = q_holder.single_mut();

    // TODO: use holder.state and the model to calculate the better action
    ev_step_action.send(game::StepActionEvent::ThrusterRight);

    commands.add(|world: &mut World| {
        world.run_schedule(game::PreGameStepSchedule);
    })
}

fn game_post_step(
    mut commands: Commands,
    mut q_holder: Query<&mut GameHolder>,
    mut ev_step_result: EventReader<game::StepResultEvent>,
) {
    let mut holder = q_holder.single_mut();
    let (next_state, reward, done) = ev_step_result.read().next().unwrap().unpack();

    // TODO: train model, update buffer...

    holder.state = next_state.clone();
    holder.total_points += reward;

    if *done {
        commands.add(|world: &mut World| {
            world.run_schedule(game::GameResetSchedule);
        })
    }
}
