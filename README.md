# Lunar Lander v2 using tch-rs
LunarLander-v2 (OpenAI Gym) example, using rust, trc-rs and bevy

![controller_ai_running](controller_ai_running.gif)

#### There are two implementations in this repository:
- [Lunar Lander v2 created by OpenAI Gym](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py), written in rust replacing the original model's python, using the rapier2d physics engine replacing box2d, and bevy, game engine replacing pygame.
- Solution for the lunar lander, based on the solution from [DeepLearning.AI](https://www.deeplearning.ai/) & [Stanford University](https://www.stanford.edu/), using tch-rs (pytorch bindings for rust) replacing tensorflow. The solution was also powered with Double Deep Q-Network implementation using the information in the article [Deep Reinforcement Learning with Double Q-learning” (Hasselt et al., 2015)](https://arxiv.org/pdf/1509.06461.pdf).

#### Building and running
You need the rust compiler (cargo), then run:
```bashrc
cargo run --release
```

#### Controlling manually
There is an human-controller implementation, for this lunar lander game, to enable it change the human_controller boolean to true in [main.rs](src/main.rs):
```diff
-    let human_controller = false;
+    let human_controller = true;
```

#### Versions
- [Version 2.0.0](https://github.com/Robsutar/lunar-lander-tch/tree/v2.0.0): Train model with Double Deep Q-Network. 
- [Version 1.1.0](https://github.com/Robsutar/lunar-lander-tch/tree/v1.1.0): Lunar Lander environment creation, DQN network implementation, with replay buffers, soft updates, and more base fundamentals for Deep Q-Networks.
