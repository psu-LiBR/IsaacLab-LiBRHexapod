# Hexapod Goal Curriculum Design

Train from scratch with goal stages 1.0, 2.0, 3.5, and 5.0 m. Evaluate non-overlapping windows of at least 4096 completed episodes and advance when a window reaches 70% success. Clear the window counters after each evaluation. Never regress stages.

Use a 45 s episode and 0.3 m success radius. Reward progress at +10 per meter, time at -0.2 per second, success at +50 once, and base-contact failure at -25 once. Retain joint-limit protection; disable phase-one gait, posture, energy, contact, and smoothness shaping.

At reset, the curriculum reads reach and fall outcomes, updates stage statistics, then changes the fixed forward pose-command range before command resampling. Log active distance, success rate, episode count, successes, and falls.

Use PPO gamma 0.999, 96 rollout steps, actor/critic observation normalization, actor observation noise, and an uncorrupted privileged critic. Do not load a locomotion checkpoint.

Tests cover reward direction and magnitude, reset-safe progress state, terminal signals, curriculum advancement and cap, disabled phase-one shaping, and PPO overrides. Simulator training and TensorBoard evaluation run on the cluster. Smoothness fine-tuning is out of scope.
