"""
Run experiment for prompt diversity study on Iterated Prisoner's Dilemma.
This script trains the Qwen3-1.7B model using REINFORCE on IPD environments
with mirror self-play and configurable communication rounds.
"""

import argparse
import time
import ray
import unstable
import unstable.reward_transformations as retra

# Import the game environments and custom templates
from create_games import register_environments, IPDStaticEnv, apply_strategic_game_template, extract_strategic_action_and_format_feedback

# Default configuration - can be overridden with command line arguments
DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B-Base",
    "env_id": "IPD-Static-v0",
    "eval_env_id": "IPD-Static-v0",  # Separate eval environment for fair comparison
    "num_rounds": 5,
    "communication_turns": 0,  # Critical: no conversation as specified
    "max_train_seq_len": None,
    "max_generation_length": 1024,
    "batch_size": 384,
    "mini_batch_size": 1,
    "learning_rate": 1e-5,
    "grad_clip": 0.2,
    "learning_steps": 200,
    "num_train_workers": 128,
    "num_eval_workers": 16,
    "wandb_project": "IPD-PromptDiversity",
    "num_actor_gpus": 7,  # Number of GPUs for actors (inference)
    "num_learner_gpus": 1  # Number of GPUs for learner (training)
}

def create_config(args):
    """Create configuration dictionary from args and defaults."""
    config = DEFAULT_CONFIG.copy()
    
    # Update with command line arguments
    config.update({
        "env_id": args.env_id,
        "eval_env_id": args.eval_env_id,
        "num_rounds": args.num_rounds,
        "communication_turns": args.communication_turns,
        "learning_steps": args.learning_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_workers": args.num_train_workers,
        "num_eval_workers": args.num_eval_workers,
        "wandb_project": args.wandb_project,
        "num_actor_gpus": args.num_actor_gpus,
        "num_learner_gpus": args.num_learner_gpus,
    })
    
    return config

def main():
    """Main experiment function."""
    # Parse arguments and create configuration
    args = get_args()
    config = create_config(args)
    
    # LoRA configuration for efficient fine-tuning
    lora_config = {
        "lora_rank": 32, 
        "lora_alpha": 32, 
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }

    # vLLM configuration for inference
    vllm_config = {
        "model_name": config["model_name"], 
        "temperature": 0.7, 
        "max_tokens": config["max_generation_length"],
        "max_parallel_seq": 64, 
        "max_loras": 16, 
        "lora_config": lora_config,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.8,
    }
    
    # Initialize Ray for distributed computing with specified GPUs
    ray.init(namespace="unstable", num_gpus=config["num_actor_gpus"])

    # Register custom template to avoid confusing prompts
    try:
        import unstable.utils.templates as templates
        templates.OBSERVATION_FORMATTING["strategic-game"] = apply_strategic_game_template
        templates.ACTION_EXTRACTION["strategic-action"] = extract_strategic_action_and_format_feedback
        print("✓ Registered custom strategic game templates in run_experiment.py")
    except ImportError:
        print("⚠️  Could not import unstable templates in run_experiment.py - will be registered during environment setup")
    
    # CRITICAL FIX: Register environments with experiment-specific parameters
    print("Registering custom game environments with experiment-specific parameters...")
    register_environments(
        num_rounds=config["num_rounds"], 
        communication_turns=config["communication_turns"]
    )
    
    # Initialize environment sampler
    env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
        train_env_specs=[
            unstable.TrainEnvSpec(
                env_id=config["env_id"], 
                num_players=2, 
                num_actors=2,  # Mirror self-play (num_players == num_actors)
                prompt_template="strategic-game",
                action_extraction_fn="strategic-action"
            ),
        ],
        eval_env_specs=[
            unstable.EvalEnvSpec(
                env_id=config["eval_env_id"], 
                num_players=2, 
                prompt_template="strategic-game",
                action_extraction_fn="strategic-action",
            ),
        ]
    )
    
    # Initialize tracker for logging and monitoring
    tracker = unstable.Tracker.options(name="Tracker").remote(
        run_name=f"{config['env_id'].replace('-v0', '')}-{config['model_name'].split('/')[-1]}-{int(time.time())}", 
        wandb_project=config["wandb_project"]
    )
    
    # Initialize model registry to manage checkpoints
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
    
    # Add fixed opponent for evaluation (optional baseline)
    ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))
    
    # Initialize model sampler for opponent selection
    model_sampler = unstable.samplers.model_samplers.BaseModelSampler(
        model_registry=model_registry
    )
    
    # Build game scheduler to coordinate environment and model sampling
    game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
        model_sampler=model_sampler, 
        env_sampler=env_sampler, 
        logging_dir=ray.get(tracker.get_log_dir.remote())
    )
    
    # Initialize step buffer for REINFORCE algorithm
    step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
        max_buffer_size=config["batch_size"]*2,  # Buffer size for storing training data
        tracker=tracker,
        final_reward_transformation=retra.ComposeFinalRewardTransforms([
            retra.RoleAdvantageByEnvFormatter()
        ]),
        step_reward_transformation=retra.ComposeStepRewardTransforms([
            retra.RewardForFormat(1.5),  # Reward proper response format
            retra.PenaltyForInvalidMove(1.0, -1.0)  # Penalize invalid moves
        ]),
        sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([
            retra.NormalizeRewardsByEnv(True)  # Normalize rewards by environment
        ]),
    )
    
    # Initialize collector for data collection
    collector = unstable.Collector.options(name="Collector").remote(
        vllm_config=vllm_config, 
        tracker=tracker, 
        buffer=step_buffer, 
        game_scheduler=game_scheduler,
    )
    
    # Initialize REINFORCE learner
    learner = unstable.REINFORCELearner.options(num_gpus=config["num_learner_gpus"], name="Learner").remote(
        model_name=config["model_name"],
        lora_cfg=lora_config,
        batch_size=config["batch_size"],
        mini_batch_size=config["mini_batch_size"],
        learning_rate=config["learning_rate"],
        grad_clip=config["grad_clip"],
        buffer=step_buffer,
        tracker=tracker,
        model_registry=model_registry,
        activation_checkpointing=True,
        gradient_checkpointing=True,
        use_trainer_cache=False
    )
    
    # Initialize the learning algorithm
    ray.get(learner.initialize_algorithm.remote(
        max_train_len=config["max_train_seq_len"], 
        max_generation_len=config["max_generation_length"]
    ))
    
    print(f"\nStarting training...")
    print(f"Monitor progress at: https://wandb.ai (project: {config['wandb_project']})")
    print(f"Use 'unstable-terminal' in another terminal for real-time monitoring")
    
    try:
        # Start data collection
        collector.collect.remote(
            num_train_workers=config["num_train_workers"], 
            num_eval_workers=config["num_eval_workers"]
        )
        
        # Train for specified number of iterations
        print(f"Training for {config['learning_steps']} iterations...")
        ray.get(learner.train.remote(config["learning_steps"]))
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Clean up resources
        ray.kill(collector, no_restart=True)
        ray.shutdown()
        print("Resources cleaned up.")

def get_args():
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Run IPD Prompt Diversity Experiment")
    
    parser.add_argument(
        "--env-id", 
        type=str, 
        default=DEFAULT_CONFIG["env_id"], 
        choices=["IPD-Static-v0", "IPD-Diverse-v0", "StagHunt-v0", "MatchingPennies-v0"],
        help="Environment to train on"
    )
    
    parser.add_argument(
        "--eval-env-id", 
        type=str, 
        default=DEFAULT_CONFIG["eval_env_id"], 
        choices=["IPD-Static-v0", "IPD-Diverse-v0", "StagHunt-v0", "MatchingPennies-v0"],
        help="Environment to evaluate on"
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=DEFAULT_CONFIG["num_rounds"],
        help="Number of game rounds per episode"
    )
    
    parser.add_argument(
        "--communication-turns",
        type=int, 
        default=DEFAULT_CONFIG["communication_turns"],
        help="Number of communication turns before decisions (0 = no conversation)"
    )
    
    parser.add_argument(
        "--learning-steps",
        type=int,
        default=DEFAULT_CONFIG["learning_steps"], 
        help="Number of learning iterations to run"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--num-train-workers", 
        type=int,
        default=DEFAULT_CONFIG["num_train_workers"],
        help="Number of parallel training workers"
    )
    
    parser.add_argument(
        "--num-eval-workers",
        type=int, 
        default=DEFAULT_CONFIG["num_eval_workers"],
        help="Number of parallel evaluation workers"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_CONFIG["wandb_project"],
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--num-actor-gpus",
        type=int,
        default=DEFAULT_CONFIG["num_actor_gpus"],
        help="Number of GPUs for actors (inference)"
    )
    
    parser.add_argument(
        "--num-learner-gpus",
        type=int,
        default=DEFAULT_CONFIG["num_learner_gpus"],
        help="Number of GPUs for learner (training)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
