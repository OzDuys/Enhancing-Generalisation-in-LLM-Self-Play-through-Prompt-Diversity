Of course. This is the exciting part! Let's transition from the abstract idea to a concrete, actionable plan to get this paper written and submitted.

First, let's address your questions on game selection.

### Game Selection: Depth vs. Breadth

This is a critical decision. Given your budget and the core research question, you should prioritize **depth over breadth**.

**Should you stick to only Iterated Prisoner's Dilemma (IPD)?**
No, you need at least one other game for out-of-domain (OOD) evaluation to test for generalization.

**Should you use all those other games?**
No, that would be too complex and costly. The key is to select games that test different aspects of strategic reasoning.

**Do they have different payout structures?**
Yes, critically so.
*   **Prisoner's Dilemma:** A game of **mistrust**. The dominant strategy is to always defect, even though mutual cooperation is better. Payoffs: `T > R > P > S`.
*   **Stag Hunt:** A game of **coordination**. Two equilibria exist (both cooperate, both defect). It's better to cooperate, but only if you trust the other player will too. Payoffs: `R > T > P > S`.
*   **Matching Pennies:** A **zero-sum, anti-coordination** game. One player wins if they match, the other if they don't. No pure strategy Nash Equilibrium.
*   **Rock, Paper, Scissors / 2/3s Average:** These are less suitable. They are harder to rephrase with a rich narrative, which is the core of your experiment.

**Recommendation:**
1.  **Primary Training Game:** **Iterated Prisoner's Dilemma (IPD)**. It's the most famous and a perfect testbed.
2.  **Primary OOD Evaluation Game:** **Iterated Stag Hunt**. It is structurally similar (2x2 matrix) but requires a different reasoning process (coordination vs. mistrust). This is a perfect *near-domain* generalization test.
3.  **Secondary OOD Evaluation Game:** **Iterated Matching Pennies**. This is a zero-sum game, which tests if the learned strategic reasoning can be adapted to a purely competitive, non-cooperative setting. This is a great *far-domain* generalization test.

This tiered evaluation (In-Domain -> Near-Domain -> Far-Domain) will make your results section extremely compelling.

---

### The Complete Experimental and Writing Plan

Here is a comprehensive plan designed to maximize your $350 budget and produce a high-quality paper.

**GPU Selection:**
The most cost-effective option for your needs is the **`8x A100 (40 GB SXM4)`** instance at **$10.32 / hr**.

*   **Why?** `UnstableBaselines` is built for parallel data collection. Using 7 GPUs as actors and 1 as a learner will dramatically speed up your experiments compared to a single powerful GPU. You can complete a full training run in ~6-8 hours.
*   **Budget:** $350 / $10.32/hr ≈ **33.9 hours** of compute on a powerful cluster. This is more than enough for 2-3 full experimental runs plus evaluation, with a buffer for debugging.

---

### **Phase 1: Setup & Baseline Run (Est. Time: 2 days, Est. Cost: ~$75)**

The goal here is to get your entire pipeline working and establish the first critical data point.

1.  **Code & Environment Setup:**
    *   Create a new project folder. Set up a Python virtual environment.
    *   `pip install unstable-rl`.
    *   Create a new script, `create_games.py`. In here, use the `textarena` API to define your game environments.
        *   **`IPD-Static-v0`**: A standard Iterated Prisoner's Dilemma with one fixed narrative prompt.
        *   **`IPD-Diverse-v0`**: For this, generate ~50-100 high-quality, semantically equivalent but linguistically diverse re-phrasings of the Prisoner's Dilemma (e.g., "The CEO's Dilemma," "The Alien Pact," "The Baker's Choice"). **Pro-tip:** Use a powerful LLM like GPT-4 via openrouter to generate these for you to ensure quality.
        *   **`StagHunt-v0` / `MatchingPennies-v0`**: Create the static versions for evaluation.

2.  **Create the Training Script:**
    *   Create `run_experiment.py`. Adapt the script from the `neurips-competition.md` blog post.
    *   Your model will be `Qwen/Qwen3-1.7B-Base` (as per your original plan, it's a good choice).
    *   Configure the `env_sampler` to only use `unstable.TrainEnvSpec(env_id="IPD-Static-v0", ...)`.

3.  **Execute the Baseline Run:**
    *   Launch the `8x A100` lambda cloud instance.
    *   Run `python3 run_experiment.py`. Use `unstable-terminal` and W&B to monitor.
    *   Let it run for a set number of iterations (e.g., `200` as in the examples, or until performance plateaus). This will be your `Static-Prompt` model.
    *   **Save everything:** the final LoRA checkpoint, the W&B logs, and the console output.

---

### **Phase 2: Core Experiment Run (Est. Time: 1 day, Est. Cost: ~$75)**

Now you run the experimental condition. The only thing that changes is the data.

1.  **Modify the Training Script:**
    *   In `run_experiment.py`, change **only one line**: the `env_sampler` should now point to `unstable.TrainEnvSpec(env_id="IPD-Diverse-v0", ...)`.
    *   **Crucially, keep all other hyperparameters (learning rate, batch size, etc.) identical to Phase 1.**

2.  **Execute the Experimental Run:**
    *   Launch a fresh `8x A100` instance.
    *   Run the script for the *exact same number of iterations* as the baseline. This will be your `Diverse-Prompt` model.
    *   Monitor closely. Does it learn slower? Is the final performance in the training environment lower? These are interesting results in themselves.
    *   Save the final LoRA checkpoint and all logs.

---

### **Phase 3: Rigorous Evaluation (Est. Time: 1 day, Est. Cost: ~$40)**

This is where you generate the core results for your paper. You don't need the 8x machine for this; a single A100 or H100 will be cheaper and sufficient.

1.  **Prepare Models:**
    *   Use the `merge_model.py` script from the blog post to merge the LoRA weights from the best `Static-Prompt` and `Diverse-Prompt` checkpoints into the `Qwen3-1.7B-Base` model.
    *   Upload both merged models to the Hugging Face Hub for easy access.

2.  **Create Evaluation Script (`evaluate.py`):**
    *   Adapt the "Offline Evaluation" script from the blog post. This script should load your two models and a strong baseline opponent (e.g., `google/gemini-2.0-flash-lite-001` or `openai/gpt-4.1-nano`).
    *   It should run a set number of games (e.g., 100-200) for each condition to get statistically meaningful results.

3.  **Run Evaluation Gauntlet:**
    *   **In-Domain (ID):** Evaluate `Static-Model` vs. `Diverse-Model` on `IPD-Static-v0`.
        *   *Hypothesis: The static model might have a slight edge here due to overfitting.*
    *   **Near-Domain (OOD):** Evaluate both on `StagHunt-v0`.
        *   *Hypothesis: The diverse model should significantly outperform the static model.*
    *   **Far-Domain (OOD):** Evaluate both on `MatchingPennies-v0`.
        *   *Hypothesis: The diverse model should adapt better and show stronger performance.*
    *   **Crucial Ablation Study:** Evaluate the `Static-Model` on the `IPD-Diverse-v0` environment. This tests if the base model could already generalize to different prompts without training. If it performs poorly, it strengthens your paper's claim immensely.

---

### **Phase 4: Paper Writing & Analysis (Est. Time: 1-2 weeks, Est. Cost: $0)**

You have all your data. Now, build the story.

1.  **Structure Your Paper:**
    *   **Abstract:** You're done!
    *   **Introduction:** Motivate the problem. Start with the `SPIRAL` findings (self-play leads to generalization) and introduce your hypothesis: *how* the training data is presented (linguistic diversity) is a key mechanism for improving this generalization.
    *   **Methodology:** Describe `UnstableBaselines`, the model (Qwen3-1.7B), and your experimental setup. Detail the two conditions (Static vs. Diverse prompts) and your evaluation games (IPD, Stag Hunt, Matching Pennies).
    *   **Results:** This is the core.
        *   **Table 1: Main Results.** A clear table showing win rates for both models across ID, Near-OOD, and Far-OOD evaluations. Use bold to highlight the superior model in each case.
        *   **Figure 1: Training Dynamics.** A plot from W&B showing the evaluation win-rate-over-time for both models during their training runs.
        *   **Table 2: Ablation Study.** Show the results of the static model on the diverse prompts.
    *   **Discussion:** Analyze the results. Why did the diverse model generalize better? Talk about learning abstract representations vs. memorizing surface-level patterns.
    *   **Conclusion & Future Work:** Summarize your findings and suggest next steps (e.g., testing on more complex games, exploring different methods of prompt generation).

**Total Estimated Cost: $75 + $75 + $40 = $190.** This leaves you with a **$160 buffer** for debugging, re-runs, or adding one more experiment if needed. You are in a great position. Good luck