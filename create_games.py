"""
Create game environments for the prompt diversity experiment.
This script implements:
1. IPD-Static-v0: Standard Iterated Prisoner's Dilemma with fixed narrative
2. IPD-Diverse-v0: IPD with 50+ semantically equivalent but linguistically diverse prompts
3. StagHunt-v0: Coordination game for near-domain evaluation
4. MatchingPennies-v0: Zero-sum game for far-domain evaluation
"""

import json
import os
import random
import re
from typing import Dict, Any, Optional, Tuple, List
import textarena as ta


def load_diverse_prompts() -> List[Dict[str, str]]:
    """Load diverse prompts from JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(script_dir, "diverse_prompts.json")
    
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find diverse_prompts.json at {prompts_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in diverse_prompts.json: {e}")


class BaseIteratedGameEnv(ta.Env):
    """A base class for two-player, multi-round games with communication."""
    
    def __init__(self, num_rounds: int = 5, communication_turns: int = 3):
        self.num_rounds = num_rounds
        self.communication_turns = communication_turns
        
    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        game_state = {
            "round": 1,
            "num_rounds": self.num_rounds,
            "phase": "conversation" if self.communication_turns > 0 else "decision",
            "conversation_round": 0,
            "total_conversation_rounds": self.communication_turns,
            "decisions": {0: None, 1: None},
            "scores": {0: 0, 1: 0},
        }
        # Add any game-specific state modifications
        game_state = self._initialize_game_state(game_state)
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        
        # If no conversation turns, immediately prompt for decisions
        if self.communication_turns == 0:
            self.state.add_observation(
                message=f"Round {game_state['round']}. Please reply with {self._get_valid_actions_message()}.",
                observation_type=ta.ObservationType.GAME_BOARD
            )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(
            to_id=self.state.current_player_id,
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION
        )
        
        if self.state.game_state["phase"] == "conversation":
            self._handle_conversation_phase(action)
        elif self.state.game_state["phase"] == "decision":
            self._handle_decision_phase(action)
            
        return self.state.step()
    
    def _handle_conversation_phase(self, action: str):
        # If no communication turns allowed, this should never be called
        if self.communication_turns == 0:
            return
            
        # Share conversation with other player
        self.state.add_observation(
            to_id=1 - self.state.current_player_id,
            from_id=self.state.current_player_id,
            message=action.strip(),
            observation_type=ta.ObservationType.PLAYER_ACTION
        )
        
        # Advance conversation after second player's turn
        if self.state.current_player_id == 1:
            self.state.game_state["conversation_round"] += 1
            
            if (self.state.game_state["conversation_round"] >= 
                self.state.game_state["total_conversation_rounds"]):
                # Switch to decision phase
                self.state.game_state["phase"] = "decision"
                self.state.add_observation(
                    message=f"{self._get_conversation_end_message()} "
                           f"Please reply with {self._get_valid_actions_message()}.",
                    observation_type=ta.ObservationType.GAME_BOARD
                )
    
    def _handle_decision_phase(self, action: str):
        # Parse decision using robust logic
        decision = self._parse_action(action)
        self.state.game_state["decisions"][self.state.current_player_id] = decision
        
        # Resolve when both players have decided
        if all(d is not None for d in self.state.game_state["decisions"].values()):
            self._resolve_round()
            
            # Advance to next round or finish
            self.state.game_state["round"] += 1
            if self.state.game_state["round"] > self.state.game_state["num_rounds"]:
                self._determine_winner()
            else:
                # Reset for next round
                self.state.game_state.update({
                    "phase": "conversation" if self.communication_turns > 0 else "decision",
                    "conversation_round": 0,
                    "decisions": {0: None, 1: None}
                })
                next_round_msg = f"--- Starting {self._get_round_name()} {self.state.game_state['round']} ---"
                if self.communication_turns == 0:
                    next_round_msg += f" Please reply with {self._get_valid_actions_message()}."
                self.state.add_observation(
                    message=next_round_msg,
                    observation_type=ta.ObservationType.GAME_MESSAGE
                )
    
    def _determine_winner(self):
        s0 = self.state.game_state["scores"][0]
        s1 = self.state.game_state["scores"][1]
        
        if s0 == s1:
            self.state.set_draw(reason=f"Draw! Both players scored {s0} points.")
        else:
            winner = 0 if s0 > s1 else 1
            self.state.set_winner(
                player_id=winner,
                reason=f"Player {winner} wins {max(s0, s1)} - {min(s0, s1)}."
            )
    
    # --- Abstract Methods to be implemented by child classes ---
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        raise NotImplementedError("Child classes must implement _prompt method")

    def _resolve_round(self):
        raise NotImplementedError("Child classes must implement _resolve_round method")
        
    def _parse_action(self, action: str) -> str:
        raise NotImplementedError("Child classes must implement _parse_action method")
    
    # --- Hook methods with default implementations ---
    def _initialize_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for child classes to add game-specific state"""
        return game_state
    
    def _get_conversation_end_message(self) -> str:
        """Hook for customizing conversation end message"""
        return f"Conversation finished for round {self.state.game_state['round']}."
    
    def _get_valid_actions_message(self) -> str:
        """Hook for customizing valid actions message"""
        return "your choice"
    
    def _get_round_name(self) -> str:
        """Hook for customizing round naming"""
        return "Round"


class IPDStaticEnv(BaseIteratedGameEnv):
    """Standard IPD with fixed narrative - control condition"""
    
    def __init__(self, num_rounds: int = 5, communication_turns: int = 3):
        super().__init__(num_rounds, communication_turns)
        
        # Standard payoff matrix
        self.cooperate_reward = 3
        self.defect_reward = 5
        self.sucker_reward = 0
        self.mutual_defect_reward = 1
        
        # Action patterns
        self.cooperate_pattern = re.compile(r"\[Cooperate\]", re.IGNORECASE)
        self.defect_pattern = re.compile(r"\[Defect\]", re.IGNORECASE)
    
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        # Build game structure description based on communication turns
        if game_state['total_conversation_rounds'] > 0:
            structure_text = (
                f"Game Structure:\n"
                f"- Before each decision you have {game_state['total_conversation_rounds']} "
                f"turns to communicate freely.\n"
                f"- After that, both players simultaneously choose [Cooperate] or [Defect].\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- During conversation: type any text you wish.\n"
                f"- During decision phase: include '[Cooperate]' or '[Defect]' (case-insensitive). "
                f"You may add extra text before/after the action.\n\n"
            )
        else:
            structure_text = (
                f"Game Structure:\n"
                f"- Both players simultaneously choose [Cooperate] or [Defect] each round.\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- Include '[Cooperate]' or '[Defect]' (case-insensitive) in your response. "
                f"You may add extra text before/after the action.\n\n"
            )
            
        return (
            f"You are Player {player_id} in an Iterated Prisoner's Dilemma spanning "
            f"{game_state['num_rounds']} rounds.\n\n"
            f"{structure_text}"
            f"Payoff Matrix (fixed each round):\n"
            f"- Both Cooperate ➜ each gets {self.cooperate_reward} points\n"
            f"- Both Defect ➜ each gets {self.mutual_defect_reward} point\n"
            f"- One Defects, one Cooperates ➜ Defector gets {self.defect_reward} points, "
            f"Cooperator gets {self.sucker_reward} points\n\n"
            f"{how_to_play_text}"
            f"Your goal is to maximize your total points across all {game_state['num_rounds']} rounds."
        )
    
    def _parse_action(self, action: str) -> str:
        """Robust action parsing with explicit checks for both options"""
        if self.cooperate_pattern.search(action):
            return "cooperate"
        elif self.defect_pattern.search(action):
            return "defect"
        else:
            # Default to defect (less cooperative) for invalid format
            return "defect"
    
    def _get_valid_actions_message(self) -> str:
        return "'[Cooperate]' or '[Defect]'"
    
    def _resolve_round(self):
        d0 = self.state.game_state["decisions"][0]
        d1 = self.state.game_state["decisions"][1]
        
        # Calculate payoffs
        if d0 == d1 == "cooperate":
            r0 = r1 = self.cooperate_reward
            outcome = "Both players cooperated."
        elif d0 == d1 == "defect":
            r0 = r1 = self.mutual_defect_reward
            outcome = "Both players defected."
        elif d0 == "cooperate" and d1 == "defect":
            r0, r1 = self.sucker_reward, self.defect_reward
            outcome = "Player 0 cooperated, Player 1 defected."
        else:
            r0, r1 = self.defect_reward, self.sucker_reward
            outcome = "Player 0 defected, Player 1 cooperated."
        
        # Update scores
        self.state.game_state["scores"][0] += r0
        self.state.game_state["scores"][1] += r1
        
        # Announce results
        self.state.add_observation(
            message=(f"Round {self.state.game_state['round']} results:\n{outcome}\n"
                    f"Player 0 earned {r0} points (total {self.state.game_state['scores'][0]}), "
                    f"Player 1 earned {r1} points (total {self.state.game_state['scores'][1]})."),
            observation_type=ta.ObservationType.GAME_MESSAGE
        )


class IPDDiverseEnv(IPDStaticEnv):
    """IPD with diverse prompts - experimental condition"""
    
    def __init__(self, num_rounds: int = 5, communication_turns: int = 3):
        super().__init__(num_rounds, communication_turns)
        self.prompt_templates = load_diverse_prompts()
    
    def reset(self, num_players: int, seed: Optional[int] = None):
        # Select a random prompt template for this game
        if seed is not None:
            random.seed(seed)
        self.current_prompt = random.choice(self.prompt_templates)
        
        # Update action patterns based on selected prompt
        self.cooperate_pattern = re.compile(
            rf"\[{re.escape(self.current_prompt['cooperate_action'])}\]", 
            re.IGNORECASE
        )
        self.defect_pattern = re.compile(
            rf"\[{re.escape(self.current_prompt['defect_action'])}\]", 
            re.IGNORECASE
        )
        
        super().reset(num_players, seed)
    
    def _get_valid_actions_message(self) -> str:
        return f"'[{self.current_prompt['cooperate_action']}]' or '[{self.current_prompt['defect_action']}]'"
    
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        # Build game structure description based on communication turns
        if game_state['total_conversation_rounds'] > 0:
            structure_text = (
                f"Game Structure:\n"
                f"- Before each decision you have {game_state['total_conversation_rounds']} "
                f"turns to communicate freely.\n"
                f"- After that, both players simultaneously choose "
                f"[{self.current_prompt['cooperate_action']}] or [{self.current_prompt['defect_action']}].\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- During conversation: type any text you wish.\n"
                f"- During decision phase: include '[{self.current_prompt['cooperate_action']}]' or "
                f"'[{self.current_prompt['defect_action']}]' (case-insensitive). "
                f"You may add extra text before/after the action.\n\n"
            )
        else:
            structure_text = (
                f"Game Structure:\n"
                f"- Both players simultaneously choose "
                f"[{self.current_prompt['cooperate_action']}] or [{self.current_prompt['defect_action']}] each round.\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- Include '[{self.current_prompt['cooperate_action']}]' or "
                f"'[{self.current_prompt['defect_action']}]' (case-insensitive) in your response. "
                f"You may add extra text before/after the action.\n\n"
            )
            
        return (
            f"You are Player {player_id} in '{self.current_prompt['title']}'.\n\n"
            f"Scenario: {self.current_prompt['scenario']}.\n"
            f"Context: {self.current_prompt['context']}.\n\n"
            f"This interaction will span {game_state['num_rounds']} rounds.\n\n"
            f"{structure_text}"
            f"Payoff Matrix (fixed each round):\n"
            f"- Both choose {self.current_prompt['cooperate_action']} ➜ each gets {self.cooperate_reward} points\n"
            f"- Both choose {self.current_prompt['defect_action']} ➜ each gets {self.mutual_defect_reward} point\n"
            f"- One chooses {self.current_prompt['defect_action']}, one chooses {self.current_prompt['cooperate_action']} ➜ "
            f"{self.current_prompt['defect_action']} chooser gets {self.defect_reward} points, "
            f"{self.current_prompt['cooperate_action']} chooser gets {self.sucker_reward} points\n\n"
            f"{how_to_play_text}"
            f"Your goal is to maximize your total points across all {game_state['num_rounds']} rounds."
        )


class StagHuntEnv(BaseIteratedGameEnv):
    """Stag Hunt game - coordination game for near-domain evaluation"""
    
    def __init__(self, num_rounds: int = 5, communication_turns: int = 3):
        super().__init__(num_rounds, communication_turns)
        
        # Stag Hunt payoff matrix: R > T > P > S
        self.stag_reward = 4  # Both hunt stag (cooperation)
        self.hare_alone_reward = 2  # Hunt hare while other hunts stag
        self.both_hare_reward = 2  # Both hunt hare
        self.stag_alone_reward = 0  # Hunt stag while other hunts hare
        
        # Action patterns
        self.stag_pattern = re.compile(r"\[Hunt Stag\]", re.IGNORECASE)
        self.hare_pattern = re.compile(r"\[Hunt Hare\]", re.IGNORECASE)
    
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        # Build game structure description based on communication turns
        if game_state['total_conversation_rounds'] > 0:
            structure_text = (
                f"Game Structure:\n"
                f"- Before each decision you have {game_state['total_conversation_rounds']} "
                f"turns to communicate freely.\n"
                f"- After that, both hunters simultaneously choose [Hunt Stag] or [Hunt Hare].\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- During conversation: type any text you wish.\n"
                f"- During decision phase: include '[Hunt Stag]' or '[Hunt Hare]' (case-insensitive). "
                f"You may add extra text before/after the action.\n\n"
            )
        else:
            structure_text = (
                f"Game Structure:\n"
                f"- Both hunters simultaneously choose [Hunt Stag] or [Hunt Hare] each round.\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- Include '[Hunt Stag]' or '[Hunt Hare]' (case-insensitive) in your response. "
                f"You may add extra text before/after the action.\n\n"
            )
            
        return (
            f"You are Hunter {player_id} in the Stag Hunt game spanning "
            f"{game_state['num_rounds']} rounds.\n\n"
            f"Story: Two hunters can either cooperate to hunt a stag (high reward but requires both) "
            f"or hunt hare individually (lower but guaranteed reward).\n\n"
            f"{structure_text}"
            f"Payoff Matrix (fixed each round):\n"
            f"- Both Hunt Stag ➜ each gets {self.stag_reward} points (cooperation succeeds!)\n"
            f"- Both Hunt Hare ➜ each gets {self.both_hare_reward} points (safe choice)\n"
            f"- One hunts Stag, one hunts Hare ➜ Stag hunter gets {self.stag_alone_reward} points, "
            f"Hare hunter gets {self.hare_alone_reward} points\n\n"
            f"{how_to_play_text}"
            f"Your goal is to maximize your total points across all {game_state['num_rounds']} rounds."
        )
    
    def _parse_action(self, action: str) -> str:
        """Robust action parsing with explicit checks for both options"""
        if self.stag_pattern.search(action):
            return "stag"
        elif self.hare_pattern.search(action):
            return "hare"
        else:
            # Default to hare (safe choice) for invalid format
            return "hare"
    
    def _get_valid_actions_message(self) -> str:
        return "'[Hunt Stag]' or '[Hunt Hare]'"
    
    def _get_conversation_end_message(self) -> str:
        return f"Planning finished for round {self.state.game_state['round']}."
    
    def _get_round_name(self) -> str:
        return "Hunt"
    
    def _resolve_round(self):
        d0 = self.state.game_state["decisions"][0]
        d1 = self.state.game_state["decisions"][1]
        
        if d0 == d1 == "stag":
            r0 = r1 = self.stag_reward
            outcome = "Both hunters cooperated to catch the stag!"
        elif d0 == d1 == "hare":
            r0 = r1 = self.both_hare_reward
            outcome = "Both hunters played it safe and caught hares."
        elif d0 == "stag" and d1 == "hare":
            r0, r1 = self.stag_alone_reward, self.hare_alone_reward
            outcome = "Hunter 0 tried for stag but failed, Hunter 1 caught a hare."
        else:
            r0, r1 = self.hare_alone_reward, self.stag_alone_reward
            outcome = "Hunter 0 caught a hare, Hunter 1 tried for stag but failed."
        
        self.state.game_state["scores"][0] += r0
        self.state.game_state["scores"][1] += r1
        
        self.state.add_observation(
            message=(f"Hunt {self.state.game_state['round']} results:\n{outcome}\n"
                    f"Hunter 0 earned {r0} points (total {self.state.game_state['scores'][0]}), "
                    f"Hunter 1 earned {r1} points (total {self.state.game_state['scores'][1]})."),
            observation_type=ta.ObservationType.GAME_MESSAGE
        )


class MatchingPenniesEnv(BaseIteratedGameEnv):
    """Matching Pennies game - zero-sum game for far-domain evaluation"""
    
    def __init__(self, num_rounds: int = 5, communication_turns: int = 3):
        super().__init__(num_rounds, communication_turns)
        
        # Zero-sum payoffs
        self.win_reward = 1
        self.lose_reward = -1
        
        # Action patterns
        self.heads_pattern = re.compile(r"\[Heads\]", re.IGNORECASE)
        self.tails_pattern = re.compile(r"\[Tails\]", re.IGNORECASE)
    
    def _initialize_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Add game-specific state for role assignment and randomize it"""
        # Use the environment's seeded random number generator for perfect reproducibility
        game_state["matcher"] = self.state.rng.choice([0, 1])
        return game_state
    
    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        if player_id == game_state["matcher"]:
            role = "Matcher"
            goal = "match your opponent's choice"
            win_condition = "both choose the same side"
        else:
            role = "Mismatcher"
            goal = "choose differently from your opponent"
            win_condition = "you choose different sides"
            
        # Build game structure description based on communication turns
        if game_state['total_conversation_rounds'] > 0:
            structure_text = (
                f"Game Structure:\n"
                f"- Before each decision you have {game_state['total_conversation_rounds']} "
                f"turns to communicate freely.\n"
                f"- After that, both players simultaneously choose [Heads] or [Tails].\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- During conversation: type any text you wish.\n"
                f"- During decision phase: include '[Heads]' or '[Tails]' (case-insensitive). "
                f"You may add extra text before/after the action.\n\n"
            )
        else:
            structure_text = (
                f"Game Structure:\n"
                f"- Both players simultaneously choose [Heads] or [Tails] each round.\n\n"
            )
            how_to_play_text = (
                f"How to Play:\n"
                f"- Include '[Heads]' or '[Tails]' (case-insensitive) in your response. "
                f"You may add extra text before/after the action.\n\n"
            )
            
        return (
            f"You are Player {player_id} (the {role}) in Matching Pennies spanning "
            f"{game_state['num_rounds']} rounds.\n\n"
            f"Game Rules: This is a competitive coin-flipping game.\n"
            f"Your objective: {goal} to win each round.\n"
            f"You win when: {win_condition}.\n\n"
            f"{structure_text}"
            f"Scoring:\n"
            f"- When you win: +{self.win_reward} point\n"
            f"- When you lose: {self.lose_reward} point\n\n"
            f"{how_to_play_text}"
            f"Your goal is to maximize your total points across all {game_state['num_rounds']} rounds."
        )
    
    def _parse_action(self, action: str) -> str:
        """Robust action parsing with explicit checks for both options"""
        if self.heads_pattern.search(action):
            return "heads"
        elif self.tails_pattern.search(action):
            return "tails"
        else:
            # Default to tails for invalid format (arbitrary choice for zero-sum game)
            return "tails"
    
    def _get_valid_actions_message(self) -> str:
        return "'[Heads]' or '[Tails]'"
    
    def _get_conversation_end_message(self) -> str:
        return f"Discussion finished for round {self.state.game_state['round']}."
    
    def _resolve_round(self):
        d0 = self.state.game_state["decisions"][0]
        d1 = self.state.game_state["decisions"][1]
        
        matcher_id = self.state.game_state["matcher"]
        match = (d0 == d1)
        
        if (match and matcher_id == 0) or (not match and matcher_id == 1):
            # Player 0 wins
            r0, r1 = self.win_reward, self.lose_reward
            winner_role = "Matcher" if matcher_id == 0 else "Mismatcher"
            outcome = f"Player 0 ({winner_role}) wins!"
        else:
            # Player 1 wins
            r0, r1 = self.lose_reward, self.win_reward
            winner_role = "Matcher" if matcher_id == 1 else "Mismatcher"
            outcome = f"Player 1 ({winner_role}) wins!"
        
        # Add details to the outcome message for clarity in logs
        outcome_details = f"Details: P0 chose {d0}, P1 chose {d1}. "
        outcome_details += "Match." if match else "Mismatch."
        
        self.state.game_state["scores"][0] += r0
        self.state.game_state["scores"][1] += r1
        
        self.state.add_observation(
            message=(f"Round {self.state.game_state['round']} results: {outcome}\n"
                    f"{outcome_details}\n"
                    f"Player 0 earned {r0} points (total {self.state.game_state['scores'][0]}), "
                    f"Player 1 earned {r1} points (total {self.state.game_state['scores'][1]})."),
            observation_type=ta.ObservationType.GAME_MESSAGE
        )


def register_environments(num_rounds: int = 5, communication_turns: int = 0):
    """Register all game environments with TextArena with specific parameters"""
    
    # Unregister existing environments to ensure clean registration
    for env_id in ["IPD-Static-v0", "IPD-Diverse-v0", "StagHunt-v0", "MatchingPennies-v0"]:
        try:
            if hasattr(ta, 'envs') and hasattr(ta.envs, 'ENV_REGISTRY') and env_id in ta.envs.ENV_REGISTRY:
                del ta.envs.ENV_REGISTRY[env_id]
        except (AttributeError, KeyError):
            pass  # Environment wasn't registered or registry doesn't exist
    
    # Register environments with TextArena
    try:
        # Register IPD-Static-v0
        ta.envs.register(
            id="IPD-Static-v0",
            entry_point=lambda: IPDStaticEnv(num_rounds=num_rounds, communication_turns=communication_turns),
        )
        
        # Register IPD-Diverse-v0  
        ta.envs.register(
            id="IPD-Diverse-v0",
            entry_point=lambda: IPDDiverseEnv(num_rounds=num_rounds, communication_turns=communication_turns),
        )
        
        # Register StagHunt-v0
        ta.envs.register(
            id="StagHunt-v0", 
            entry_point=lambda: StagHuntEnv(num_rounds=num_rounds, communication_turns=communication_turns),
        )
        
        # Register MatchingPennies-v0
        ta.envs.register(
            id="MatchingPennies-v0",
            entry_point=lambda: MatchingPenniesEnv(num_rounds=num_rounds, communication_turns=communication_turns), 
        )
        
        print(f"✓ Successfully registered environments with {num_rounds} rounds and {communication_turns} communication turns:")
        print("  - IPD-Static-v0: Standard Prisoner's Dilemma (control condition)")
        print("  - IPD-Diverse-v0: Diverse-prompt Prisoner's Dilemma (experimental condition)")  
        print("  - StagHunt-v0: Coordination game (near-domain evaluation)")
        print("  - MatchingPennies-v0: Zero-sum game (far-domain evaluation)")
        
    except Exception as e:
        print(f"Warning: Could not register environments with TextArena: {e}")
        print("Environment classes are still available for direct instantiation")
    
    # Load diverse prompts to get count
    try:
        diverse_prompts = load_diverse_prompts()
        print(f"  - IPD-Diverse includes {len(diverse_prompts)} unique prompt scenarios")
    except Exception as e:
        print(f"  - Warning: Could not load diverse prompts: {e}")
    
    print("\nTo use these environments in your training scripts:")
    print("from IPD.create_games import IPDStaticEnv, IPDDiverseEnv, StagHuntEnv, MatchingPenniesEnv")


if __name__ == "__main__":
    register_environments()
    
    print("\nTesting environment creation...")
    
    try:
        # Test basic instantiation
        static_env = IPDStaticEnv()
        diverse_env = IPDDiverseEnv()
        stag_env = StagHuntEnv()
        pennies_env = MatchingPenniesEnv()
        print("✓ All environment classes instantiated successfully")
        
        # Test that diverse environment can load prompts
        if hasattr(diverse_env, 'prompt_templates') and diverse_env.prompt_templates:
            sample_prompt = diverse_env.prompt_templates[0]
            print(f"\nSample diverse prompt scenario: '{sample_prompt['title']}'")
            print(f"Actions: [{sample_prompt['cooperate_action']}] vs [{sample_prompt['defect_action']}]")
        
        print(f"\n✓ Ready for experimental phases:")
        print(f"  - Phase 1: Train model on IPDStaticEnv (control)")
        print(f"  - Phase 2: Train model on IPDDiverseEnv (experimental)")
        print(f"  - Phase 3: Evaluate both models on StagHuntEnv and MatchingPenniesEnv")
        
    except Exception as e:
        print(f"✗ Error testing environments: {e}")
        import traceback
        traceback.print_exc()
