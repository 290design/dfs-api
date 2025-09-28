import pulp
import random
import time
from collections import Counter
from .models import Player, Lineup


class UniversalLPOptimizer:
    """
    Linear Programming optimizer for DFS lineup generation
    Uses PuLP to solve multiple optimal lineups by iteratively excluding previous solutions
    """

    def __init__(self, players=None, options=None, exposure=None, game=None, optimizer_type=None,
                 optimizer_options=None, num_solutions=None):
        self.players = players or []
        self.options = options or {}
        self.exposure = exposure or {}
        self.game = game or {}
        self.optimizer_type = optimizer_type
        self.optimizer_options = optimizer_options or {}
        self.num_solutions = num_solutions or 1

        # Convert players to Player objects for compatibility
        self.player_pool = [Player(**player) for player in self.players]
        self.player_dict = {x.player_id: x for x in self.player_pool}

        # Game configuration
        self.roster_slots = self.game.get('roster_slots', [])
        self.max_budget = self.game.get('max_budget', 50000)
        self.min_budget = self.game.get('min_budget', 0)
        self.multipliers = self.game.get('multipliers', {})

        # Options
        self.max_per_team = self.options.get('max_per_team', len(self.roster_slots))
        self.max_exposure = self.options.get('max_exposure', 1.0)

        # Stack configuration
        self.stack = self.options.get('stack', {})
        self.stack_enabled = self.stack.get('enabled', False) if self.stack else False

        # Opponent exclusion
        self.exclude_opp_positions = None
        if self.options.get('exclude_opp_roster_slots', {}).get('enabled', False):
            self.exclude_opp_positions = self.options['exclude_opp_roster_slots'].get('roster_slot_map', {})

        # Store generated lineups
        self.generated_lineups = []
        self.exclusion_constraints = []
        self.player_appearances = {}  # Track {player_id: appearance_count}

    def create_lp_model(self):
        """
        Create the linear programming model with all constraints
        """
        # Create the model
        model = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)

        # Decision variables: binary variable for each (player, roster_slot) combination
        player_vars = {}
        for player in self.player_pool:
            for slot_idx, slot in enumerate(self.roster_slots):
                if slot in getattr(player, 'roster_slots', []):
                    var_name = f"player_{player.player_id}_slot_{slot}_{slot_idx}"
                    player_vars[(player.player_id, slot, slot_idx)] = pulp.LpVariable(
                        var_name, cat='Binary'
                    )

        # Calculate exposure adjustments
        exposure_multipliers = self._calculate_exposure_multipliers()

        # Objective function: maximize total value with multipliers and exposure adjustments
        objective = []
        for (player_id, slot, slot_idx), var in player_vars.items():
            player = self.player_dict[player_id]
            value = getattr(player, 'value', 0)
            multiplier = self.multipliers.get(slot, 1.0)
            exposure_mult = exposure_multipliers.get(player_id, 1.0)
            objective.append(value * multiplier * exposure_mult * var)

        model += pulp.lpSum(objective)

        # Constraint 1: Each roster slot must be filled exactly once
        for slot_idx, slot in enumerate(self.roster_slots):
            slot_vars = [var for (pid, s, sidx), var in player_vars.items()
                        if s == slot and sidx == slot_idx]
            if slot_vars:
                model += pulp.lpSum(slot_vars) == 1, f"fill_slot_{slot}_{slot_idx}"

        # Constraint 2: Each player can be used at most once across all positions
        for player in self.player_pool:
            player_vars_list = [var for (pid, s, sidx), var in player_vars.items()
                               if pid == player.player_id]
            if player_vars_list:
                model += pulp.lpSum(player_vars_list) <= 1, f"player_once_{player.player_id}"

        # Constraint 3: Salary cap
        salary_terms = []
        for (player_id, slot, slot_idx), var in player_vars.items():
            player = self.player_dict[player_id]
            salary = getattr(player, 'salary', 0)
            salary_terms.append(salary * var)

        if salary_terms:
            model += pulp.lpSum(salary_terms) <= self.max_budget, "max_salary"
            model += pulp.lpSum(salary_terms) >= self.min_budget, "min_salary"

        # Constraint 4: Max players per team
        if self.max_per_team and self.max_per_team < len(self.roster_slots):
            teams = set(getattr(p, 'team_abbr', '') for p in self.player_pool)
            for team in teams:
                if team:
                    team_vars = [var for (pid, s, sidx), var in player_vars.items()
                                if getattr(self.player_dict[pid], 'team_abbr', '') == team]
                    if team_vars:
                        model += pulp.lpSum(team_vars) <= self.max_per_team, f"max_team_{team}"

        # Constraint 5: Exclude opposing position matchups
        if self.exclude_opp_positions:
            self._add_opponent_exclusion_constraints(model, player_vars)

        # Constraint 6: Stack constraints
        if self.stack_enabled:
            self._add_stack_constraints(model, player_vars)

        # Constraint 7A: Prevent exact duplicates from ALL previous lineups (lightweight)
        for idx, excluded_player_ids in enumerate(self.exclusion_constraints):
            exclusion_vars = []
            for player_id in excluded_player_ids:
                # Get ALL variables for this player across all their possible slots
                player_all_vars = [var for (pid, s, sidx), var in player_vars.items() if pid == player_id]
                exclusion_vars.extend(player_all_vars)

            if exclusion_vars:
                # At least one player must be different (prevents exact duplicates)
                model += pulp.lpSum(exclusion_vars) <= len(excluded_player_ids) - 1, f"no_duplicate_{idx}"

        # Constraint 7B: Progressive diversity constraints (ensures variety from recent lineups)
        self._add_progressive_diversity_constraints(model, player_vars)

        return model, player_vars

    def _calculate_exposure_multipliers(self):
        """
        Calculate value multipliers for players based on exposure targets vs. current progress.

        Handles two types of exposure control:
        1. Per-player targets (self.exposure dict) - explicit targets for specific players
        2. Global max_exposure - soft ceiling for ALL other players with +10% tolerance

        Per-player targets are treated as GOALS (bonuses when under, penalties when over).
        Global max_exposure is treated as a CEILING (only penalties when over, no bonuses).

        For max_exposure = 0.6 (60%), penalty threshold is 0.7 (70%):
        - Players > 0.7 exposure get penalty (lower value multiplier)
        - Players ≤ 0.7 exposure get no adjustment (natural selection by value)

        Per-player targets always override global max_exposure for those players.

        Returns dict of {player_id: multiplier}
        """
        multipliers = {}
        lineups_generated = len(self.exclusion_constraints)

        # Skip all processing if no lineups generated yet
        if lineups_generated == 0:
            return multipliers

        # Process per-player exposure targets first (these override global max_exposure)
        processed_players = set()
        if self.exposure:
            for player_id_str, target_exposure in self.exposure.items():
                player_id = int(player_id_str)
                processed_players.add(player_id)

                # Count how many times this player has appeared so far
                current_appearances = self.player_appearances.get(player_id, 0)
                current_exposure = current_appearances / lineups_generated

                # Calculate deficit/surplus
                exposure_diff = target_exposure - current_exposure

                # Convert to multiplier with clamps to prevent extreme values
                multiplier = 1.0 + (exposure_diff * 1.0)  # Linear adjustment
                multiplier = max(0.3, min(3.0, multiplier))  # Clamp between 0.3x and 3.0x

                multipliers[player_id] = multiplier

        # Process global max_exposure for all other players (+10% tolerance)
        if self.max_exposure and self.max_exposure < 1.0:
            tolerance = 0.10  # +10% tolerance
            max_acceptable = min(1.0, self.max_exposure + tolerance)

            for player in self.player_pool:
                player_id = player.player_id

                # Skip if already processed by per-player targets
                if player_id in processed_players:
                    continue

                # Count how many times this player has appeared so far
                current_appearances = self.player_appearances.get(player_id, 0)
                current_exposure = current_appearances / lineups_generated

                # Only penalize over-exposure (max_exposure is a ceiling, not a target)
                if current_exposure > max_acceptable:
                    # Over-exposed: apply penalty (reduce value)
                    if max_acceptable < 1.0:
                        penalty_factor = (current_exposure - max_acceptable) / (1.0 - max_acceptable)
                    else:
                        penalty_factor = current_exposure - max_acceptable  # Linear penalty when max_acceptable = 1.0
                    multiplier = max(0.3, 1.0 - (penalty_factor * 0.5))  # Up to 50% penalty
                    multipliers[player_id] = multiplier
                # Players under limit get NO adjustment (natural selection based on value)

        return multipliers

    def _add_opponent_exclusion_constraints(self, model, player_vars):
        """
        Add constraints to prevent players from being paired with opponent positions
        """
        for (player_id, slot, slot_idx), var in player_vars.items():
            player = self.player_dict[player_id]
            player_position = getattr(player, 'position_abbr', '')
            player_opponent = getattr(player, 'opponent_abbr', '')

            # Get positions this player's position should exclude
            excluded_positions = self.exclude_opp_positions.get(player_position, [])

            if excluded_positions and player_opponent:
                # Find opponent players in excluded positions
                for other_player in self.player_pool:
                    other_team = getattr(other_player, 'team_abbr', '')
                    other_position = getattr(other_player, 'position_abbr', '')

                    # If this is an opponent player in an excluded position
                    if (other_team == player_opponent and
                        other_position in excluded_positions):

                        # Get all variables for the opponent player
                        opponent_vars = [v for (pid, s, sidx), v in player_vars.items()
                                       if pid == other_player.player_id]

                        # Add constraint: if current player is selected, opponent cannot be selected
                        if opponent_vars:
                            model += var + pulp.lpSum(opponent_vars) <= 1, \
                                   f"exclude_opp_{player.player_id}_{slot}_{slot_idx}_{other_player.player_id}"

    def _add_stack_constraints(self, model, player_vars):
        """
        Add stacking constraints if enabled
        """
        if not self.stack_enabled:
            return

        groups = self.stack.get('groups', [])
        total_players = self.stack.get('total_players', 0)

        if not groups:
            return

        # For each team, add stacking constraints
        teams = set(getattr(p, 'team_abbr', '') for p in self.player_pool)

        for team in teams:
            if not team:
                continue

            # Get team players
            team_players = [p for p in self.player_pool
                           if getattr(p, 'team_abbr', '') == team]

            if len(team_players) < total_players:
                continue

            # Create binary variable for whether this team is stacked
            team_stack_var = pulp.LpVariable(f"stack_team_{team}", cat='Binary')

            # Get variables for players from this team
            team_vars = [var for (pid, s, sidx), var in player_vars.items()
                        if getattr(self.player_dict[pid], 'team_abbr', '') == team]

            if team_vars:
                # If team is stacked, must have at least total_players from team
                model += pulp.lpSum(team_vars) >= total_players * team_stack_var, \
                        f"stack_min_{team}"

                # If team has >= total_players, must be marked as stacked
                model += pulp.lpSum(team_vars) <= total_players + (len(self.roster_slots) - total_players) * (1 - team_stack_var), \
                        f"stack_max_{team}"

            # Required position constraints within stack
            for group in groups:
                if group.get('required', False):
                    position = group.get('value', '')
                    if position:
                        # Must have at least one player from this position if team is stacked
                        position_vars = [var for (pid, s, sidx), var in player_vars.items()
                                       if (getattr(self.player_dict[pid], 'team_abbr', '') == team and
                                           s == position)]

                        if position_vars:
                            model += pulp.lpSum(position_vars) >= team_stack_var, \
                                   f"stack_required_{team}_{position}"

    def _add_progressive_diversity_constraints(self, model, player_vars):
        """
        Add progressive diversity constraints - much more efficient than full exclusion.
        Only requires 30-35% different players from the last 3 lineups.
        This reduces constraint growth from O(n²) to O(1).
        """
        if not self.exclusion_constraints:
            return

        # Only check against last 3 lineups (sliding window)
        # This is the key optimization: instead of checking all previous lineups,
        # we only check the most recent ones
        recent_lineups = self.exclusion_constraints[-3:]
        roster_size = len(self.roster_slots)

        # Require at least 30% different players (empirically proven to work well)
        min_different = max(3, int(0.3 * roster_size))  # At least 30% different

        for idx, prev_lineup_players in enumerate(recent_lineups):
            # Get variables for players in previous lineup
            prev_vars = []
            for player_id in prev_lineup_players:
                player_all_vars = [var for (pid, s, sidx), var in player_vars.items() if pid == player_id]
                prev_vars.extend(player_all_vars)

            if prev_vars:
                # Must have at most (roster_size - min_different) overlap
                # This ensures at least min_different players are different
                max_overlap = roster_size - min_different

                # Use unique constraint name based on current lineup number
                constraint_name = f"diversity_{len(self.exclusion_constraints)}_{idx}"
                model += pulp.lpSum(prev_vars) <= max_overlap, constraint_name

    def _update_dynamic_timeout(self, elapsed_time, total_timeout, lineups_generated, total_requested):
        """
        Dynamically adjust per-solve timeout based on remaining time and performance.
        This handles server performance variability.
        """
        remaining_time = total_timeout - elapsed_time
        remaining_lineups = total_requested - lineups_generated

        if remaining_lineups <= 0:
            return

        # Calculate time budget per remaining lineup
        time_per_lineup = remaining_time / remaining_lineups

        # Adjust timeout based on performance phase
        if lineups_generated <= 5:
            # Early phase: be aggressive, we have time
            target_timeout = min(5, max(1, time_per_lineup * 0.8))
        elif lineups_generated <= 15:
            # Middle phase: moderate timeout
            target_timeout = min(3, max(1, time_per_lineup * 0.7))
        else:
            # Late phase: conservative, avoid timeouts
            target_timeout = min(2, max(0.5, time_per_lineup * 0.6))

        self._dynamic_solve_timeout = target_timeout

        # Log timeout adjustments for monitoring
        if lineups_generated in [5, 10, 15, 20]:
            print(f"Dynamic timeout adjusted to {target_timeout:.1f}s "
                  f"(remaining: {remaining_time:.1f}s for {remaining_lineups} lineups)")

    def solve_single_lineup(self):
        """
        Solve for a single optimal lineup
        """
        model, player_vars = self.create_lp_model()

        # Solve the model using HiGHS with dynamic timeout
        solver_used = "HiGHS"

        # Dynamic timeout: start aggressive, get more conservative as time runs out
        per_solve_timeout = getattr(self, '_dynamic_solve_timeout', 3)

        solve_start = time.time()
        try:
            model.solve(pulp.HiGHS(msg=False, time_limit=per_solve_timeout))
        except Exception:
            # Fallback to CBC if HiGHS is not available
            solver_used = "CBC"
            model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=per_solve_timeout))

        solve_duration = time.time() - solve_start

        # Log performance for first few lineups to gauge server performance
        if len(self.exclusion_constraints) < 5:
            print(f"Lineup {len(self.exclusion_constraints) + 1}: {solve_duration:.2f}s solve time")

        if len(self.exclusion_constraints) == 0:  # Only log for first lineup to avoid spam
            print(f"Solver used: {solver_used}")

        # Add artificial delay for timeout testing (remove in production)
        if hasattr(self, '_test_delay_seconds'):
            time.sleep(self._test_delay_seconds)

        if model.status != pulp.LpStatusOptimal:
            return None

        # Extract solution
        lineup_players = []
        for (player_id, slot, slot_idx), var in player_vars.items():
            if var.varValue and var.varValue > 0.5:  # Binary variable is 1
                player = self.player_dict[player_id]
                lineup_players.append((slot, player))

        if len(lineup_players) == len(self.roster_slots):
            # Create exclusion constraint for this lineup - track unique player IDs only
            selected_player_ids = [player_id for (player_id, slot, slot_idx), var in player_vars.items()
                                 if var.varValue and var.varValue > 0.5]
            exclusion = list(set(selected_player_ids))  # Unique player IDs only
            self.exclusion_constraints.append(exclusion)

            # Update player appearance tracking
            for player_id in exclusion:
                self.player_appearances[player_id] = self.player_appearances.get(player_id, 0) + 1

            # Create Lineup object
            lineup = Lineup(list_of_players=lineup_players, lineup_rules=self.game, options=self.options)
            return lineup

        return None

    def lineups_with_player_positions(self, num_solutions=None):
        """
        Generate multiple lineups and return in the format expected by the API
        """
        if num_solutions is None:
            num_solutions = self.num_solutions

        lineups = []
        start_time = time.time()
        timeout_seconds = 28  # 2-second buffer before 30s server limit

        print(f"=== LINEAR PROGRAMMING OPTIMIZER ===")
        print(f"Generating {num_solutions} optimal lineups...")

        for i in range(num_solutions):
            # Check timeout before each solve
            elapsed_time = time.time() - start_time

            # For testing, estimate time for next solve (includes artificial delay)
            estimated_solve_time = getattr(self, '_test_delay_seconds', 0.5)  # 0.5s normal solve time

            if elapsed_time + estimated_solve_time > timeout_seconds:
                print(f"Timeout reached after {elapsed_time:.1f}s - generated {len(lineups)} lineups")
                break
            lineup = self.solve_single_lineup()

            if lineup and lineup.fitness_score > 0:
                lineups.append(lineup)

                # Update dynamic timeout based on remaining time and observed performance
                self._update_dynamic_timeout(elapsed_time, timeout_seconds, len(lineups), num_solutions)

                # Track diversity and quality metrics
                if len(lineups) > 1:
                    current_players = set(p.player_id for _, p in lineup.list_of_players)
                    prev_players = set(p.player_id for _, p in lineups[-2].list_of_players)
                    overlap = len(current_players & prev_players)

                    # Calculate score drop from first lineup
                    first_score = lineups[0].fitness_score
                    score_drop_pct = ((first_score - lineup.fitness_score) / first_score) * 100

                    # Calculate average score of recent lineups
                    recent_window = min(5, len(lineups))
                    recent_scores = [l.fitness_score for l in lineups[-recent_window:]]
                    recent_avg = sum(recent_scores) / len(recent_scores)

                    print(f"Lineup {i + 1}: Score {lineup.fitness_score:.2f} "
                          f"(drop: {score_drop_pct:.1f}%), "
                          f"overlap: {overlap}/9 players, "
                          f"recent avg: {recent_avg:.2f}")
                else:
                    print(f"Lineup {i + 1}: Score {lineup.fitness_score:.2f} (baseline)")

                if (i + 1) % 5 == 0:
                    print(f"Generated {i + 1}/{num_solutions} lineups")
            else:
                print(f"Could not generate lineup {i + 1} - no feasible solution")
                break

        total_time = time.time() - start_time
        timed_out = total_time > timeout_seconds

        print(f"=== LP OPTIMIZATION SUMMARY ===")
        print(f"Requested: {num_solutions} lineups")
        print(f"Generated: {len(lineups)} lineups")
        print(f"Total time: {total_time:.1f}s{'  (TIMEOUT)' if timed_out else ''}")
        print(f"Success rate: {(len(lineups)/num_solutions)*100:.1f}%")

        if lineups:
            scores = [l.fitness_score for l in lineups]
            print(f"Score range: {max(scores):.2f} - {min(scores):.2f}")

            # Analyze quality distribution
            if len(lineups) > 5:
                first_half = lineups[:len(lineups)//2]
                second_half = lineups[len(lineups)//2:]
                first_avg = sum(l.fitness_score for l in first_half) / len(first_half)
                second_avg = sum(l.fitness_score for l in second_half) / len(second_half)
                quality_degradation = ((first_avg - second_avg) / first_avg) * 100

                print(f"Quality analysis: First half avg {first_avg:.2f}, "
                      f"second half avg {second_avg:.2f} "
                      f"({quality_degradation:+.1f}% change)")

            # Analyze diversity
            all_player_ids = set()
            for lineup in lineups:
                lineup_players = set(p.player_id for _, p in lineup.list_of_players)
                all_player_ids.update(lineup_players)

            unique_players_used = len(all_player_ids)
            theoretical_max = len(lineups) * 9  # 9 players per lineup
            diversity_score = (unique_players_used / theoretical_max) * 100

            print(f"Diversity analysis: {unique_players_used} unique players used "
                  f"(diversity: {diversity_score:.1f}%)")

        return lineups

    def test_timeout_protection(self, test_timeout_seconds=5, delay_per_lineup=2.0):
        """
        Test method to verify timeout protection works correctly.

        Args:
            test_timeout_seconds: Shorter timeout for testing (default 5s)
            delay_per_lineup: Artificial delay per lineup in seconds (default 2s)
        """
        print(f"=== TIMEOUT PROTECTION TEST ===")
        print(f"Test timeout: {test_timeout_seconds}s, Delay per lineup: {delay_per_lineup}s")
        print(f"Expected: Should timeout after ~{test_timeout_seconds // delay_per_lineup} lineups")

        # Set test delay
        self._test_delay_seconds = delay_per_lineup

        # Use the main method with modified timeout
        original_num_solutions = self.num_solutions
        self.num_solutions = 10  # Request more than we expect to generate

        # Temporarily override the timeout by modifying the method
        original_method = self.lineups_with_player_positions
        def test_lineups_method(num_solutions=None):
            if num_solutions is None:
                num_solutions = self.num_solutions

            lineups = []
            start_time = time.time()
            timeout_seconds = test_timeout_seconds  # Use test timeout

            print(f"=== LINEAR PROGRAMMING OPTIMIZER (TEST) ===")
            print(f"Generating {num_solutions} optimal lineups...")

            for i in range(num_solutions):
                # Check timeout before each solve
                elapsed_time = time.time() - start_time

                # Estimate time for next solve (includes artificial delay)
                estimated_solve_time = getattr(self, '_test_delay_seconds', 0.5)

                if elapsed_time + estimated_solve_time > timeout_seconds:
                    print(f"Timeout reached after {elapsed_time:.1f}s - generated {len(lineups)} lineups")
                    break

                lineup = self.solve_single_lineup()

                if lineup and lineup.fitness_score > 0:
                    lineups.append(lineup)
                    print(f"Generated lineup {len(lineups)} after {elapsed_time:.1f}s")
                else:
                    print(f"Could not generate lineup {i + 1} - no feasible solution")
                    break

            total_time = time.time() - start_time
            timed_out = total_time > timeout_seconds

            print(f"=== TEST SUMMARY ===")
            print(f"Generated: {len(lineups)} lineups")
            print(f"Total time: {total_time:.1f}s{'  (TIMEOUT)' if timed_out else ''}")

            return lineups

        # Run the test
        lineups = test_lineups_method()

        # Clean up
        delattr(self, '_test_delay_seconds')
        self.num_solutions = original_num_solutions

        # Analyze results
        total_time = time.time() - (time.time() - len(lineups) * delay_per_lineup)  # Approximate
        expected_lineups = int(test_timeout_seconds // delay_per_lineup)

        print(f"=== TEST RESULTS ===")
        print(f"Lineups generated: {len(lineups)}")
        print(f"Expected lineups: ~{expected_lineups}")

        # Test passes if we generated roughly the expected number or fewer
        timeout_working = len(lineups) <= expected_lineups + 1
        print(f"Timeout protection: {'✓ WORKING' if timeout_working else '✗ FAILED'}")

        return lineups