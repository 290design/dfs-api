import random
import itertools
from random import randint
from hopcroftkarp import HopcroftKarp
from collections import Counter
import operator
from timeit import timeit


class Player(object):
    """
    this class gets created dictionary passed by the solver so nothing to add here
    """

    def __init__(self, **player_attributes):
        for key in player_attributes:
            setattr(self, key, player_attributes[key])


class Lineup(object):
    fitness_score = None
    list_of_players = []
    max_length = None
    max_per_position = None
    max_salary = None
    stack = None
    max_per_team = None
    exclude_opp_positions = None

    def __init__(self, list_of_players=None, lineup_rules=None, options=None):
        self.list_of_players = list_of_players  # player objects
        self.max_length = len(lineup_rules["roster_slots"])
        self.max_per_position = Counter(lineup_rules["roster_slots"])

        self.max_salary = lineup_rules.get("max_budget", None)
        self.min_salary = lineup_rules.get('min_budget', 0)

        self.options = options
        self.roster_slots = lineup_rules['roster_slots']
        self.multipliers = lineup_rules.get('multipliers', {})
        # Set max per team to roster slot count if value not sent in request
        self.max_per_team = options.get('max_per_team', len(self.roster_slots))

        if options:
            self.stack = options.get('stack', None)
            opponent_pos_data = options.get('exclude_opp_roster_slots', None)
            if opponent_pos_data and opponent_pos_data.get('enabled') is True:
                self.exclude_opp_positions = opponent_pos_data.get('roster_slot_map', None)

        # create list of player_ids
        self.playerid_list = [x[1].player_id for x in list_of_players]

        # check lineup fitness
        try:
            self.fitness_score = self._fitness(list_of_players)
        except Exception as e:
            self.fitness_score = -1
            # print(e)

    def _fitness(self, players):
        """
        this checks to see if the lineup passes all of the fitness rules
        :return: returns fitness score
        """

        # check to see if there are enough players
        if len(players) != self.max_length:
            return -1

        # check for duplicates
        if len(self.playerid_list) != len(set(self.playerid_list)):
            return -1

        # check for too many on a team
        if self.max_per_team:
            team_count = Counter([x[1].team_abbr for x in self.list_of_players]).most_common(1)[0][1]
            if team_count > self.max_per_team:
                return -1

        templineup = list(players)
        roster_slots_with_index = []
        for rosterindex in range(len(self.roster_slots)):
            roster_slots_with_index.append((self.roster_slots[rosterindex], rosterindex))

        for player_item in self.list_of_players:
            assigned_roster_slot = player_item[0]
            player_roster_slots = player_item[1].roster_slots
            if assigned_roster_slot not in player_roster_slots:
                return -1

        # magic is done here
        tempplayerdict = {
            position: [player[1].player_id for player in templineup if position[0] in player[1].roster_slots] for
            position in roster_slots_with_index}
        graph = HopcroftKarp(tempplayerdict).maximum_matching()
        graph_playerid_list = [playeridingraph for playeridingraph in graph if playeridingraph in self.playerid_list]
        if len(graph_playerid_list) != len(templineup):
            return -1

        # check salary
        if self.max_salary:
            lineup_sum = sum(player[1].salary for player in players)
            if lineup_sum > self.max_salary or lineup_sum < self.min_salary:
                return -1

        # # check for stack
        # if self.stack:
        #     if self.stack['enabled']:
        #         if not self.is_stack():
        #             return -1

        # set lineup score by value
        value_sum = sum(player[1].value * self.multipliers.get(player[0], 1.0) for player in players)
        score = value_sum if value_sum > 1 else -1

        # check for stack
        if self.stack:
            if self.stack['enabled']:
                if not self.is_stack():
                    score = score * 0.75  # reduce value of lineups that don't meet stack criteria

        invalid_opp_matchup_count = self.invalid_opp_position_count()
        if invalid_opp_matchup_count > 0:
            # Reduce score by 15% + 10% for every opponent matchup. Makes lineups with multiple of conflicts worth less
            penalty = 0.15 + 0.1 * invalid_opp_matchup_count
            score = score * (1 - penalty)

        return score

    def invalid_opp_position_count(self):
        if self.exclude_opp_positions is None:
            return 0

        opp_count = 0
        for lineup_item in self.list_of_players:
            player = lineup_item[1]
            opp_positions = self.exclude_opp_positions.get(player.position_abbr)
            if opp_positions is None:
                continue
            for other_lineup_item in self.list_of_players:
                other_player = other_lineup_item[1]
                if player.opponent_abbr == other_player.team_abbr and other_player.position_abbr in opp_positions:
                    opp_count += 1

        return opp_count

    def rescore_fitness(self):
        """
        checks the fitness again
        :return: score
        """
        self.fitness_score = self._fitness(self.list_of_players)
        return self.fitness_score

    def is_stack(self):
        """
        this method returns value of true if the line up meets the stacking regulations
        groups are for something like QB-WR or QB-TE or QB-RB
        fields are like for team_abbr
        :return: true if it is stacked
        """
        groups = self.stack['groups']

        required_roster_slots = set()
        optional_roster_slots = set()
        combined_roster_slots = set()

        for group in groups:
            value = group['value']
            required = group['required']
            combined_roster_slots.add(value)
            if required:
                required_roster_slots.add(value)
            else:
                optional_roster_slots.add(value)

        total_players = self.stack['total_players']
        team_list = set([x[1].team_abbr for x in self.list_of_players])
        team_position_tuple = {self.list_of_players.index(x): [x[1].team_abbr] for x in self.list_of_players}
        team_graph = HopcroftKarp(team_position_tuple).maximum_matching()
        list_of_players_on_same_team = []
        multiple_position_list = [len(team_position_tuple[team]) for team in team_list if
                                  len(team_position_tuple[team]) > 0]
        min_position_list = min(multiple_position_list)
        for team in team_list:
            if len(team_position_tuple[team]) > min_position_list:
                list_of_players_on_same_team.append((team, team_position_tuple[team]))

        is_valid_stack = False
        for team_sets in list_of_players_on_same_team:
            # player_roster_slots = [getattr(self.list_of_players[x][1], field) for x in team_sets[1]]
            # player_roster_slots = sum(player_roster_slots, [])  # concatenate to one list
            player_roster_slots = [self.list_of_players[x][0] for x in team_sets[1]]
            # valid_roster_slots = combined_roster_slots.intersection(set(player_roster_slots))

            has_required_roster_slots = required_roster_slots.issubset(set(player_roster_slots))
            has_count = len(player_roster_slots) == total_players
            is_valid_stack = has_required_roster_slots & has_count
            if is_valid_stack:
                break

        return is_valid_stack


class NewUniversalGeneticSolver:
    max_per_position = None
    positions = "roster_slots"

    def __init__(self, players=None, options=None, exposure=None, game=None, optimizer_type=None,
                 optimizer_options=None, num_solutions=None):
        self.players = players
        self.options = options
        self.exposure = exposure
        self.game = game
        self.optimize_type = optimizer_type
        if 'roster_config' in game:
            # do special stuff
            random_slots = self.get_roster_config_info(self.game['roster_config'])
            self.max_per_position = Counter(random_slots)
            self.game['roster_slots'] = random_slots
            pass
        else:
            self.max_per_position = Counter(self.game[self.positions])
        self.optimizer_options = optimizer_options
        self.lineup_pool = []
        self.player_pool = []
        self.player_dict = {}
        self.removed_player_pool = []
        self.done = False
        self.parents = []
        self.old_parents = []
        self.fitness_history = []
        if options:
            self.total_exposure = options.get('max_exposure', None)
        else:
            self.total_exposure = None

        self.num_solutions = num_solutions
        self.to_set_exposure = {}

        # faster?
        self.player_pool = [Player(**player) for player in players]
        self.player_dict = {x.player_id: x for x in self.player_pool}

        self.create_lineup_pool()

    def get_roster_config_info(self, roster_config):
        """ takes roster config in
            roster_slots: {"slots": [{ "name": "WK", "min": 1, "max": 4}, { "name": "BAT", "min": 3, "max": 6}] }
        """
        min_position_list = []
        max_remaining_list = []

        for slot in roster_config['slots']:
            temp_list = [slot['name']] * slot['min']
            min_position_list += temp_list
            temp_max = [slot['name']] * (slot['max'] - slot['min'])
            max_remaining_list += temp_max

        filler_list = random.sample(max_remaining_list, roster_config['size']-len(min_position_list))
        final_list = min_position_list + filler_list
        return final_list

    def get_fully_exposed_players(self):
        """
        this returns a list of all players with full exposure
        :return: [list of fully exposed players]
        """
        fully_exposed_ids = [int(k) for k, v in self.exposure.items() if v == 1]
        fully_exposed = []
        for players in fully_exposed_ids:
            fully_exposed += filter(lambda x: x.player_id == players, self.player_pool)
        return fully_exposed

    def create_lineup_pool(self, child_pool=None):
        """
        creates a lineup_pool or a child lineup from parents pool
        :param child_pool:
        :return: This function initializes the lineup pool
        """
        position_list = []
        pool_size = 0
        full_exposure = self.get_fully_exposed_players() if self.exposure else []

        # remove tupled pool for children
        if child_pool is not None:
            player_pool = [x[1] for x in child_pool]
        else:
            player_pool = self.player_pool

        child_counter = 0

        stack_enabled = False
        stack_teams = None
        stack_teams_players_by_position = {}
        if self.stack_enabled():
            if self.options.get('stack', False):
                stack_options = self.options['stack']
                if stack_options['enabled']:
                    stack_enabled = True
                    stack_teams = stack_options.get('teams', set([x.team_abbr for x in player_pool]))
                    # if stack_teams is None or len(stack_teams) == 0:
                    #     stack_teams = set([x.team_abbr for x in player_pool])

                    # generate player position maps for each team once for performance
                    for team in stack_teams:
                        # create a tuple of players by position
                        team_players_by_position = {position: [player for player in player_pool if
                                                               position in getattr(player,
                                                                                   self.positions) and team == getattr(
                                                                   player, "team_abbr")] for position in
                                                    self.max_per_position.keys()}
                        stack_teams_players_by_position[team] = team_players_by_position

        # create a dictionary of players by position
        players_by_position = {}
        for position in self.max_per_position.keys():
            players_by_position[position] = [player for player in player_pool if
                                             position in getattr(player, self.positions)]

        max_iterations = 10000  # Allow loop to bail if all lineups being generated are infeasible
        iteration_counter = 0

        while pool_size < self.optimizer_options["pop_size"] and child_counter < 500 \
                and iteration_counter < max_iterations:
            iteration_counter += 1

            if stack_enabled:
                team = random.choice(list(stack_teams))
                team_players_by_position = stack_teams_players_by_position[team]
                temp_lineup = self.generate_lineup_stacked(self.options, team, players_by_position,
                                                           team_players_by_position, player_pool)
            else:
                temp_lineup = self.generate_lineup(players_by_position, player_pool)

            # add lineup to lineup_pool if it is fit
            lineup_size = len(temp_lineup.list_of_players)
            retry_count = 0
            retry_max = int(lineup_size / 2)
            while temp_lineup.fitness_score <= 0 and retry_count < retry_max:
                # Remove half of the players and recreate the lineup
                retry_count += 1
                player_list = temp_lineup.list_of_players
                players_to_remove = int(lineup_size / 2)
                for i in range(players_to_remove):
                    random_index = random.randint(0, len(player_list) - 1)
                    player_list.pop(random_index)
                temp_lineup = self.complete_lineup(roster_slots=temp_lineup.roster_slots,
                                                   player_list=player_list,
                                                   players_by_position=players_by_position,
                                                   player_pool=player_pool,
                                                   max_exposure=self.total_exposure)

            if temp_lineup.fitness_score > 0:
                if child_pool is not None:
                    return temp_lineup
                else:
                    self.lineup_pool.append(temp_lineup)

            pool_size = len(child_pool) if child_pool is not None else len(self.lineup_pool)
            if child_pool is not None:
                child_counter += 1

    def stack_enabled(self):
        if self.options:
            if self.options.get('stack', False):
                stack_options = self.options['stack']
                if stack_options['enabled']:
                    return True

        return False

    def complete_lineup(self, roster_slots, player_list, players_by_position, player_pool, max_exposure):
        # there are 1 or more spots that need to be filled by players without exposure
        # determine which roster slots are empty
        empty_roster_slots = roster_slots.copy()
        for player_item in player_list:
            roster_slot = player_item[0]
            empty_roster_slots.remove(roster_slot)

        # fill the vacant roster slots
        for roster_slot in empty_roster_slots:
            players = players_by_position[roster_slot]

            player_ids = [item[1].player_id for item in player_list]

            # get eligible players randomly filtered by max exposure
            # ensures no player is added to a lineup at a higher percentage than max exposure
            max_exposure = max_exposure or 1.0
            eligible_players = [player for player in players if
                                player.player_id not in player_ids and random.uniform(0, 1) < max_exposure]

            if len(eligible_players) > 0:
                player = random.choice(eligible_players)
            else:
                player = random.choice(player_pool)

            player_list.append((roster_slot, player))

        # create new lineup object
        return Lineup(list_of_players=player_list, lineup_rules=self.game, options=self.options)

    def generate_lineup(self, players_by_position, player_pool):
        # get shuffled roster slots
        roster_slots = self.game['roster_slots']
        roster_slots = random.sample(roster_slots, len(roster_slots))

        # prefill the lineup with players that have exposure
        player_list = self.assign_exposure_to_roster_slots(roster_slots=roster_slots)

        # if the lineup is complete with exposure players then return
        if len(player_list) >= len(roster_slots):
            return Lineup(list_of_players=player_list, lineup_rules=self.game, options=self.options)

        # there are 1 or more spots that need to be filled by players without exposure
        # determine which roster slots are empty
        empty_roster_slots = roster_slots.copy()
        for player_item in player_list:
            roster_slot = player_item[0]
            empty_roster_slots.remove(roster_slot)

        # fill the vacant roster slots
        return self.complete_lineup(roster_slots=roster_slots,
                                    player_list=player_list,
                                    players_by_position=players_by_position,
                                    player_pool=player_pool,
                                    max_exposure=self.total_exposure)

    def generate_lineup_stacked(self, options, team, players_by_position, team_players_by_position, player_pool):
        max_per_team = options.get('max_per_team', None)
        stack_options = options.get('stack', None)

        total_players = stack_options.get('total_players', 0)

        if total_players == 0:
            groups = stack_options['groups']
            required_count = 0

            for group in groups:
                if group['required']:
                    required_count += 1

            total_players = required_count
            stack_options['total_players'] = total_players

        stack_count_target = min(max_per_team, total_players)

        groups = list(stack_options['groups'])  # create a copy of the groups list

        # get shuffled roster slots
        roster_slots = self.game['roster_slots']
        roster_slots = random.sample(roster_slots, len(roster_slots))

        team_exposure_player_list = self.assign_exposure_to_roster_slots(roster_slots=roster_slots, team=team)
        player_ids = [player_item[1].player_id for player_item in team_exposure_player_list]

        group_roster_slots = []
        for roster_slot in roster_slots:
            group_index = None
            for index, group in enumerate(groups):
                value = group['value']
                if value == roster_slot:
                    group_index = index
                    break

            if group_index is not None:
                group = groups.pop(group_index)
                is_required = group['required']

                # Assign team players with exposure to stack slots
                lineup_player = None
                item_index = None
                for index, item in enumerate(team_exposure_player_list):
                    player = item[1]
                    if roster_slot in player.roster_slots:
                        lineup_player = player
                        item_index = index
                        break

                if lineup_player is not None:
                    group_tuple = (roster_slot, True, is_required, lineup_player)
                    del team_exposure_player_list[item_index]
                else:
                    group_tuple = (roster_slot, True, is_required, None)

                if is_required:
                    group_roster_slots.insert(0, group_tuple)  # place required roster slot stack positions first
                else:
                    group_roster_slots.append(group_tuple)  # roster slot, is stack, is required, player
            else:
                group_roster_slot = (roster_slot, False, False, None)
                group_roster_slots.append(group_roster_slot)  # roster slot, is stack, is required, player

        stack_count = 0
        player_list = []
        for group_roster_slot in group_roster_slots:
            roster_slot = group_roster_slot[0]
            is_stack_slot = group_roster_slot[1]
            player = group_roster_slot[3]
            team_players = team_players_by_position[roster_slot]
            player_ids = [x[1].player_id for x in player_list]
            eligible_team_players = [player for player in team_players if player.player_id not in player_ids]

            if is_stack_slot and stack_count < stack_count_target and player:
                player_list.append((roster_slot, player))
                stack_count += 1
            elif is_stack_slot and stack_count < stack_count_target and len(eligible_team_players) > 0:
                player = random.choice(eligible_team_players)
                player_list.append((roster_slot, player))
                stack_count += 1
            else:
                # Assign players with exposure first. Then choose random players last.
                if player is None:
                    exposure_player_list = self.assign_exposure_to_roster_slots(roster_slots=[roster_slot], team=None,
                                                                                exclude_team=team,
                                                                                exclude_player_ids=player_ids)
                    if len(exposure_player_list) > 0:
                        player = exposure_player_list[0][1]
                    else:
                        players = players_by_position[roster_slot]
                        non_team_players = [player for player in players if
                                            getattr(player, "team_abbr") != team and player.player_id not in player_ids]
                        if len(non_team_players) > 0:
                            player = random.choice(non_team_players)
                        else:
                            player = random.choice(player_pool)

                player_list.append((roster_slot, player))

        return Lineup(list_of_players=player_list, lineup_rules=self.game, options=self.options)

    def assign_exposure_to_roster_slots(self, roster_slots, team=None, exclude_team=None, exclude_player_ids=None):
        player_list = []

        # get shuffled roster slots
        available_roster_slots = random.sample(roster_slots, len(roster_slots))

        exposure = []

        if self.exposure:
            exposure = random.sample(self.exposure.items(), len(self.exposure))

        for player_exposure in exposure:
            player_id = int(player_exposure[0])
            player = self.player_dict.get(player_id)

            if player is None:
                continue

            if team and player.team_abbr != team:
                continue

            if exclude_team and player.team_abbr == exclude_team:
                continue

            if exclude_player_ids and player.player_id in exclude_player_ids:
                continue

            # Ramdomly choose if this player will be added based on exposure.
            if player_exposure[1] != 1 and random.uniform(0, 1) > player_exposure[1]:
                continue

            eligible_roster_slots = list(set(available_roster_slots) & set(player.roster_slots))

            if len(eligible_roster_slots) > 0:
                selected_roster_slot = eligible_roster_slots[0]
                player_list.append((selected_roster_slot, player))
                available_roster_slots.remove(selected_roster_slot)

        return player_list

    def grade(self, evolution=None):
        """
        This grades the pool of lineups and gives the average
        :param evolution: so that it can print out the progress of the solver, but also spits out the ending
        :return: n/a
        """
        fitness_sum = 0
        index_list = []
        for index, x in enumerate(self.lineup_pool):
            if x:
                fitness_sum += x.fitness_score
            else:
                # self.lineup_pool.pop(index)
                index_list.append(index)
                index -= 1
        pop_fitness = fitness_sum / self.optimizer_options["pop_size"]
        self.fitness_history.append(pop_fitness)

        self.lineup_pool = [x for x in self.lineup_pool if x]

        self.lineup_pool.sort(key=lambda x: x.fitness_score, reverse=True)

        if len(self.lineup_pool) == 0:
            print("Evolution", evolution, " No Lineups Generated")
            return

        if pop_fitness == self.lineup_pool[0].fitness_score:
            self.done = True

        if evolution is not None:
            if evolution % 5 == 0:
                print("Evolution", evolution, "Pop fitness:", pop_fitness, "best score: ",
                      self.lineup_pool[0].fitness_score)

    def mate(self):
        """
        mates two parents to breed children
        :return: mutates the parent pool
        """

        target_children_size = self.optimizer_options["pop_size"] - len(self.parents)
        children = []

        if len(self.parents) > 1:
            max_iterations = target_children_size * 10
            iteration_counter = 0
            while iteration_counter < max_iterations and len(children) < target_children_size:
                iteration_counter += 1
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                if father != mother:
                    players_in_parents = []
                    players_in_parents += father.list_of_players
                    players_in_parents += mother.list_of_players
                    child = self.create_lineup_pool(players_in_parents)
                    children.append(child)
            self.lineup_pool = self.parents + children

    def mutate(self):
        """
        mutates the parents for proper genetic algorithm
        :return: n/a
        """
        if len(self.parents) == 0:
            return

        lineup_size = len(self.parents[0].list_of_players) - 1
        pool_size = len(self.parents) - 1
        player_set = []
        for lineup in self.parents:
            player_set += list(set(lineup.list_of_players) - set(player_set))

        for lineup in self.parents:
            if self.optimizer_options["mutate"] > random.random():
                num_of_times_to_mutate = 3
                for mutations in range(num_of_times_to_mutate):
                    mutate_position = randint(0, lineup_size)
                    while True:
                        random_lineup = self.parents[randint(0, pool_size)]
                        random_player = random_lineup.list_of_players[randint(0, lineup_size)]
                        player_position = lineup.list_of_players[randint(0, lineup_size)]
                        if player_position[0] in getattr(random_player[1], self.positions):
                            lineup.list_of_players[mutate_position] = random_player
                            lineup.rescore_fitness()
                            break

    def select_parents(self):
        """
        selects the fittest parents based on the parameters passed
        :return: modifies parent list
        """

        # sort lineup_pool
        self.lineup_pool.sort(key=lambda x: x.fitness_score, reverse=True)

        retain_length = self.optimizer_options["retain"] * len(self.lineup_pool)
        self.parents = self.lineup_pool[:int(retain_length)]

        # add some random lineups to parents
        unretained_lineups = self.lineup_pool[int(retain_length):]
        for unretained_lineup in unretained_lineups:
            if self.optimizer_options["random_select"] > random.random():
                self.parents.append(unretained_lineup)

    def evolve(self):
        self.select_parents()
        self.mutate()
        self.mate()
        # reset parents
        self.old_parents += list(self.parents[:int(0.2 * len(self.parents))])
        self.parents = []

    def lineups_with_player_positions(self, num_solutions=25):
        """
        gets the lineup list in dictionary format for return
        :return: returns a dict list with players
        """
        lineups = []
        print("Number of lineups in before set: ", len(self.old_parents))
        self.old_parents.sort(key=lambda x: x.fitness_score, reverse=True)

        old_parents_copy = self.old_parents.copy()
        # remove duplicate lineups
        templist = []
        templist_tuple = []
        for index, l in enumerate(old_parents_copy):
            temp_tuple = set([x[1].player_id for x in l.list_of_players])
            if temp_tuple not in templist:
                templist.append(temp_tuple)
                templist_tuple.append(l)
            else:
                old_parents_copy.pop(index)

        final_list = list(templist_tuple)
        final_list.sort(key=lambda x: x.fitness_score, reverse=True)
        print("Number of lineups in after set: ", len(old_parents_copy))
        master_list = final_list[:]

        # check for pool exposure
        if self.exposure:
            index_list = self.is_exposure(final_list, num_solutions)
            final_list = [final_list[x] for x in index_list]

        if self.total_exposure:
            first_set_of_solutions = int(round((num_solutions * self.total_exposure) - 0.5))

            final = final_list[:first_set_of_solutions] if first_set_of_solutions < len(final_list) else final_list[:]

            #  add items to final list and check the exposure counts if they are too high skip the list
            begin_list_num = len(final)
            end_list_num = len(master_list)

            for exp_list in range(begin_list_num, end_list_num):
                list_to_check = final[:]
                list_to_check.append(master_list[exp_list])
                templist = [[y[1].player_id for y in x.list_of_players] for x in list_to_check]
                over_exposed_list = self.check_total_exposure(templist)
                # if self.exposure:
                #     over_exposed_list[0].subtract(self.exposure.keys())

                # check if item is greater than first set of solutions
                max_num_of_player = max(over_exposed_list[0].values() or [0])
                # players_over_list = []
                # for k,v in over_exposed_list[0].items():
                #     if v >= first_set_of_solutions + 1:
                #         players_over_list.append(k)

                players_over_list = [k for (k, v) in over_exposed_list[0].items() if v >= first_set_of_solutions + 1]
                is_in_templist = False
                if len(set(players_over_list) & set(templist[len(templist) - 1])) > 0:
                    is_in_templist = True

                if (max_num_of_player > first_set_of_solutions + 1) and is_in_templist:
                    continue
                else:
                    final.append(master_list[exp_list])
                if len(final) >= num_solutions + 2:
                    break

            final_list = final

        # for lineup in final_list:
        #     if lineup.rescore_fitness() > 0:
        #         lineups.append([{"playerid": player[1].player_id, "position": player[0]}
        #                         for player in lineup.list_of_players])
        #
        # return lineups

        result = []
        for lineup in final_list:
            if lineup.rescore_fitness() > 0:
                result.append(lineup)

        return result

    def is_exposure(self, lineups, num_solutions, max_exposure=None):
        """
        returns the value of exposure fitness
        :return: list of lineups to use
        """

        # get shuffled roster slots
        roster_slots = self.game['roster_slots']
        roster_slots = random.sample(roster_slots, len(roster_slots))

        playerid_list_in_pool = [lineup.playerid_list for lineup in lineups]
        # count_of_playerids = Counter(i for i in list(itertools.chain.from_iterable(playerid_list_in_pool)))
        percent_exposure_list = [list() for _ in range(num_solutions)]

        # create lineup templates
        for x in range(num_solutions):
            player_list = self.assign_exposure_to_roster_slots(roster_slots=roster_slots)
            player_ids = [item[1].player_id for item in player_list]
            percent_exposure_list[x] = player_ids

        # now return a list of indexes that match the template
        index_list = []
        for template in percent_exposure_list:
            for i, match in enumerate(playerid_list_in_pool):
                if set(template).issubset(set(match)) and i not in index_list:
                    index_list.append(i)
                    break
        return index_list

    def check_exposure(self, checked_exposure, lineup_count):
        """
        returns adjusted lineup pool
        :return:
        """
        for players in checked_exposure.items():
            players_count = players[1]
            player = players[0][1]
            player_exposure = float(players_count) / lineup_count
            if player_exposure > self.total_exposure:
                self.removed_player_pool.append(player)
                self.player_pool.remove(player)

    def check_total_exposure(self, final_list):
        """
        returns a list of the total player exposures
        :param final_list:
        :return:
        """
        # reduced_list = reduce(operator.add, enumerate(final_list))
        player_index_dict = {}
        reduced_list = []
        # get list of items in list then create dictionary where the indexes of each item are stored
        for index, list_of_players in enumerate(final_list):
            for item in list_of_players:
                if self.exposure and self.exposure.get(str(item)) == 1:
                    continue

                reduced_list.append(item)
                # player_index_dict_item = player_index_dict.get(item, {item: []})
                if player_index_dict.get(item, None):
                    player_index_dict[item].append(index)
                else:
                    player_index_dict[item] = [index]

        newlist = Counter(reduced_list)
        for player in newlist.items():
            player_count = player[1]
            player_exposure = float(player_count) / len(final_list)
            exposure = self.exposure and self.exposure.get(str(player[0])) or self.total_exposure
            # if player_exposure > self.total_exposure:
            if player_exposure <= exposure:
                continue

            if player_exposure > self.total_exposure / 1.25:
                self.to_set_exposure[player[0]] = player_count

        return newlist, player_index_dict
