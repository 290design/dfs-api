import random
import itertools
from random import randint
from hopcroftkarp import HopcroftKarp
from collections import Counter
import operator


class Player(object):
    """
    Player class for DFS optimization
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
        self.max_per_team = options.get('max_per_team', len(self.roster_slots)) if options else len(self.roster_slots)

        # Initialize all attributes to prevent AttributeError
        self.stack = None
        self.exclude_opp_positions = None

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

        # check salary constraints
        lineup_salary = sum([x[1].salary for x in players])
        if lineup_salary > self.max_salary:
            return -1

        if lineup_salary < self.min_salary:
            return -1

        # Calculate total value with multipliers
        total_value = 0
        for slot, player in players:
            value = getattr(player, 'value', 0)
            multiplier = self.multipliers.get(slot, 1.0)
            total_value += value * multiplier

        return total_value

    def to_dict(self, django_format=False):
        """Convert lineup to dictionary format for API response"""
        if django_format:
            # Django format: array of simple player objects
            return [
                {
                    "playerid": player.player_id,
                    "slateplayerid": getattr(player, 'slate_player_id', 0),
                    "position": slot
                }
                for slot, player in self.list_of_players
            ]
        else:
            # Original detailed format (for debugging/testing)
            return {
                'players': [
                    {
                        'player_id': player.player_id,
                        'name': getattr(player, 'name', ''),
                        'position': slot,
                        'team': getattr(player, 'team_abbr', ''),
                        'salary': getattr(player, 'salary', 0),
                        'value': getattr(player, 'value', 0)
                    }
                    for slot, player in self.list_of_players
                ],
                'total_salary': sum([getattr(player, 'salary', 0) for _, player in self.list_of_players]),
                'total_value': self.fitness_score,
                'fitness_score': self.fitness_score
            }