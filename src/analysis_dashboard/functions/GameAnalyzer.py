import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import re
import os
from tabulate import tabulate
from pyvis.network import Network
import streamlit as st

class HearthstoneLogAnalyzer:
    """
    A class to parse Hearthstone environment log data from a JSON-based log file.
    It provides convenient methods to explore gameplay statistics, 
    reward progression, action distributions, and more.
    """

    def __init__(self, file_path: str, experiment_name: str):
        """
        Initialize the HearthstoneLogAnalyzer by providing the path to the log file.

        :param file_path: Path to the log file containing JSON lines
        """
        self.file_path = file_path
        self.experiment_name = experiment_name
        self.data = []
        self.combo_networks = []
        self.df = None  # Pandas DataFrame to store the data
        self._load_data()
        self._create_dataframe()
        
        # Define placeholders for resulting figures
        # Reward figures
        self.fig_cumulative_reward_by_player = None
        self.fig_reward_by_game_grid = None
        self.fig_reward_by_game_by_player = None
        self.fig_action_distribution = None
        
        # Overall info
        self.player_info = None
        self.events_logged = None
        self.unique_games = None
        
        # Winrate figures
        self.winrate_df = None
        self.win_matrix = None
        self.win_matrix_heatmap_fig_p1 = None
        self.win_matrix_heatmap_fig_p2 = None
        
        # Class usage figures
        self.fig_class_usage = None
        
        # Agent type performance figures
        self.fig_agent_type_performance = None
        
        # Action string distribution figures
        self.fig_action_string_distributions = []
        
        # Combo distribution figures
        self.fig_combo_distribution_lengths = None
        self.fig_combo_distribution_grid = None
        self.fig_combo_distribution_counts = None
        
        # Top combo data
        self.top_combos = {}

    def _load_data(self) -> None:
        """
        Loads data from the specified file. Each line in the file should be valid JSON.
        """
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event_data = json.loads(line)
                    self.data.append(event_data)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")

    def _create_dataframe(self) -> None:
        """
        Converts the loaded JSON data into a Pandas DataFrame for easier analysis.
        """
        if len(self.data) > 0:
            self.df = pd.DataFrame(self.data)
            # Flatten nested dictionaries (such as 'action') if you want direct columns
            action_cols = ["action_type", "card_index", "attacker_index", 
                           "target_index", "discover_index", "choose_index"]
            for col in action_cols:
                self.df[col] = self.df["action"].apply(lambda x: x[col] if isinstance(x, dict) else None)
            # Drop the original action column if you prefer to keep flattened columns
            # self.df.drop(columns=["action"], inplace=True)
        else:
            print("No valid data found in log file.")

    
    def get_summary_statistics(self) -> None:
        """
        Prints a set of summary statistics about the log data,
        including the total number of events, the number of games, 
        total rewards, etc.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return

        print("------ Summary Statistics ------")
        print(f"Total events logged: {len(self.df)}")
        print(f"Unique game numbers: {self.df['game_number'].nunique()}")
        
        # Print players and their classes
        print("Players and the agent types:")
        print(self.df[["player", "agent_type"]].drop_duplicates())
              
        print(f"Total cumulative reward: {self.df['reward'].sum()}")
        print(f"Number of 'done' (terminal) states: {self.df[self.df['done']].shape[0]}")        
        
        self.player_info = self.df[["player", "agent_type"]].drop_duplicates()        
        self.events_logged = len(self.df)
        self.unique_games = self.df['game_number'].nunique()
        
    def get_winrates(self, folder_path: str, experiment_name: str):
        """
        Calculates win rates per player-class-deck and constructs a win matrix,
        then saves both as CSV files.

        This function creates a complete table of game/player combinations, merges in
        wins and losses (filling in missing losses when only one win is recorded), aggregates
        overall win rates per (player, card_class, deck), and builds a win matrix for two players.
        """
        import os
        import pandas as pd

        # --- 1. Build a complete table for each game and player ---
        # Get unique details (card_class and deck) for each game/player.
        details = self.df.drop_duplicates(subset=['game_number', 'player'])[
            ['game_number', 'player', 'card_class', 'deck']
        ]
        # Build all combinations of games and players.
        games = self.df['game_number'].unique()
        players = self.df['player'].unique()
        all_combinations = pd.MultiIndex.from_product(
            [games, players], names=['game_number', 'player']
        ).to_frame(index=False)
        # Merge to get the card_class and deck for each game/player.
        full_stats = all_combinations.merge(details, on=['game_number', 'player'], how='left')

        # --- 2. Compute wins and losses for each game/player ---
        wins_df = (
            self.df[self.df['player_result'] == 'won']
            .groupby(['game_number', 'player'])
            .size()
            .rename('wins')
            .reset_index()
        )
        losses_df = (
            self.df[self.df['player_result'] == 'lost']
            .groupby(['game_number', 'player'])
            .size()
            .rename('losses')
            .reset_index()
        )
        full_stats = (
            full_stats.merge(wins_df, on=['game_number', 'player'], how='left')
                    .merge(losses_df, on=['game_number', 'player'], how='left')
        )
        full_stats[['wins', 'losses']] = full_stats[['wins', 'losses']].fillna(0)

        # For each game, if one player has a win recorded and the other has no record, assume that missing record is a loss.
        def fix_game(group):
            if group['wins'].sum() > 0:  # game has a win recorded
                missing_mask = (group['wins'] == 0) & (group['losses'] == 0)
                group.loc[missing_mask, 'losses'] = 1
            if group['losses'].sum() > 0:  # game has a loss recorded
                missing_mask = (group['losses'] == 0) & (group['wins'] == 0)
                group.loc[missing_mask, 'wins'] = 1                
            return group

        full_stats = full_stats.groupby('game_number', group_keys=False).apply(fix_game)

        # --- 3. Aggregate overall win rates per (player, card_class, deck) ---
        agg_stats = (
            full_stats.groupby(['player', 'card_class', 'deck'])
                    .agg({'wins': 'sum', 'losses': 'sum'})
                    .reset_index()
        )
        # If player 1 has more wins than player 2 has losses, overwrite player 1's wins with player 2's losses. (and vice versa)
        # If player 1 has less losses than player 2 has wins, overwrite player 1's losses with player 2's wins. (and vice versa)
        
        
        agg_stats['Win Rate (%)'] = (
            100 * agg_stats['wins'] / (agg_stats['wins'] + agg_stats['losses'])
        ).round(2)
        agg_stats = agg_stats.sort_values(by=['player', 'Win Rate (%)'], ascending=[True, False])

        self.winrate_df = agg_stats
        
        # Add the win rate overall to the self.player_info DataFrame
        info_stats = agg_stats.copy().groupby('player').agg({'Win Rate (%)': 'mean'}).reset_index()
        info_stats['Win Rate (%)'] = info_stats['Win Rate (%)'].round(2)
        self.player_info = pd.merge(self.player_info, info_stats[['player', 'Win Rate (%)']], on='player', how='left')
        
        winrate_csv = os.path.join(folder_path, f"winrates_{experiment_name}.csv")
        agg_stats.to_csv(winrate_csv, index=False)

        # --- 4. Build the win matrix ---
        # For head-to-head analysis between two players, assign opponent details.
        # Merge full_stats with itself on game_number to get each player's opponent info.
        opponents = full_stats[['game_number', 'player', 'card_class', 'deck']].rename(
            columns={
                'player': 'opponent',
                'card_class': 'card_class_opponent',
                'deck': 'deck_opponent'
            }
        )
        matchups = pd.merge(full_stats, opponents, on='game_number')
        # Exclude self-matchups.
        matchups = matchups[matchups['player'] != matchups['opponent']]

        # Separate matchups for Player 1 and Player 2.
        matchups_p1 = matchups[matchups['player'] == "Player 1"]
        matchups_p2 = matchups[matchups['player'] == "Player 2"]

        # Define the card/deck tuples (order matters for rows and columns).
        card_decks = [
            ("Druid", "Token/Swarm"),
            ("Druid", "Jade Golem"),
            ("Hunter", "Midrange/Beast"),
            ("Hunter", "Deathrattle"),
            ("Mage", "Tempo/Elemental"),
            ("Mage", "Big Spell/Control"),
            ("Paladin", "Silver Hand Recruit Aggro"),
            ("Paladin", "Control/N'Zoth"),
            ("Priest", "Dragon"),
            ("Priest", "Combo/Resurrection"),
            ("Rogue", "Tempo/Weapon"),
            ("Rogue", "Miracle/Gadgetzan"),
            ("Shaman", "Totem & Overload Synergy"),
            ("Shaman", "Control/Big Spells"),
            ("Warlock", "Zoo/Discard Mix"),
            ("Warlock", "Control/Demon"),
            ("Warrior", "Tempo/Taunt"),
            ("Warrior", "Mech/Control")
        ]

        # Create an empty win matrix DataFrame with MultiIndex rows and columns.
        win_matrix = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(card_decks, names=['card_class', 'deck']),
            columns=pd.MultiIndex.from_tuples(card_decks, names=['card_class', 'deck'])
        )
        win_matrix = win_matrix.fillna("0/0").astype(str)

        # Fill in the win matrix by aggregating wins for each matchup.
        # For a given cell, the value "X/Y" means:
        # - X: total wins of Player 1 when using (idx) against an opponent using (col)
        # - Y: total wins of Player 2 when using (col) against an opponent using (idx)
        for idx in card_decks:
            for col in card_decks:
                # Count wins from Player 1's perspective.
                p1_wins = matchups_p1[
                    (matchups_p1['card_class'] == idx[0]) & (matchups_p1['deck'] == idx[1]) &
                    (matchups_p1['card_class_opponent'] == col[0]) & (matchups_p1['deck_opponent'] == col[1])
                ]['wins'].sum()
                # Count wins from Player 2's perspective (reverse the matchup).
                p2_wins = matchups_p2[
                    (matchups_p2['card_class'] == col[0]) & (matchups_p2['deck'] == col[1]) &
                    (matchups_p2['card_class_opponent'] == idx[0]) & (matchups_p2['deck_opponent'] == idx[1])
                ]['wins'].sum()
                win_matrix.at[idx, col] = f"{int(p1_wins)}/{int(p2_wins)}"

        self.win_matrix = win_matrix
        
        win_matrix_csv = os.path.join(folder_path, f"win_matrix_{experiment_name}.csv")
        win_matrix.to_csv(win_matrix_csv)        
    
        
        # Player 1 win matrix heatmap
        
        # Create the heatmap DataFrame with numeric percentages instead of strings
        win_matrix_heatmap_p1 = win_matrix.apply(
            lambda col: col.map(
                lambda x: round(
                    int(x.split('/')[0]) / (int(x.split('/')[0]) + int(x.split('/')[1])) * 100, 2
                ) if (int(x.split('/')[0]) + int(x.split('/')[1])) > 0 else np.nan
            )
        )

        
        fig_p1, ax_p1 = plt.subplots(figsize=(15, 10))
        
        # Generate a heatmap of the win matrix, masking the N/A values
        sns.heatmap(win_matrix_heatmap_p1, annot=True, fmt=".2f", cmap="viridis", ax=ax_p1, mask=win_matrix_heatmap_p1.isnull())

        ax_p1.set_title("Win Matrix Heatmap Player 1")
        ax_p1.set_xlabel("Player 2 Card/Deck")
        ax_p1.set_ylabel("Player 1 Card/Deck")
        
        for item in ([ax_p1.title, ax_p1.xaxis.label, ax_p1.yaxis.label] + ax_p1.get_xticklabels() + ax_p1.get_yticklabels()):
            item.set_fontsize(18)
            
        self.win_matrix_heatmap_fig_p1 = fig_p1
        
        # Player 2 win matrix heatmap
        
        # Create the heatmap DataFrame with numeric percentages instead of strings
        win_matrix_heatmap_p2 = win_matrix.apply(
            lambda col: col.map(
                lambda x: round(
                    int(x.split('/')[1]) / (int(x.split('/')[0]) + int(x.split('/')[1])) * 100, 2
                ) if (int(x.split('/')[0]) + int(x.split('/')[1])) > 0 else np.nan
            )
        )
        
        fig_p2, ax_p2 = plt.subplots(figsize=(15, 10))
        
        # Generate a heatmap of the win matrix, masking the N/A values
        sns.heatmap(win_matrix_heatmap_p2, annot=True, fmt=".2f", cmap="viridis", ax=ax_p2, mask=win_matrix_heatmap_p2.isnull())
        
        ax_p2.set_title("Win Matrix Heatmap Player 2")
        ax_p2.set_xlabel("Player 2 Card/Deck")
        ax_p2.set_ylabel("Player 1 Card/Deck")
        
        for item in ([ax_p2.title, ax_p2.xaxis.label, ax_p2.yaxis.label] + ax_p2.get_xticklabels() + ax_p2.get_yticklabels()):
            item.set_fontsize(18)
            
        self.win_matrix_heatmap_fig_p2 = fig_p2
        
    def plot_cumulative_reward_by_player(self) -> None:
        """
        Plots the cumulative reward over the full sequence of events (steps),
        with one line per player.
        The x-axis is simply the event order in which logs appear.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return

        # Make a copy so we don't alter the main DataFrame
        df_plot = self.df.copy()

        # Sort by the order in which events actually occurred (the DataFrame index).
        # If your DataFrame is already in chronological order, you can skip this.
        df_plot.sort_index(inplace=True)

        # Create a "step" column that identifies each event in sequence
        df_plot["step"] = range(len(df_plot))

        # Group by the agent dimension you prefer. Here I’ll use "agent_type".
        # If you want to do it by "player", then replace agent_type with player.
        df_plot["cumulative_reward"] = (
            df_plot.groupby("player")["reward"]
                .cumsum()
        )

        # Plot: one line per agent_type, showing how its cumulative reward evolves
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(ax=ax, data=df_plot, x="step", y="cumulative_reward", hue="player")
        
        # Add vertical lines per game ending
        game_endings = df_plot[df_plot["done"]].index
        for i in game_endings:
            ax.axvline(i, color="gray", linestyle="--", alpha=0.5)
            
        
        ax.set_title("Cumulative Reward Over Time by Player")
        ax.set_xlabel("Step (event order)")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(title="Players")
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        
        self.fig_cumulative_reward_by_player = fig
        
    
    def plot_reward_by_game_grid(self) -> None:
        """
        Plots the cumulative reward over steps for each game in the log,
        placing each game's plot on a grid. Each subplot shows per-player 
        lines for that game.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        # Make a copy to avoid modifying the main DataFrame
        df_plot = self.df.copy()
        # Sort by index (or by a timestamp column if you have one)
        df_plot.sort_index(inplace=True)
        
        # Get the unique game numbers in ascending order
        unique_games = sorted(df_plot["game_number"].unique())
        n_games = len(unique_games)
        
        # Decide how many rows/columns for subplots (e.g., up to 3 columns wide)
        n_cols = 3
        n_rows = math.ceil(n_games / n_cols)
        
        # Create the figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5 * n_cols, 4 * n_rows), 
                                sharex=False, sharey=False)
        # If there's only 1 row or 1 column, `axes` might not be a 2D array
        # Flatten for easier iteration
        axes = np.array(axes).reshape(-1)
        
        for i, game_number in enumerate(unique_games):
            # Filter the dataframe to the current game
            df_game = df_plot[df_plot["game_number"] == game_number].copy()
            
            # Get the winner of the game
            # Either by looking where the 'player_result' is 'won'
            # Or looking where the 'player_result' is 'lost' and taking the other player
            if "won" in df_game["player_result"].values:
                winner = df_game[df_game["player_result"] == "won"]["player"].values[0]
                loser = df_game[df_game["player"] != winner]["player"].values[0]
            else:
                loser = df_game[df_game["player_result"] == "lost"]["player"].values[0]
                winner = df_game[df_game["player"] != loser]["player"].values[0]
            
            # Create a step for event order within just this game
            df_game["step"] = range(len(df_game))
            # Compute cumulative rewards per player within this game
            df_game["cumulative_reward"] = (
                df_game.groupby("player")["reward"]
                    .cumsum()
            )
            
            ax = axes[i]
            # Plot a line per player in this game
            # Ensure the same player always has the same color
            hue_order = ["Player 1", "Player 2"]
            sns.lineplot(
                ax=ax,
                data=df_game,
                x="step",
                y="cumulative_reward",
                hue="player",
                hue_order=hue_order,
                palette=["#1f77b4", "#ff7f0e"],
                linestyle="-"
            )
            ax.set_title(f"Game {game_number} (Winner: {winner}, Loser: {loser})")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative Reward")
            ax.legend(title="Player")
            
        # Hide any unused subplots if there are empty slots in the grid
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        fig.tight_layout()
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)
        
        self.fig_reward_by_game_grid = fig

    
    def plot_reward_by_game_by_player(self) -> None:
        """
        Plots the sum of rewards per game number and player in double bar chart format.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        # Get rewards per game per player
        player1_rewards = self.df[self.df["player"] == "Player 1"].groupby("game_number")["reward"].sum()
        player2_rewards = self.df[self.df["player"] == "Player 2"].groupby("game_number")["reward"].sum()
        
        # Create a DataFrame for plotting
        df_plot = pd.DataFrame({
            "Game": player1_rewards.index,
            "Player 1": player1_rewards.values,
            "Player 2": player2_rewards.values
        })
        
        # Create fig with two bars per game for each player
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.4
        bar_positions = np.arange(len(df_plot))
        
        ax.bar(bar_positions, df_plot["Player 1"], width=bar_width, label="Player 1", color="#1f77b4")
        ax.bar(bar_positions + bar_width, df_plot["Player 2"], width=bar_width, label="Player 2", color="#ff7f0e")
        
        ax.set_xlabel("Game Number")
        ax.set_ylabel("Total Reward")
        ax.set_title("Total Reward by Game and Player")
        ax.set_xticks(bar_positions + bar_width / 2)
        ax.set_xticklabels(df_plot["Game"])
        ax.legend()
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)
        
        self.fig_reward_by_game_by_player = fig

    
    def plot_action_distribution(self) -> None:
        """
        Plots the distribution of action types taken across all events.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(ax=ax, x="action_type", data=self.df, palette="viridis", hue="action_type", legend=False)
        ax.set_title("Distribution of Action Types")
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)
        
        self.fig_action_distribution = fig

    
    def plot_class_usage(self, by_player: bool = False):
        """
        Plots the usage frequency of different classes in the log.
        If by_player=True, it creates a grouped bar chart for each player-class combination.

        :param by_player: Whether to split usage frequency by each player or combine.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count unique class occurrences per player per game
        if by_player:
            df_class_usage = self.df.groupby(["player", "card_class"])["game_number"].nunique().reset_index()
            sns.barplot(ax=ax, x="card_class", y="game_number", data=df_class_usage, palette="viridis", hue="player", legend=False)
            ax.set_title("Class Usage Frequency by Player")
            ax.set_xlabel("Card Class")
            ax.set_ylabel("Count")
        else:
            df_class_usage = self.df.groupby(["card_class"])["game_number"].nunique().reset_index()
            sns.barplot(ax=ax, x="card_class", y="game_number", data=df_class_usage, palette="viridis", hue="card_class", legend=False)
            ax.set_title("Class Usage Frequency")
            ax.set_xlabel("Card Class")
            ax.set_ylabel("Count")

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        self.fig_class_usage = fig

    
    def plot_agent_type_performance(self) -> None:
        """
        Groups the data by 'agent_type' and plots the average reward per agent type.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return

        agent_rewards = self.df.groupby("agent_type")["reward"].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(ax=ax, x="agent_type", y="reward", data=agent_rewards, palette="viridis", hue="agent_type", legend=False)
        ax.set_title("Average Reward by Agent Type")
        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Average Reward")
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        
        self.fig_agent_type_performance = fig

    
    def plot_action_string_distribution(self) -> None:
        """
        Plots the average distribution of action strings across all games per player per hero_class.
        Possible actions:
        0: Play card - Card: 3 Power of the Wild (2 Mana) 'Choose One - Give your minions +1/+1; or Summon a 3/2 Panther.' - Choose Option: Summon a Panther (2 Mana) 'Summon a 3/2 Panther.
        0: Attack with minion - Attacker: 0 Panther (Health: 3/3) - Target: 1 Houndmaster (Health: 3/3)
        0: Use hero power (Lightning Jolt 'Hero Power Deal 2 damage.') - Target: 5 Hooked Reaver (Health: 7/7)
        :return: A dictionary with the distribution of action strings.
        """
        
        # Goal is to parse action strings into types (Play card, Attack with minion, Use hero power, etc.) and count them
        # Format is setup like for name, value in {"Card": card, "Discover Option": discover, "Choose Option": choose, "Attacker": attacker, "Target": target}.items():
        # if value:
        #     print_string += f" - {name}: {value}"
        
        
        # Add a column to the DataFrame with a cleaned-up version of the action string
        self.df["action_string_stripped"] = self.df["action_string"].apply(lambda x: self._clean_action_string(x) if isinstance(x, str) else None)
        
        # Count the frequency of each action string per player per game
        # Calculate the average distribution across all games per player per hero_class
        distributions = {}
        for player in self.df["player"].unique():
            # Get the unique hero classes for this player
            hero_classes = self.df[self.df["player"] == player]["card_class"].unique()
            distributions[player] = {}
            for hero_class in hero_classes:
                decks = self.df[(self.df["player"] == player) & (self.df["card_class"] == hero_class)]["deck"].unique()
                distributions[player][hero_class] = {}
                for deck in decks:
                    # Filter the DataFrame to the current player and hero class
                    df_player_class = self.df[(self.df["player"] == player) & (self.df["card_class"] == hero_class) & (self.df["deck"] == deck)]
                    # Count the frequency of each action string
                    action_string_counts = df_player_class["action_string_stripped"].value_counts(normalize=True)
                    # Remove the End turn actions since they are not as useful for this analysis
                    action_string_counts = action_string_counts[~action_string_counts.index.str.contains("End turn")]                
                    distributions[player][hero_class][deck] = action_string_counts
                
        # Plot the distribution of top k action strings per player per hero_class
        top_k = 20
        
        for player, player_data in distributions.items():
            for hero_class, decks in player_data.items():
                for deck, action_string_counts in decks.items():
                    # Get the top k action strings
                    top_k_action_strings = action_string_counts.head(top_k)
                    
                    fig, ax = plt.subplots(figsize=(20, 15))
                    sns.barplot(ax=ax, x=top_k_action_strings.values, y=top_k_action_strings.index, palette="viridis", hue = top_k_action_strings.values, dodge=False)
                    ax.set_title(f"Top {top_k} Action Strings for {player} - {hero_class} - {deck}")
                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Action String")
                    
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(30)
                    
                    self.fig_action_string_distributions.append(fig)

    
    
    def plot_combo_distribution(self) -> None:
        """
        Plots the distribution of combo strings across all games per player per hero_class.
        Goal is to check if action strings are happening during the same sequence (incidated by player switch in between) and count them.
        """
        
        # First clean up the action strings
        self.df["action_string_stripped"] = self.df["action_string"].apply(lambda x: self._clean_action_string(x) if isinstance(x, str) else None)
        
        df = self.df.copy()
        
        # Filter out the end turn actions DISABLED DUE TO COMBO INTERFERENCE
        # df = df[~df["action_string_stripped"].str.contains("End turn")]
        
        # Create a flag that indicates a change in game or player
        df['new_combo'] = ((df['game_number'] != df['game_number'].shift(1)) | 
                            (df['player'] != df['player'].shift(1))).astype(int)
        
        # Cumulatively sum the flag to assign a unique combo id
        df['combo_id'] = df['new_combo'].cumsum()
        
        # Compute the length of each combo
        seq_lengths = df.groupby('combo_id').size().reset_index(name='combo_length')
        # Also get game and player info (we use the first row of each combo)
        seq_info = df.groupby('combo_id').first().reset_index()[['combo_id', 'game_number', 'player']]
        seq_df = pd.merge(seq_lengths, seq_info, on='combo_id')
        
        # group by game_number, combo_id, plus also player/card_class
        # Because you might want to attach which player is in that combo.

        combo_stats = (
            df
            .groupby(["game_number", "combo_id", "player", "card_class", "deck"], as_index=False)
            .size()    # .size() = number of rows in that group => length of the combo
            .rename(columns={"size": "combo_length"})
        )
        
        # how many combos did each player do per game_number, card_class?
        combo_counts = (
            combo_stats
            .groupby(["game_number", "player", "card_class", "deck"], as_index=False)
            .size()
            .rename(columns={"size": "num_combos_in_that_game"})
        )
        
        # combo_stats has the length of *each* combo. 
        # We can directly plot the distribution, facetting by player and/or card_class.
        # Ensure only integer values are used for the combo_length in the plots
        combo_stats["combo_length"] = combo_stats["combo_length"].astype(int)
        
        # Hist plot should have bin per integer value
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            ax=ax,
            data=combo_stats,
            x="combo_length",
            hue="player",    # or facet by card_class, depends on your preference
            multiple="dodge",
            binwidth=1,                # make each bin 1 wide
            binrange=(0, combo_stats["combo_length"].max() + 1),  # from 0 up to max
            alpha=0.5,
            discrete=True
        )
 
        ax.set_xlabel("Number of Actions")
        
        ax.set_title("Counts of number of actions taken within one turn by player")
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        
        # Label each bar with the category
        for container in ax.containers:
            labels = []
            for bar in container:
                # The x position of the bar's center
                x_center = bar.get_x() + bar.get_width() / 2
                
                # Convert the x position to an integer label (rounded)
                bin_label = str(int(round(x_center)))
                
                labels.append(bin_label)
            
            # Place the labels near the top edge of each bar
            ax.bar_label(
                container,
                labels=labels,
                label_type='edge',   # put the label above the bar
                rotation=0,          # rotate if you prefer
                padding=3,           # extra space above the bar
                fontsize=6,          # smaller font size
            )
        
        self.fig_combo_distribution_lengths = fig

        # Facet by card_class
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure that only 3 columns are shown per row
        g = sns.FacetGrid(combo_stats, col="card_class", hue="player", sharey=False, col_wrap=3)
        g.map(
            sns.histplot, 
            "combo_length", 
            alpha=0.5,
            discrete=True,
            binwidth=1,
            binrange=(0, combo_stats["combo_length"].max() + 1),
            multiple="dodge"
            )
        
        # Label each bar with the category
        for ax in g.axes.flat:
            for container in ax.containers:
                labels = []
                for bar in container:
                    # The x position of the bar's center
                    x_center = bar.get_x() + bar.get_width() / 2

                    # Convert the x position to an integer label (rounded)
                    bin_label = str(int(round(x_center)))

                    labels.append(bin_label)

                # Place the labels near the top edge of each bar
                ax.bar_label(
                    container,
                    labels=labels,
                    label_type='edge',   # put the label above the bar
                    rotation=0,          # rotate if you prefer
                    padding=3,           # extra space above the bar
                    fontsize=6,          # smaller font size
                )
        
        g.add_legend()
        ax.set_title("Counts of number of actions taken within one turn by player and card class")
        
        # Set the x axis label for the last row of the grid
        for ax in g.axes.flat:
            ax.set_xlabel("Number of Actions")
            
        
        self.fig_combo_distribution_grid = g
        
        # Create a dataframe that shows the top 5 longest combos per player per card_class
        top_combos = combo_stats.groupby(["player", "card_class", "deck"]).apply(lambda x: x.nlargest(5, "combo_length")).reset_index(drop=True)
        
        # store the top 5 longest combos per player per card_class
        for idx, row in top_combos.iterrows():
            game_number = row["game_number"]
            combo_id = row["combo_id"]
            player = row["player"]
            card_class = row["card_class"]
            deck = row["deck"]
            combo_length = row["combo_length"]
            action_string = list(df[(df["game_number"] == game_number) & (df["combo_id"] == combo_id)]["action_string"].values)
            
            # Add to the top_combos dictionary
            if player not in self.top_combos:
                self.top_combos[player] = {}
            if card_class not in self.top_combos[player]:
                self.top_combos[player][card_class] = {}
            if deck not in self.top_combos[player][card_class]:
                self.top_combos[player][card_class][deck] = []
                
            self.top_combos[player][card_class][deck].append({
                "game_number": game_number,
                "combo_id": combo_id,
                "combo_length": combo_length,
                "action_string": action_string
            })
           
     
    def plot_combo_network(self, width_threshold: int = 1):
        print("Generating combo network plot...")
        
        self.combo_networks = []
        
        empty_network_expander = st.expander("Empty Networks", expanded=False)
        
        # Create a folder to store the combo network plots
        graph_folder = os.path.join("analysis", "combo_networks")
        
        # Check if subfolder with experiment name exists, if not create it
        if not os.path.exists(os.path.join(graph_folder, f"{self.experiment_name}")):
            os.makedirs(os.path.join(graph_folder, f"{self.experiment_name}"))
        
        self.df["action_string_stripped"] = self.df["action_string"].apply(lambda x: self._clean_action_string(x) if isinstance(x, str) else None)
        
        action_type_color_map = {
            0: "blue",
            1: "green",
            2: "purple",
            3: "red",
            4: "orange",
        }
        
        df = self.df.copy()
        
        # Filter out the end turn actions
        df = df[~df["action_string_stripped"].str.contains("End turn")]
        
        unique_combos = df[["player","card_class","deck"]].drop_duplicates()

        for idx, row in unique_combos.iterrows():
            ply = row["player"]
            cls = row["card_class"]
            deck = row["deck"]

            sub_df = df[(df["player"] == ply) & (df["card_class"] == cls) & (df["deck"] == deck)]
            if sub_df.empty:
                continue
            
            sub_df = sub_df.sort_values(["game_number", "turn"]).reset_index(drop=True)

            # Within each game_number, let's find the *next* action_string
            sub_df["next_action"] = (
                sub_df
                .groupby("game_number")["action_string_stripped"]
                .shift(-1)   # row i gets row i+1's action_string
            )
            
            sub_df = sub_df.sort_values(["game_number", "turn"]).reset_index(drop=True)

            # Within each game_number, let's find the *next* action_string
            sub_df["next_action"] = (
                sub_df
                .groupby("game_number")["action_string_stripped"]
                .shift(-1)   # row i gets row i+1's action_string
            )
            
            pairs_df = (
                sub_df
                .dropna(subset=["next_action"])   # discard if there's no next action
                .groupby(["action_string_stripped","next_action"], as_index=False)
                .size()
                .rename(columns={"size":"weight"})
            )

            # Build a dictionary from action_string -> action_type
            # (If the same action_string appears with multiple action_types, 
            # you might need a more robust approach, e.g. the 'most common' type.)
            string_type_map = (
                sub_df[["action_string_stripped","action_type"]]
                .drop_duplicates(subset=["action_string_stripped"])
                .set_index("action_string_stripped")["action_type"]
                .to_dict()
            )
            
            # Take the average weight of the pairs
            pairs_df["weight"] = ((pairs_df["weight"] / pairs_df["weight"].count()) * 100).round(0)

            import networkx as nx

            G = nx.DiGraph()

            # Add edges from pairs_df
            for _, row in pairs_df.iterrows():
                a1 = row["action_string_stripped"]
                a2 = row["next_action"]
                w  = row["weight"]
                if a1 == a2 or w < width_threshold:
                    continue
                G.add_edge(a1, a2, weight=w)
                
            # Check how many edges we have, if none, print heighest 5 weights with counts
            if len(G.edges) == 0:
                empty_network_expander.warning(f"No edges found for {ply} - {cls} - {deck}. Top 5 weights:")
                empty_network_expander.write(pairs_df.nlargest(5, "weight"))
                continue

            # Add node attribute: action_type
            for n in G.nodes():
                # If not found in the dict, fallback to something
                G.nodes[n]["action_type"] = string_type_map.get(n, np.nan)
            
            # Initiate PyVis network object
            network = Network(
                            height=f'1000px',
                            width='100%',
                            bgcolor='white',
                            directed=False
                            )

            # Take Networkx graph and translate it to a PyVis graph format
            network.from_nx(G)
                    
            # Color nodes based on type
            for node in network.nodes: 
                action_type = node["action_type"]
                
                node["color"] = action_type_color_map.get(action_type, "white")
                node["title"] = f"Name: {node["id"]}"
                # Make font bold and bigger
                node["font"] = {"size": 20, "color": "black", "strokeWidth": 1, "strokeColor": "black"}
                

            # Adjust network with specific layout settings
            network.repulsion(
                node_distance=150,
                central_gravity=0.33,
                spring_length=110,
                spring_strength=0.10,
                damping=0.95
            )                
            
            # Add hover functionality
            for edge in network.edges:
                edge["color"] = "lightblue"
                edge["title"] = f"Avg count: {edge['width']}"
            
            # Sub \ and / are not allowed in file names, so replace them with _
            ply = ply.replace("/", "_")
            cls = cls.replace("/", "_")
            deck = deck.replace("/", "_")
            
            file_path = os.path.join(graph_folder, f"{self.experiment_name}", f"{ply}_{cls}_{deck}_combo_network.html")
            
            network.save_graph(file_path)
            
            self.combo_networks.append(file_path)
        print("Combo network plots saved in 'analysis/combo_networks' folder.")

    def _clean_action_string(self, action_string: str):
        """
        Cleans up an action string by removing unnecessary parts.
        """
        # Clean the 0: from the start of the action string
        action_stripped = re.sub(r"^\d+: ", "", action_string)
            
        # Strip the (Attack: x - Health: n/m) part from the Attacker and Target using regex for Minions
        action_stripped = re.sub(r"\(Attack: \d+ - Health: \d+/\d+\)", "", action_stripped)

        
        # If the attacker is the hero, remove the (Attack: n, Health: m, Armor: o) part
        action_stripped = re.sub(r"\(Attack: \d+, Health: \d+, Armor: \d+\)", "", action_stripped)
        
        # Strip the Target and whatever is targeted from the action string
        action_stripped = re.sub(r" - Target: \d+ .+", "", action_stripped)
        
        # Remove index after Card, Attacker, Discover Option, Choose Option since it's not as useful for this analysis
        # Keep Card, Attacker, Discover Option, Choose Option
        action_stripped = re.sub(r" - (Card|Attacker|Discover Option|Choose Option): \d+ ", r" - \1 ", action_stripped)
        
        # Replace any Jade_Golem with Jade Golem
        action_stripped = re.sub(r"Jade_Golem", r"Jade Golem", action_stripped)
        
        # If any digit/digit Jade Golem is mentioned, replace it with x/x Jade Golem
        action_stripped = re.sub(r"(\d+)/(\d+) Jade Golem", r"x/x Jade Golem", action_stripped)
        
        # Replace any "an x/x" with "a x/x" for better readability
        action_stripped = re.sub(r"an x/x", r"a x/x", action_stripped)
        
        return action_stripped