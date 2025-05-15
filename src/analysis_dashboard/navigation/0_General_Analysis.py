# Load default libraries
import streamlit as st

from functions.GameAnalyzer import HearthstoneLogAnalyzer

# Load configuration
import yaml
from yaml.loader import SafeLoader

try:
    if "config" not in st.session_state:
        with open('config.yaml') as config_file:
            st.session_state.config = yaml.load(config_file, Loader=SafeLoader)

except Exception as e:
    st.error(st.session_state.config["MESSAGES"]["ERRORS"]["START"])
    st.error(e)
    st.stop()
    
if "loaded_experiment_name" not in st.session_state:
    st.session_state.loaded_experiment_name = None

import os
from functions.GameAnalyzer import HearthstoneLogAnalyzer

st.sidebar.title("Select an experiment")

# Get all possible log folders from "logs" folder
log_files = os.listdir(st.session_state.config["LOGS_FOLDER"])

# Strip out any non-folder files
log_files = [log_file for log_file in log_files if os.path.isdir(os.path.join(st.session_state.config["LOGS_FOLDER"], log_file))]

selected_main_folder = st.sidebar.selectbox(
    "Select Main Folder", 
    log_files, 
    index=log_files.index(st.session_state.config["DEFAULT_MAIN_FOLDER"]),
    key="main_folder"
    )

# Get all log files from the selected main folder
log_files = os.listdir(os.path.join(st.session_state.config["LOGS_FOLDER"], selected_main_folder))

selected_experiment_folder = st.sidebar.selectbox("Select Experiment Folder", log_files, key="experiment_folder")

# Get all log files from the selected experiment folder
log_files = os.listdir(os.path.join(st.session_state.config["LOGS_FOLDER"], selected_main_folder, selected_experiment_folder))

# Strip games_ prefix and .log suffix
log_files = [log_file.replace("games_", "").replace(".log", "") for log_file in log_files]

experiment_name = st.sidebar.selectbox("Select Experiment", log_files)

filename = f"games_{experiment_name}.log"

log_file = os.path.join(st.session_state.config["LOGS_FOLDER"], filename)

btn_placeholder = st.sidebar.empty()
if st.session_state.loaded_experiment_name != experiment_name:
    st.sidebar.info("Please click the button to load the log file")

if btn_placeholder.button("Load log"):
    log_path = os.path.join(st.session_state.config["LOGS_FOLDER"], selected_main_folder, selected_experiment_folder, filename)
    st.session_state.loaded_log_file = log_path
    st.session_state.loaded_experiment_name = experiment_name
    
if "loaded_log_file" not in st.session_state:
    st.info("Please select an experiment to load the log file")
    st.stop()
    
if "width_threshold" not in st.session_state:
    st.session_state.width_threshold = 1
    
st.title(f"Analysis for experiment {st.session_state.loaded_experiment_name}")

try:
    if "analyzer" not in st.session_state:
        # Create the analyzer instance
        st.session_state.analyzer = HearthstoneLogAnalyzer(
            file_path = st.session_state.loaded_log_file, 
            experiment_name = st.session_state.loaded_experiment_name
            )
    elif st.session_state.analyzer.experiment_name != st.session_state.loaded_experiment_name:
        st.session_state.analyzer = HearthstoneLogAnalyzer(
            file_path = st.session_state.loaded_log_file, 
            experiment_name = st.session_state.loaded_experiment_name
            )
except Exception as e:
    st.error(st.session_state.config["MESSAGES"]["ERRORS"]["ANALYZER"])
    st.error(e)
    st.stop()
    
# Function to highlight cells that are not "0/0"
def highlight_not_0_0(val):
    return 'background-color: #A7C7E7' if val != '0/0' else ''

progress_bar = st.progress(0, text="Loading data...")

# Run all analyses depending on the state of the analyzer object
if st.session_state.analyzer.player_info is None or st.session_state.analyzer.unique_games is None or st.session_state.analyzer.events_logged is None:
    progress_bar.progress(0.1, text="Getting summary statistics...")
    st.session_state.analyzer.get_summary_statistics()

if st.session_state.analyzer.winrate_df is None or st.session_state.analyzer.win_matrix is None or st.session_state.analyzer.win_matrix_heatmap_fig_p1 is None or st.session_state.analyzer.win_matrix_heatmap_fig_p2 is None:
    progress_bar.progress(0.2, text="Getting winrates...")
    st.session_state.analyzer.get_winrates(folder_path="analysis/wins_analysis/", experiment_name=experiment_name)

if st.session_state.analyzer.fig_cumulative_reward_by_player is None:
    progress_bar.progress(0.3, text="Plotting cumulative rewards...")
    st.session_state.analyzer.plot_cumulative_reward_by_player()
    
if st.session_state.analyzer.fig_action_distribution is None:
    progress_bar.progress(0.4, text="Plotting action distribution...")
    st.session_state.analyzer.plot_action_distribution()
    
if st.session_state.analyzer.fig_class_usage is None:
    progress_bar.progress(0.8, text="Plotting class usage...")
    st.session_state.analyzer.plot_class_usage(by_player=True)
    
if st.session_state.analyzer.fig_agent_type_performance is None:
    progress_bar.progress(0.9, text="Plotting agent type performance...")
    st.session_state.analyzer.plot_agent_type_performance()
    
progress_bar.progress(1.0, text="Done!")
progress_bar.empty()

st.sidebar.divider()
st.sidebar.header("Log Info")
st.sidebar.header("Player Information")
st.sidebar.dataframe(st.session_state.analyzer.player_info, hide_index=True)
    
st.sidebar.subheader(f"Unique Games: {st.session_state.analyzer.unique_games}")
st.sidebar.subheader(f"Events Logged: {st.session_state.analyzer.events_logged}")


st.header("General Analysis")
    
# Display the figures and dataframes
with st.expander("Winrate Dataframes", expanded=True):
    col1, col2 = st.columns(2)
    col1.subheader("Winrates Player 1")
    col2.subheader("Winrates Player 2")
    col1.dataframe(st.session_state.analyzer.winrate_df[st.session_state.analyzer.winrate_df["player"] == "Player 1"])
    col2.dataframe(st.session_state.analyzer.winrate_df[st.session_state.analyzer.winrate_df["player"] == "Player 2"])
    
    st.subheader("Win Matrix")
    st.dataframe(st.session_state.analyzer.win_matrix.style.map(highlight_not_0_0), height=700)
    
    st.subheader("Winrate Heatmap")
    player_selected = st.radio("Select a player", ["Player 1", "Player 2"])
    if player_selected == "Player 1":
        st.pyplot(st.session_state.analyzer.win_matrix_heatmap_fig_p1)
    else:
        st.pyplot(st.session_state.analyzer.win_matrix_heatmap_fig_p2)
    
with st.expander("Cumulative Reward by Player", expanded=False):
    st.pyplot(st.session_state.analyzer.fig_cumulative_reward_by_player)

with st.expander("Action Strings", expanded=False):
    for fig in st.session_state.analyzer.fig_action_string_distributions:
        st.pyplot(fig)
    
with st.expander("Action Distribution", expanded=False):
    st.pyplot(st.session_state.analyzer.fig_action_distribution)
    
with st.expander("Class Usage and Agent Type Performance", expanded=False):
    st.pyplot(st.session_state.analyzer.fig_class_usage)
    st.pyplot(st.session_state.analyzer.fig_agent_type_performance)

    






