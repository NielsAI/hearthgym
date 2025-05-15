# Load default libraries
import streamlit as st

from functions.GameAnalyzer import HearthstoneLogAnalyzer

# Load configuration
import yaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np
import streamlit.components.v1 as components

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
    
if st.session_state.analyzer.fig_combo_distribution_lengths is None or st.session_state.analyzer.fig_combo_distribution_grid is None or st.session_state.analyzer.top_combos is None:
    progress_bar.progress(0.5, text="Plotting Action Counts...")
    st.session_state.analyzer.plot_combo_distribution()
    
    
progress_bar.progress(1.0, text="Done!")
progress_bar.empty()

st.sidebar.divider()
st.sidebar.header("Log Info")
st.sidebar.header("Player Information")
st.sidebar.dataframe(st.session_state.analyzer.player_info, hide_index=True)
    
st.sidebar.subheader(f"Unique Games: {st.session_state.analyzer.unique_games}")
st.sidebar.subheader(f"Events Logged: {st.session_state.analyzer.events_logged}")

st.header("Action Sequences")
    
col1, col2, col3 = st.columns(3)
player_selected = col1.selectbox("Select a player", list(st.session_state.analyzer.top_combos.keys()))

class_selected = col2.selectbox("Select a class", list(st.session_state.analyzer.top_combos[player_selected].keys()))

deck_selected = col3.selectbox("Select a deck", list(st.session_state.analyzer.top_combos[player_selected][class_selected].keys()))

st.header("Top Action Sequences:")
combos = st.session_state.analyzer.top_combos[player_selected][class_selected][deck_selected]
for combo in combos:
    with st.expander(f"Game Number: {combo['game_number']}, Action sequence length: {combo['combo_length']}", expanded=False):
        for action in combo["action_string"]:
            st.info(action)