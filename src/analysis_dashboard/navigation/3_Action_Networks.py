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
    progress_bar.progress(0.1, text="Getting Summary Statistics...")
    st.session_state.analyzer.get_summary_statistics()
    
if len(st.session_state.analyzer.combo_networks) == 0:
    progress_bar.progress(0.6, text="Creating Action Networks...")
    st.session_state.analyzer.plot_combo_network(width_threshold=st.session_state.width_threshold)
    
progress_bar.progress(1.0, text="Done!")
progress_bar.empty()

st.sidebar.divider()
st.sidebar.header("Log Info")
st.sidebar.header("Player Information")
st.sidebar.dataframe(st.session_state.analyzer.player_info, hide_index=True)
    
st.sidebar.subheader(f"Unique Games: {st.session_state.analyzer.unique_games}")
st.sidebar.subheader(f"Events Logged: {st.session_state.analyzer.events_logged}")

st.header("Action Networks")
    
width_threshold = st.slider("Width threshold", min_value=1, max_value=10, value=1)

if st.session_state.width_threshold != width_threshold:
    if st.button("Reload Action Networks"):
        st.session_state.width_threshold = width_threshold
        st.session_state.analyzer.plot_combo_network(width_threshold=width_threshold)


if len(st.session_state.analyzer.combo_networks) > 0:
    
    # Strip analysis\combo_networks\ prefix and experiment name suffix
    options = [network.replace("analysis\\combo_networks\\", "").replace(f"{st.session_state.analyzer.experiment_name}\\", "").replace(".html", "").replace("_combo_network", "") for network in st.session_state.analyzer.combo_networks if type(network) == str]
    
    selected_network = st.selectbox("Select a network", options)
    
    file_path = os.path.join("analysis/combo_networks", st.session_state.analyzer.experiment_name, f"{selected_network}_combo_network.html")
    
    HtmlFile = open(file_path, 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=1600)
else:
    st.info("No action networks available for this experiment")