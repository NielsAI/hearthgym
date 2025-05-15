# Load default libraries
import streamlit as st

from functions.TensorBoardAnalyzer import TensboardLogAnalyzer

# Load configuration
import yaml
from yaml.loader import SafeLoader

import os

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
    
if "tensorboard_analyzer" not in st.session_state:
    st.session_state.tensorboard_analyzer = TensboardLogAnalyzer()
    
if "tensorboard_logs" not in st.session_state:
    st.session_state.tensorboard_logs = {}

tab1, tab2 = st.tabs(["Model Selection", "Tensorboard Plot"])

with tab1:
    # Get all possible log files from "logs" folder
    log_folders = os.listdir(st.session_state.config["TENSORBOARD_FOLDER"])

    number_logs = st.sidebar.slider("Number of logs", min_value=1, max_value=25, value=1)
    step_limit = st.slider("Step limit", min_value=1e6, max_value=5e6, value=2e6, step=5e5)

    # If number of logs is smaller than the previous number of logs, remove the last logs
    if number_logs < len(st.session_state.tensorboard_logs):
        for i in range(number_logs, len(st.session_state.tensorboard_logs)):
            last_key = list(st.session_state.tensorboard_logs.keys())[-1]
            st.session_state.tensorboard_logs.pop(last_key)
        st.rerun()

    col1, col2, col3, col4 = st.columns(4)

    model_expanders = []
    log_inputs = {}
    for i in range(number_logs):
        # Get correct column (First 4 models are in first row, next 4 in second row)
        if i % 4 == 0:
            col = col1
        elif i % 4 == 1:
            col = col2
        elif i % 4 == 2:
            col = col3
        else:
            col = col4
        
        model_expanders.append(col.expander(f"Model {i+1}", expanded=True))
        
        log_inputs[f"model_{i}_name"] = model_expanders[i].text_input(f"Model {i+1} Name", f"Model {i+1} Name", key=f"model_{i}_name")
        log_inputs[f"model_{i}_folder"] = model_expanders[i].selectbox(f"Model {i+1} folder", log_folders, key=f"model_{i}_folder")
        
        # Get the available files in the selected folder
        log_files = os.listdir(os.path.join(st.session_state.config["TENSORBOARD_FOLDER"], log_inputs[f"model_{i}_folder"]))
        
        log_inputs[f"model_{i}_file"] = model_expanders[i].selectbox(f"Model {i+1} file", log_files, key=f"model_{i}_file")
        
    if st.sidebar.button("Load logs"):
        with st.sidebar:
            with st.spinner("Loading logs..."):
                for i in range(number_logs):
                    # Get the actual event file
                    event_file = os.listdir(os.path.join(st.session_state.config["TENSORBOARD_FOLDER"], log_inputs[f"model_{i}_folder"], log_inputs[f"model_{i}_file"]))[0]
                    file_path = os.path.join(st.session_state.config["TENSORBOARD_FOLDER"], log_inputs[f"model_{i}_folder"], log_inputs[f"model_{i}_file"], event_file)
                    st.session_state.tensorboard_logs[f"model_{i}"] = [log_inputs[f"model_{i}_name"], file_path]
                
                st.session_state.tensorboard_analyzer.load_tensorboard_logs(st.session_state.tensorboard_logs)
                st.rerun()
            
    if st.session_state.tensorboard_analyzer.tensorboard_logs == {}:
        st.info("Please select the logs to load")
        st.stop()
    
with tab2:
    smooth_value = st.slider("Smooth value", min_value=0.0, max_value=0.99, value=0.95, step=0.01)
        
    with st.spinner("Loading tensorboard plot..."):
        tensorboard_fig = st.session_state.tensorboard_analyzer.plot_rewards(smooth_value=smooth_value)
        
    st.pyplot(tensorboard_fig)
    