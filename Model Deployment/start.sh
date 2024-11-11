#!/bin/bash

# Grant execute permissions for the startup script (needed for Render)
chmod +x start.sh

# Start the Streamlit app
streamlit run main.py
