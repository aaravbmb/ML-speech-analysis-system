import streamlit as st
from streamlit_option_menu import option_menu

def sidebar_ui():
    with st.sidebar:
        # Logo
        st.image("https://raw.githubusercontent.com/your-username/your-repo/main/logo.png", width=130)

        # App name
        st.markdown(
            "<h3 style='text-align: center; color: white;'>MoodMatrix: Speech<br>Analysis System</h3>",
            unsafe_allow_html=True
        )

        # Navigation
        selected = option_menu(
            menu_title=None,
            options=["Analyze", "Project Details", "About Us"],
            icons=["mic", "book", "info-circle"],
            menu_icon=None,
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#443C56"},
                "icon": {"color": "white", "font-size": "16px"},
                "
