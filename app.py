import streamlit as st
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(page_title="MoodMatrix", layout="wide")

# Sidebar UI
def sidebar_ui():
    with st.sidebar:
        # Logo from GitHub (replace with your actual username/repo)
        st.image("https://raw.githubusercontent.com/your-username/your-repo/main/logo.png", width=130)

        # App name
        st.markdown(
            "<h3 style='text-align: center; color: white;'>MoodMatrix:<br>Speech Analysis System</h3>",
            unsafe_allow_html=True
        )

        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Analyze", "Project Details", "About Us"],
            icons=["mic", "book", "info-circle"],
            menu_icon=None,
            default_index=0,
            orientation="vertical",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#443C56"
                },
                "icon": {
                    "color": "white",
                    "font-size": "16px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "10px 0",
                    "color": "white",
                    "border-radius": "12px"
                },
                "nav-link-selected": {
                    "background-color": "#5A526C",
                    "font-weight": "bold",
                    "color": "white"
                },
            }
        )

        # Spacer to push icons to bottom
        st.markdown("<div style='height:150px;'></div>", unsafe_allow_html=True)

        # Footer icons
        st.markdown(
            """
            <div style='display: flex; justify-content: center; gap: 30px;'>
                <a href='https://github.com/your-username/your-repo' target='_blank'>
                    <img src='https://img.icons8.com/ios-filled/50/ffffff/github.png' width='30'/>
                </a>
                <a href='https://your-external-link.com' target='_blank'>
                    <img src='https://img.icons8.com/ios-filled/50/ffffff/external-link.png' width='30'/>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

        return selected

# Main UI Logic
def main():
    page = sidebar_ui()

    if page == "Analyze":
        st.title("🔍 Analyze Speech Emotion")
        st.write("Upload an audio file or record your voice for emotion analysis.")
        # Add your recording/upload logic here

    elif page == "Project Details":
        st.title("📘 Project Details")
        st.write("This section contains information about the MoodMatrix project.")
        # Add project details content

    elif page == "About Us":
        st.title("👤 About Us")
        st.write("Meet the team behind MoodMatrix.")
        # Add team or author info

if __name__ == "__main__":
    main()
