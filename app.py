import streamlit as st

from ui.home import render_home
from ui.data_module import render_data_module
from ui.modelling_module import render_modelling_module
from ui.control_module import render_control_module
from ui.copilot import render_copilot_sidebar
from ui.side_bar_data_selection import render_sidebar_data_selection

st.set_page_config(
    page_title="Battery R&D Platform",
    layout="wide",
)

# 默认页面
if "page" not in st.session_state:
    st.session_state.page = "home"


# 👇 Home按钮（核心）
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "home"
    st.rerun()

st.sidebar.divider()

# 👇 Copilot
render_copilot_sidebar(st.session_state.page)
global_data_config = render_sidebar_data_selection()

# 👇 页面路由
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "data":
    render_data_module()
elif st.session_state.page == "modelling":
    render_modelling_module()
elif st.session_state.page == "control":
    render_control_module()