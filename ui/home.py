import streamlit as st


CARD_STYLE = """
<style>
.home-title {
    font-size: 42px;
    font-weight: 750;
    color: #0F172A;
    margin-bottom: 6px;
}

.home-subtitle {
    color: #64748B;
    font-size: 16px;
    margin-bottom: 34px;
}

.module-card {
    height: 220px;
    padding: 26px;
    border-radius: 22px;
    border: 1px solid #E2E8F0;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 14px;
    transition: all 0.25s ease;
}

.module-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.13);
    border-color: #CBD5E1;
}

.module-icon {
    font-size: 34px;
    margin-bottom: 20px;
}

.module-title {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
}

.module-subtitle {
    font-size: 14px;
    line-height: 1.55;
    color: #64748B;
    margin-top: 12px;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.25s ease;
}

.module-card:hover .module-subtitle {
    opacity: 1;
    transform: translateY(0);
}

.home-footer {
    text-align: center;
    margin-top: 42px;
    color: #94A3B8;
    font-size: 13px;
}
</style>
"""


def module_card(icon: str, title: str, subtitle: str, button_label: str, target_page: str):
    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-icon">{icon}</div>
            <div class="module-title">{title}</div>
            <div class="module-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(button_label, key=f"open_{target_page}", use_container_width=True):
        st.session_state.page = target_page
        st.rerun()


def render_home():
    st.markdown(CARD_STYLE, unsafe_allow_html=True)

    st.markdown(
        '<div class="home-title">🔋 Battery R&D AI Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-subtitle">'
        'Data-driven modelling, prediction and optimisation for battery systems'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        module_card(
            icon="📊",
            title="Data System",
            subtitle="Explore, manage and analyse battery datasets with structured views and diagnostic plots.",
            button_label="Open Data System",
            target_page="data",
        )

    with col2:
        module_card(
            icon="🧠",
            title="Modelling & Prediction",
            subtitle="Build, evaluate and compare machine learning, hybrid and physics-informed models.",
            button_label="Open Modelling",
            target_page="modelling",
        )

    with col3:
        module_card(
            icon="⚙️",
            title="Control & Optimisation",
            subtitle="Design optimisation workflows, control strategies and decision-support pipelines.",
            button_label="Open Control",
            target_page="control",
        )

    st.markdown(
        '<div class="home-footer">'
        'Use the AI Copilot in the sidebar to interact with the platform.'
        '</div>',
        unsafe_allow_html=True,
    )