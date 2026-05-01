# Battery R&D Platform - Project Context

## Goal
Build a Streamlit-based Battery R&D platform for data browsing, modelling/prediction, AI control/optimisation, and AI Copilot interaction.

## Current Architecture
- Frontend: Streamlit
- Data backend: Supabase, planned
- Computation layer: core/ services
- AI Agent layer: agents/ router + specialised agents
- Tool layer: tools/ callable functions

## Main UI Design
Homepage contains 3 primary modules:
1. Data System
2. Modelling & Prediction
3. AI Control & Optimisation

A persistent AI Copilot panel is available from the sidebar.

## Development Principle
- Keep UI, core logic, tools, and agents separated.
- Do not put heavy computation directly in Streamlit UI files.
- Future FastAPI migration should wrap existing core services instead of rewriting logic.

## Current Progress
Initial runnable Streamlit skeleton created with:
- app.py
- ui/home.py
- ui/copilot.py
- ui/data_module.py
- ui/modelling_module.py
- ui/control_module.py
- agents/router_agent.py
- agents/data_agent.py
- agents/modelling_agent.py
- agents/control_agent.py
- core/supabase_client.py
- core/data_service.py
- core/model_service.py
- core/control_service.py

# Project Status (2026-04-29)

## Progress
- Streamlit project initialized
- NASA .mat files converted to CSV successfully
- Supabase project created
- API URL and service_role key obtained
- secrets.toml configured

## Next Step
- Create tables in Supabase
- Upload CSV data
- Connect Streamlit to Supabase
- Visualize battery data

## Notes
- Using Supabase (not BigQuery)
- Using service_role key for upload
- Data structure: cells / cycles / timeseries / impedance