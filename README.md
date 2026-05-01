# Battery R&D AI Platform

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

- `app.py`: Streamlit entry point
- `ui/`: Streamlit UI pages and Copilot panel
- `agents/`: AI Agent router and specialised agent skeletons
- `core/`: business logic and future Supabase / ML / control services
- `tools/`: future callable tools for agents
- `.streamlit/secrets.toml`: local secrets, not for GitHub public exposure

## Next steps

1. Connect Supabase
2. Add real battery data tables
3. Add Data module upload/query functions
4. Add modelling workflow
5. Add AI Agent tool calling
