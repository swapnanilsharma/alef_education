# alef_education



pids=$(lsof -ti tcp:8000); [ -n "$pids" ] && kill -9 $pids; /Users/swapnanilsharmah/Documents/alef_education/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

/Users/swapnanilsharmah/Documents/alef_education/.venv/bin/streamlit run streamlit_app.py