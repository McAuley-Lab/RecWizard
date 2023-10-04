# Use $1 as port
echo "starting api on port $1"
uvicorn api:app --port $1 & streamlit run demo.py $1
