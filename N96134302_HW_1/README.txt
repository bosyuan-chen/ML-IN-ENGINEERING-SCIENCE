HW1 - Stock Price Prediction using Agentic Workflow

Student ID: N96134302
Course: Machine Learning in Engineering Science

Environment:
- Google Antigravity IDE
- Python 3.14.3

Data Source:
- yfinance
- S&P 500 (^GSPC)
- Date range: 2021-01-01 to 2025-12-31

Models:
- Random Forest
- XGBoost

Evaluation Metric:
- Mean Squared Error (MSE)

Train/Test Split:
- Training set: 2021-01-01 to 2024-12-31
- Testing set: 2025-01-01 to 2025-12-31
- No random shuffle

Files:
- hw1_stock_prediction.py: main Python code
- stock_prediction_results.png: actual vs predicted result plot
- mse_result.txt: model comparison result

Result Summary:
- Random Forest MSE: 185143.44
- XGBoost MSE: 198557.68
- Random Forest performed better in this experiment.

Notes:
This homework was developed using Google Antigravity IDE with AI-assisted workflow. 
The AI agent helped with planning and coding, while the final logic, chronological split, 
and result checking were manually reviewed to avoid look-ahead bias.