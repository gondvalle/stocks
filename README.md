# S&P 500 Screener

Proyecto modular de screener fundamental basado únicamente en datos de `yfinance`.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Opcionalmente coloque un archivo `data/sp500_constituents.csv` con columnas `Ticker,Name,Sector`.

## Variables de entorno

Cree un archivo `.env` basado en `.env.example` con:

```
RISK_FREE=4.0
```
