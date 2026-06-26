import json 
import sys 
import datetime 
import yfinance as yf 
import pandas as pd 
import numpy as np 
 
def handler(request): 
    try: 
        ticker = request.query.get('ticker', 'GC=F') 
        df = yf.download(ticker, period="180d", progress=False) 
        close = df['Close'] 
        ma20 = close.rolling(20).mean().iloc[-1] 
        ma50 = close.rolling(50).mean().iloc[-1] 
        return { 
            'statusCode': 200, 
            'body': json.dumps({ 
                'ticker': ticker, 
                'action': action, 
                'price': round(close.iloc[-1], 2), 
                'ma20': round(ma20, 2), 
                'ma50': round(ma50, 2) 
            }) 
        } 
    except Exception as e: 
        return { 
            'statusCode': 500, 
            'body': json.dumps({'error': str(e)}) 
        } 
