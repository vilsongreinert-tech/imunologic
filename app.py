import os
from flask import Flask, render_template, request
import psycopg2

app = Flask(__name__)

DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')

# Configurações do seu banco
DB_CONFIG = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASS,
    "port": "5432"
}

def buscar_no_banco(referencia):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        # Usamos %s para evitar ataques de SQL Injection 🛡️
        query = "SELECT DESMAT, ESPMAT FROM tt_pro WHERE refBAS = %s"
        cur.execute(query, (referencia,))
        resultado = cur.fetchone()
        cur.close()
        conn.close()
        return resultado
    except Exception as e:
        print(f"Erro: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    produto = None
    if request.method == 'POST':
        ref = request.form.get('referencia')
        if ref:
            produto = buscar_no_banco(ref)
    
    return render_template('index.html', produto=produto)

if __name__ == '__main__':
    app.run(debug=True)
