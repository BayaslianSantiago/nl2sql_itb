# app_nl2sql_dynamic_fixed.py
import streamlit as st
import sqlite3
import pandas as pd
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import os
import subprocess
import sys

# ------------------------
# Inicialización NLP
# ------------------------


def preprocess(text):
    """Preprocesa texto en español: minúsculas, lematización, stopwords"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# ------------------------
# Modelo de embeddings
# ------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
    model = AutoModel.from_pretrained(
        "sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
    return tokenizer, model

tokenizer, model = load_model()

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# ------------------------
# Conexión y creación DB
# ------------------------
def init_db():
    conn = sqlite3.connect("empresa.db")
    cursor = conn.cursor()

    # Tablas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rubros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT UNIQUE NOT NULL
    );""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS productos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        precio REAL NOT NULL,
        rubro_id INTEGER,
        FOREIGN KEY(rubro_id) REFERENCES rubros(id)
    );""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clientes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        email TEXT UNIQUE
    );""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sucursales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        direccion TEXT
    );""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ventas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cliente_id INTEGER,
        sucursal_id INTEGER,
        total REAL NOT NULL,
        fecha TEXT,
        FOREIGN KEY(cliente_id) REFERENCES clientes(id),
        FOREIGN KEY(sucursal_id) REFERENCES sucursales(id)
    );""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS facturas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        venta_id INTEGER,
        fecha_emision TEXT,
        FOREIGN KEY(venta_id) REFERENCES ventas(id)
    );""")

    # Datos iniciales
    cursor.executemany("INSERT OR IGNORE INTO rubros (nombre) VALUES (?)", [
        ("Electrónica",), ("Alimentos",), ("Ropa",), ("Hogar",)
    ])
    cursor.executemany("INSERT OR IGNORE INTO productos (nombre, precio, rubro_id) VALUES (?, ?, ?)", [
        ("Televisor", 50000, 1), ("Notebook", 350000, 1),
        ("Pan", 500, 2), ("Camisa", 12000, 3), ("Silla", 15000, 4)
    ])
    cursor.executemany("INSERT OR IGNORE INTO clientes (nombre, email) VALUES (?, ?)", [
        ("Juan Pérez", "juan@example.com"), ("María Gómez", "maria@example.com")
    ])
    cursor.executemany("INSERT OR IGNORE INTO sucursales (nombre, direccion) VALUES (?, ?)", [
        ("Sucursal Centro", "Av. Siempre Viva 123"), ("Sucursal Norte", "Calle Falsa 456")
    ])
    cursor.executemany("INSERT OR IGNORE INTO ventas (cliente_id, sucursal_id, total, fecha) VALUES (?, ?, ?, ?)", [
        (1, 1, 50500, "2025-09-01"), (2, 2, 362000, "2025-09-05")
    ])
    cursor.executemany("INSERT OR IGNORE INTO facturas (venta_id, fecha_emision) VALUES (?, ?)", [
        (1, "2025-09-02"), (2, "2025-09-06")
    ])
    conn.commit()
    conn.close()

init_db()

# ------------------------
# Diccionario NL → SQL dinámico
# ------------------------
DICT_FILE = "nl2sql_dict.json"

if os.path.exists(DICT_FILE):
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        nl2sql_examples = json.load(f)
else:
    nl2sql_examples = {
        "listar clientes": "SELECT * FROM clientes;",
        "listar productos": "SELECT * FROM productos;",
        "listar rubros": "SELECT * FROM rubros;",
        "ventas por sucursal": "SELECT sucursal_id, SUM(total) AS total_ventas FROM ventas GROUP BY sucursal_id;",
        "ventas por cliente": "SELECT cliente_id, SUM(total) AS total_cliente FROM ventas GROUP BY cliente_id;",
        "ventas totales": "SELECT SUM(total) AS total_general FROM ventas;",
        "facturas emitidas": "SELECT * FROM facturas;",
        "productos por rubro": "SELECT rubro_id, COUNT(*) AS cantidad_productos FROM productos GROUP BY rubro_id;"
    }
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(nl2sql_examples, f, ensure_ascii=False, indent=4)

# Precalcular embeddings
@st.cache_data
def precalc_embeddings():
    return {k: embed(preprocess(k)) for k in nl2sql_examples.keys()}

example_embeddings = precalc_embeddings()

def query_to_sql(user_query):
    user_emb = embed(preprocess(user_query))
    sims = {k: cosine_similarity(user_emb, v)[0][0] for k, v in example_embeddings.items()}
    best_match = max(sims, key=sims.get)
    score = sims[best_match]
    return nl2sql_examples[best_match], score

# ------------------------
# Ejecutar SQL
# ------------------------
def ejecutar_sql(sql):
    with sqlite3.connect("empresa.db") as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            resultados = cur.fetchall()
            columnas = [desc[0] for desc in cur.description] if cur.description else []
        except Exception as e:
            resultados = [("Error:", str(e))]
            columnas = []
    return columnas, resultados

# ------------------------
# Streamlit UI
# ------------------------
st.title("💡 Consulta NL → SQL Dinámico")

# --- Sección de consulta ---
st.subheader("🔍 Consulta en lenguaje natural")
consulta_nl = st.text_input("Ingrese su consulta:")

if st.button("Ejecutar consulta") and consulta_nl:
    sql, score = query_to_sql(consulta_nl)
    st.markdown(f"**SQL generado:** `{sql}`")
    st.markdown(f"**Confianza:** {round(score,3)}")
    
    if score < 0.6:
        st.warning("❗ Baja confianza. Reformule la consulta.")
    
    columnas, resultados = ejecutar_sql(sql)
    if resultados:
        df = pd.DataFrame(resultados, columns=columnas)
        st.dataframe(df)
    else:
        st.info("No se encontraron resultados.")

# --- Sección para agregar nuevas frases ---
st.subheader("➕ Agregar nueva frase NL → SQL")

with st.form(key="form_agregar_nl_sql"):
    nueva_nl = st.text_input("Nueva frase en lenguaje natural")
    nuevo_sql = st.text_area("Consulta SQL correspondiente")
    
    submitted = st.form_submit_button("Agregar al diccionario")
    
    if submitted:
        if nueva_nl.strip() and nuevo_sql.strip():
            nl2sql_examples[nueva_nl] = nuevo_sql
            with open(DICT_FILE, "w", encoding="utf-8") as f:
                json.dump(nl2sql_examples, f, ensure_ascii=False, indent=4)
            example_embeddings[nueva_nl] = embed(preprocess(nueva_nl))
            st.success(f"✅ Frase agregada correctamente: '{nueva_nl}'")
        else:
            st.error("❗ Complete ambos campos antes de agregar.")


