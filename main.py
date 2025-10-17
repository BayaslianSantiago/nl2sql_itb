import streamlit as st
import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re

# ------------------------
# Configuración de página
# ------------------------
st.set_page_config(
    page_title="Consulta NL → SQL",
    page_icon="💡",
    layout="wide"
)

# ------------------------
# Preprocesamiento simplificado (sin spaCy)
# ------------------------
def preprocess(text):
    """Preprocesa texto: minúsculas y stopwords básicas"""
    # Stopwords básicas en español
    stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 
                 'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 
                 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o', 'poder', 'decir',
                 'este', 'ir', 'otro', 'ese', 'si', 'me', 'ya', 'ver', 'porque',
                 'dar', 'cuando', 'él', 'muy', 'sin', 'vez', 'mucho', 'saber',
                 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo', 'también', 'hasta',
                 'año', 'dos', 'querer', 'entre', 'así', 'primero', 'desde', 'grande',
                 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí',
                 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
                 'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora',
                 'parte', 'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar',
                 'llevar', 'dejar', 'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar'}
    
    text = text.lower()
    # Eliminar caracteres especiales y mantener solo letras y espacios
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # Filtrar stopwords
    tokens = [word for word in text.split() if word not in stopwords and len(word) > 2]
    return " ".join(tokens)

# ------------------------
# Modelo de embeddings (más ligero)
# ------------------------
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_model():
    """Carga modelo ligero de sentence-transformers"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        model = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.stop()

tokenizer, model = load_model()

def embed(text):
    """Genera embedding para un texto"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# ------------------------
# Conexión y creación DB
# ------------------------
@st.cache_resource
def init_db():
    """Inicializa la base de datos con datos de ejemplo"""
    conn = sqlite3.connect("empresa.db", check_same_thread=False)
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
    return conn

conn = init_db()

# ------------------------
# Diccionario NL → SQL en session_state
# ------------------------
if 'nl2sql_examples' not in st.session_state:
    st.session_state.nl2sql_examples = {
        "listar clientes": "SELECT * FROM clientes;",
        "mostrar clientes": "SELECT * FROM clientes;",
        "listar productos": "SELECT * FROM productos;",
        "mostrar productos": "SELECT * FROM productos;",
        "listar rubros": "SELECT * FROM rubros;",
        "mostrar rubros": "SELECT * FROM rubros;",
        "ventas por sucursal": "SELECT sucursal_id, SUM(total) AS total_ventas FROM ventas GROUP BY sucursal_id;",
        "ventas por cliente": "SELECT cliente_id, SUM(total) AS total_cliente FROM ventas GROUP BY cliente_id;",
        "ventas totales": "SELECT SUM(total) AS total_general FROM ventas;",
        "total ventas": "SELECT SUM(total) AS total_general FROM ventas;",
        "facturas emitidas": "SELECT * FROM facturas;",
        "mostrar facturas": "SELECT * FROM facturas;",
        "productos por rubro": "SELECT rubro_id, COUNT(*) AS cantidad_productos FROM productos GROUP BY rubro_id;",
        "listar sucursales": "SELECT * FROM sucursales;",
        "mostrar sucursales": "SELECT * FROM sucursales;",
    }

if 'example_embeddings' not in st.session_state:
    with st.spinner("Precalculando embeddings..."):
        st.session_state.example_embeddings = {
            k: embed(preprocess(k)) for k in st.session_state.nl2sql_examples.keys()
        }

def query_to_sql(user_query):
    """Encuentra la consulta SQL más similar usando embeddings"""
    user_emb = embed(preprocess(user_query))
    sims = {
        k: cosine_similarity(user_emb, v)[0][0] 
        for k, v in st.session_state.example_embeddings.items()
    }
    best_match = max(sims, key=sims.get)
    score = sims[best_match]
    return st.session_state.nl2sql_examples[best_match], score, best_match

# ------------------------
# Ejecutar SQL
# ------------------------
def ejecutar_sql(sql):
    """Ejecuta una consulta SQL y retorna resultados"""
    cur = conn.cursor()
    try:
        cur.execute(sql)
        resultados = cur.fetchall()
        columnas = [desc[0] for desc in cur.description] if cur.description else []
        error = None
    except Exception as e:
        resultados = []
        columnas = []
        error = str(e)
    return columnas, resultados, error

# ------------------------
# Streamlit UI
# ------------------------
st.title("💡 Consulta NL → SQL Dinámico")
st.markdown("Realiza consultas a la base de datos usando lenguaje natural")

# --- Tabs principales ---
tab1, tab2, tab3 = st.tabs(["🔍 Consultar", "➕ Agregar Frase", "📊 Ver Ejemplos"])

with tab1:
    st.subheader("Consulta en lenguaje natural")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        consulta_nl = st.text_input("Ingrese su consulta:", placeholder="Ej: listar todos los productos")
    with col2:
        ejecutar = st.button("🚀 Ejecutar", use_container_width=True)
    
    if ejecutar and consulta_nl:
        with st.spinner("Procesando consulta..."):
            sql, score, match = query_to_sql(consulta_nl)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Coincidencia detectada", match)
            with col_b:
                st.metric("Confianza", f"{round(score*100, 1)}%")
            
            st.code(sql, language="sql")
            
            if score < 0.6:
                st.warning("❗ Baja confianza. Considera reformular la consulta o agregar este patrón al diccionario.")
            
            columnas, resultados, error = ejecutar_sql(sql)
            
            if error:
                st.error(f"Error al ejecutar SQL: {error}")
            elif resultados:
                df = pd.DataFrame(resultados, columns=columnas)
                st.dataframe(df, use_container_width=True)
                st.success(f"✅ {len(resultados)} resultados encontrados")
            else:
                st.info("ℹ️ No se encontraron resultados.")

with tab2:
    st.subheader("Agregar nueva frase NL → SQL")
    st.markdown("Expande el diccionario agregando nuevos patrones de consulta")
    
    with st.form(key="form_agregar_nl_sql"):
        nueva_nl = st.text_input("Nueva frase en lenguaje natural", 
                                  placeholder="Ej: productos más caros")
        nuevo_sql = st.text_area("Consulta SQL correspondiente", 
                                  placeholder="Ej: SELECT * FROM productos ORDER BY precio DESC LIMIT 10;")
        
        submitted = st.form_submit_button("➕ Agregar al diccionario")
        
        if submitted:
            if nueva_nl.strip() and nuevo_sql.strip():
                # Agregar al diccionario
                st.session_state.nl2sql_examples[nueva_nl] = nuevo_sql
                # Calcular embedding
                st.session_state.example_embeddings[nueva_nl] = embed(preprocess(nueva_nl))
                st.success(f"✅ Frase agregada correctamente: '{nueva_nl}'")
                st.balloons()
            else:
                st.error("❗ Complete ambos campos antes de agregar.")

with tab3:
    st.subheader("Ejemplos disponibles en el diccionario")
    st.markdown(f"**Total de patrones:** {len(st.session_state.nl2sql_examples)}")
    
    # Mostrar en formato tabla
    ejemplos_df = pd.DataFrame([
        {"Frase Natural": k, "SQL": v} 
        for k, v in st.session_state.nl2sql_examples.items()
    ])
    st.dataframe(ejemplos_df, use_container_width=True, height=400)

# --- Sidebar con información ---
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    ### Cómo usar:
    1. **Consultar**: Escribe tu pregunta en lenguaje natural
    2. **Agregar**: Expande el diccionario con nuevos patrones
    3. **Ver ejemplos**: Revisa todos los patrones disponibles
    
    ### Ejemplos de consultas:
    - "listar clientes"
    - "ventas totales"
    - "productos por rubro"
    - "mostrar facturas"
    
    ### Estructura de la BD:
    - 📋 Clientes
    - 🛍️ Productos
    - 🏷️ Rubros
    - 🏢 Sucursales
    - 💰 Ventas
    - 📄 Facturas
    """)
    
    st.markdown("---")
    st.markdown("**Nota:** Los datos se reinician al recargar la app")
