# 💡 Consulta NL → SQL Dinámico

Aplicación web interactiva que permite realizar consultas a bases de datos SQL utilizando lenguaje natural en español. Powered by Streamlit y Transformers.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-app.streamlit.app)

## 🌟 Características

- 🗣️ **Consultas en lenguaje natural**: Pregunta en español y obtén resultados SQL automáticamente
- 🧠 **Machine Learning**: Usa embeddings multilingües para entender la intención del usuario
- 📊 **Base de datos de ejemplo**: Incluye datos de productos, clientes, ventas y facturas
- ➕ **Diccionario expandible**: Agrega tus propios patrones de consulta dinámicamente
- 🎯 **Score de confianza**: Muestra qué tan seguro está el modelo de la interpretación
- 🖥️ **Interfaz intuitiva**: Diseño limpio con tabs y sidebar informativo

## 🚀 Demo en vivo

Prueba la aplicación aquí: [https://nl2sql-ia.streamlit.app/](https://tu-app.streamlit.app](https://nl2sql-ia.streamlit.app/)

## 📋 Requisitos

- Python 3.8+
- Streamlit
- PyTorch
- Transformers (HuggingFace)
- Scikit-learn
- Pandas

## 🛠️ Instalación local

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/nl-to-sql.git
cd nl-to-sql
```

2. **Crea un entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecuta la aplicación**
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Uso

### 1. Realizar consultas

Escribe tu pregunta en lenguaje natural en la pestaña **"Consultar"**:

```
Ejemplos:
- "listar todos los clientes"
- "ventas totales"
- "productos por rubro"
- "mostrar facturas"
```

El sistema:
- Encuentra la consulta SQL más similar
- Muestra el score de confianza
- Ejecuta la consulta y muestra los resultados

### 2. Agregar nuevos patrones

En la pestaña **"Agregar Frase"** puedes expandir el diccionario:

**Frase natural:** `productos más vendidos`  
**SQL:** `SELECT p.nombre, COUNT(*) as ventas FROM productos p JOIN ventas v GROUP BY p.id ORDER BY ventas DESC LIMIT 10;`

### 3. Ver ejemplos disponibles

La pestaña **"Ver Ejemplos"** muestra todos los patrones del diccionario actual.

## 🗄️ Estructura de la Base de Datos

La aplicación incluye una base de datos SQLite de ejemplo con las siguientes tablas:

- **rubros**: Categorías de productos
- **productos**: Inventario de productos con precios
- **clientes**: Información de clientes
- **sucursales**: Ubicaciones de venta
- **ventas**: Registros de transacciones
- **facturas**: Documentos de facturación

### Diagrama ER

```
rubros ──┐
         │
         ├─── productos
         
clientes ──┐
           │
sucursales ├─── ventas ─── facturas
           │
```

##  Tecnología

### Modelo de Embeddings

Utiliza `paraphrase-multilingual-MiniLM-L12-v2` de Sentence Transformers:
- Multilingüe (español incluido)
- Ligero (~120MB)
- Rápido para inference
- Genera embeddings de 384 dimensiones

### Procesamiento de Lenguaje Natural

- **Preprocesamiento**: Normalización de texto, eliminación de stopwords
- **Similitud coseno**: Encuentra la consulta más similar en el diccionario
- **Threshold de confianza**: Alerta cuando el score es < 0.6

## 📁 Estructura del proyecto

```
nl-to-sql/
│
├── app.py                 # Aplicación principal
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
└── .gitignore            # Archivos ignorados
```

## 🔧 Configuración avanzada

### Cambiar el modelo de embeddings

En `app.py`, línea 35-38, puedes cambiar el modelo:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/otro-modelo-multilingue"
)
```

### Modificar el threshold de confianza

En `app.py`, línea 210, ajusta el valor:

```python
if score < 0.6:  # Cambiar este valor
```

### Agregar más tablas a la BD

Modifica la función `init_db()` en `app.py` para incluir nuevas tablas y datos.

La primera carga tardará ~5-10 minutos mientras descarga el modelo.

## Limitaciones

- La base de datos se reinicia con cada deploy en Streamlit Cloud
- El diccionario NL→SQL no persiste entre sesiones (se reinicia al recargar)
- Solo soporta consultas SELECT (por seguridad)
- El modelo puede no entender consultas muy complejas o ambiguas


##  Agradecimientos

- [Streamlit](https://streamlit.io/) - Framework de la aplicación
- [HuggingFace](https://huggingface.co/) - Modelos de Transformers
- [Sentence Transformers](https://www.sbert.net/) - Embeddings semánticos
- [Instituto Tecnológico Beltrán](https://www.ibeltran.com.ar/) - Por los recursos y a sus docentes por las enseñanzas
---
