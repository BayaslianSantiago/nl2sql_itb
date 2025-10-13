# Consulta NL → SQL Dinámico
---
Este proyecto implementa una interfaz en Streamlit para realizar consultas en lenguaje natural (NL) sobre una base de datos SQL.
Utiliza un modelo de embeddings para convertir consultas NL en sentencias SQL correspondientes, permitiendo realizar consultas de forma más intuitiva. Además, el sistema permite agregar nuevas frases en NL y sus respectivas consultas SQL al diccionario del sistema, lo que facilita su expansión.
### Requisitos
* Python 3.7+
* Streamlit
* PyTorch
* Transformers
* SpaCy
* SQLite
* Scikit-learn

# Instalación de dependencias
Para instalar las dependencias del proyecto, se recomienda usar un entorno virtual:

* Crear un entorno virtual

        bash    
        python -m venv venv
        
* Activar el entorno virtual

        bash
        En Windows:
        venv\Scripts\activate

        En macOS/Linux:
        source venv/bin/activate
  
* Instalar las dependencias

        bash
        pip install -r requirements.txt

# Requisitos adicionales

* **Modelo spaCy**: El proyecto requiere el modelo de spaCy es_core_news_sm, que se descarga automáticamente si no está presente.
* **Modelo preentrenado para embeddings:** Se utiliza el modelo sentence-transformers/distiluse-base-multilingual-cased-v1 para generar los embeddings de las consultas NL.

### Descripción del Código

El archivo principal del proyecto es app_nl2sql_dynamic_fixed.py, que se puede ejecutar en Streamlit para interactuar con el sistema de consulta NL → SQL. 
A continuación, se detallan las principales secciones del código.

## 1. Inicialización de la NLP (SpaCy)
Se utiliza el modelo de procesamiento de lenguaje natural de spaCy para preprocesar el texto en español. Esto incluye:
* Conversión a minúsculas.
* Lematización.
* Eliminación de stopwords (palabras vacías).
Si el modelo *es_core_news_sm* no está instalado, el sistema lo descarga automáticamente.

        python
        try:
            nlp = spacy.load("es_core_news_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
            nlp = spacy.load("es_core_news_sm")

## 2. Modelo de Embeddings para Consultas NL → SQL

El modelo de **embeddings** se carga utilizando la librería **Hugging Face Transformers**.
El modelo *distiluse-base-multilingual-cased-v1* se utiliza para generar representaciones vectoriales de las consultas NL.
Estas representaciones permiten comparar la similitud entre la consulta del usuario y las frases predefinidas en el diccionario.

        python
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v1")
            model = AutoModel.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v1")
            return tokenizer, model

## 3. Conexión y Creación de Base de Datos SQLite
El proyecto utiliza **SQLite** para gestionar la base de datos local *empresa.db*, que incluye tablas para:
* Rubros de productos
* Productos
* Clientes
* Sucursales
* Ventas
* Facturas
Si no existe, se crea una base de datos con datos predefinidos.

        python
        def init_db():
            conn = sqlite3.connect("empresa.db")
            cursor = conn.cursor()
            # Creación de tablas e inserción de datos iniciales...

## 4. Diccionario NL → SQL

El sistema tiene un diccionario predefinido de *frases NL y sus respectivas consultas SQL*. Este diccionario puede ser modificado por el usuario para agregar nuevas entradas mediante la *interfaz de Streamlit*.

        python
        if os.path.exists(DICT_FILE):
            with open(DICT_FILE, "r", encoding="utf-8") as f:
                nl2sql_examples = json.load(f)
        else:
            nl2sql_examples = { ... }

## 5. Generación de SQL a partir de la Consulta del Usuario

Cuando un usuario ingresa una *consulta en lenguaje natural*, esta se *preprocesa* y se convierte en un *vector de embeddings*.
Luego, se compara con las frases predefinidas en el diccionario mediante la *similitud del coseno*. La consulta que tenga la mayor similitud se *ejecuta como una consulta SQL* en la base de datos.

        def query_to_sql(user_query):
            user_emb = embed(preprocess(user_query))
            sims = {k: cosine_similarity(user_emb, v)[0][0] for k, v in example_embeddings.items()}
            best_match = max(sims, key=sims.get)
            return nl2sql_examples[best_match], sims[best_match]

## 6. Ejecución de Consultas SQL
Una vez que se genera la *consulta SQL*, esta se ejecuta en la base de datos SQLite y se *muestran los resultados* en la interfaz de Streamlit.

        python
        def ejecutar_sql(sql):
            with sqlite3.connect("empresa.db") as conn:
                cur = conn.cursor()
                cur.execute(sql)
                resultados = cur.fetchall()
                columnas = [desc[0] for desc in cur.description]
            return columnas, resultados

7. Interfaz de Usuario en Streamlit
   
   1. La interfaz de usuario se desarrolla utilizando Streamlit. La aplicación permite:
   2. Ingresar una consulta en lenguaje natural.
   3. Mostrar la consulta SQL generada y el nivel de confianza.
   4. Ejecutar la consulta y mostrar los resultados en formato tabla.
   5. Agregar nuevas frases NL → SQL al diccionario para futuras consultas.
  
        python
        st.title("💡 Consulta NL → SQL Dinámico")
        consulta_nl = st.text_input("Ingrese su consulta:")
        if st.button("Ejecutar consulta") and consulta_nl:
            sql, score = query_to_sql(consulta_nl)
            columnas, resultados = ejecutar_sql(sql)
            st.dataframe(resultados)

# Como Usar
* Ejecuta el archivo principal con Streamlit:

        bash
        streamlit run app.py
  
* Ingresa tu consulta en lenguaje natural en el campo de texto y haz clic en Ejecutar consulta.
* La aplicación generará la consulta SQL correspondiente y mostrará los resultados en formato de tabla.
* También puedes agregar nuevas frases en lenguaje natural y sus respectivas consultas SQL al diccionario mediante la interfaz de Agregar nueva frase NL → SQL.
