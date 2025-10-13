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

    # Crear un entorno virtual
    python -m venv venv
    
    # Activar el entorno virtual
    # En Windows:
    venv\Scripts\activate
    # En macOS/Linux:
    source venv/bin/activate
    
    # Instalar las dependencias
    pip install -r requirements.txt
