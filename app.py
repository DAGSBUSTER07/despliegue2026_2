import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configurar la página de Streamlit
st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")

st.title("Predicción de Nota Final - Curso")
st.write("Introduce los datos del estudiante de forma manual o sube un archivo Excel para realizar predicciones en lote.")

# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load('one_hot_columns.joblib')
        scaler = joblib.load('min_max_scaler.joblib')
        model = joblib.load('bagging_optimizado.joblib')
        return columnas_one_hot, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None

columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot and scaler and model:
    # Crear pestañas para separar la predicción individual de la predicción por archivo
    tab1, tab2 = st.tabs(["Individual", "Carga de archivo Excel (Lote)"])

    # Extraer las categorías posibles para la variable 'Felder'
    categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]

    # --- PESTAÑA 1: PREDICCIÓN INDIVIDUAL ---
    with tab1:
        st.header("Datos del Estudiante")
        felder_selected = st.selectbox("Estilo de Aprendizaje (Felder)", options=categorias_felder, key="ind_felder")
        examen_admision = st.slider("Nota de Examen de Admisión", min_value=0.0, max_value=5.0, value=3.8, step=0.05, key="ind_examen")

        if st.button("Calcular Predicción Individual", key="btn_ind"):
            df_input = pd.DataFrame([{'Felder': felder_selected, 'Examen_admisión': examen_admision}])

            # Aplicar codificación One-Hot manual de acuerdo a la lista cargada
            for col in columnas_one_hot:
                if col.startswith('Felder_'):
                    categoria = col.replace('Felder_', '')
                    df_input[col] = 1.0 if felder_selected == categoria else 0.0

            # Aplicar el Min-Max Scaler cargado
            df_input['Examen_admision_scaled'] = scaler.transform(df_input[['Examen_admisión']])[0][0]

            # Seleccionar y ordenar las columnas según las que espera el modelo
            columnas_finales = [col for col in columnas_one_hot if col in df_input.columns]
            df_procesado = df_input[columnas_finales]

            # Realizar la predicción con el modelo
            prediccion = model.predict(df_procesado)[0]

            st.success(f"### Nota Final Estimada: {prediccion:.3f}")

            with st.expander("Ver variables procesadas enviadas al modelo"):
                st.dataframe(df_procesado)

    # --- PESTAÑA 2: CARGA DE ARCHIVO EXCEL ---
    with tab2:
        st.header("Subir Archivo Excel")
        st.write("El archivo Excel debe contener al menos las columnas: `Felder` y `Examen_admisión`.")
        
        uploaded_file = st.file_uploader("Selecciona un archivo Excel (.xlsx)", type=["xlsx"])

        if uploaded_file is not None:
            try:
                # Leer el archivo Excel
                df_uploaded = pd.read_excel(uploaded_file)
                st.subheader("Vista previa de los datos subidos:")
                st.dataframe(df_uploaded.head())

                # Validar columnas necesarias
                cols_requeridas = ['Felder', 'Examen_admisión']
                missing_cols = [c for c in cols_requeridas if c not in df_uploaded.columns]

                if missing_cols:
                    st.error(f"El archivo no contiene las siguientes columnas requeridas: {missing_cols}")
                else:
                    if st.button("Procesar y Predecir en Lote", key="btn_batch"):
                        # Crear una copia para procesamiento
                        df_batch = df_uploaded.copy()

                        # Aplicar codificación One-Hot manual para cada fila
                        for col in columnas_one_hot:
                            if col.startswith('Felder_'):
                                categoria = col.replace('Felder_', '')
                                df_batch[col] = df_batch['Felder'].apply(lambda x: 1.0 if str(x).strip() == categoria else 0.0)

                        # Aplicar el Min-Max Scaler cargado
                        df_batch['Examen_admision_scaled'] = scaler.transform(df_batch[['Examen_admisión']])

                        # Seleccionar y ordenar las columnas según las que espera el modelo
                        columnas_finales = [col for col in columnas_one_hot if col in df_batch.columns]
                        df_procesado_batch = df_batch[columnas_finales]

                        # Realizar predicciones para todo el lote
                        predicciones_lote = model.predict(df_procesado_batch)

                        # Añadir la predicción al DataFrame original
                        df_uploaded['Nota_final_predicha'] = predicciones_lote

                        st.success("¡Predicciones calculadas exitosamente!")
                        st.subheader("Resultados de las predicciones:")
                        st.dataframe(df_uploaded)

                        # Permitir descargar los resultados procesados
                        @st.cache_data
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')

                        csv = convert_df(df_uploaded)
                        st.download_button(
                            label="Descargar resultados en CSV",
                            data=csv,
                            file_name="predicciones_lote.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo: {e}")
else:
    st.warning("Por favor, asegúrate de que los archivos 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' se encuentren en la ruta correcta.")
