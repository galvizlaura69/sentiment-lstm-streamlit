"""Aplicación Streamlit en español para clasificar reseñas de películas."""
import streamlit as st
from predict import load_assets, predict_review

st.set_page_config(page_title="Analizador de reseñas de películas", page_icon="🎬")
st.title("Analizador de sentimiento de reseñas de películas")
st.write("Escribe una reseña en español y el modelo estimará si su opinión general es positiva o negativa.")

with st.expander("¿Qué debe incluir la reseña?", expanded=True):
    st.markdown(
        "- **Idioma:** español.\n"
        "- **Extensión recomendada:** un párrafo de 2 a 5 oraciones.\n"
        "- **Contenido:** opinión general y detalles como la historia, las actuaciones, el ritmo o la música.\n"
        "- **Formato:** no requiere una plantilla fija; escribe la reseña como lo harías normalmente.\n"
        "- **Consejo:** expresa una valoración clara y evita incluir solo el título de la película."
    )
st.caption("El modelo se entrenó con reseñas IMDb traducidas al español. Algunas traducciones del dataset pueden contener errores.")

@st.cache_resource
def get_assets():
    return load_assets()

try:
    model, tokenizer, config = get_assets()
except FileNotFoundError as exc:
    st.warning("Todavía falta entrenar el modelo con las reseñas en español.")
    st.code("# 1. Guarda el CSV descargado en data/IMDB Dataset SPANISH.csv\npython train.py\n\n# 2. Reinicia la aplicación\nstreamlit run app.py", language="powershell")
    st.caption(str(exc))
    st.stop()

review = st.text_area(
    "Escribe aquí tu reseña en español",
    height=200,
    max_chars=5000,
    placeholder="La historia me pareció emocionante, las actuaciones fueron muy buenas y el final me sorprendió...",
)

if st.button("Analizar reseña", type="primary"):
    if not review.strip():
        st.warning("Escribe una reseña antes de analizarla.")
    elif len(review.split()) < 5:
        st.warning("Agrega un poco más de texto. Escribe al menos una opinión y algún detalle de la película.")
    else:
        sentiment, confidence, positive_probability = predict_review(review, model, tokenizer, config)
        st.subheader(f"Resultado: reseña {sentiment}")
        st.metric("Confianza estimada en esta predicción", f"{confidence:.1%}")
        with st.expander("Ver probabilidades por clase"):
            st.write(f"Positiva: **{positive_probability:.1%}**")
            st.write(f"Negativa: **{1 - positive_probability:.1%}**")
