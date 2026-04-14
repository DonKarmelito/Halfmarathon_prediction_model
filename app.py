import streamlit as st
import pandas as pd
import instructor
from langfuse.openai import OpenAI
from langfuse import observe
from dotenv import load_dotenv
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import boto3
import os
import matplotlib.pyplot as plt
from typing import Optional

load_dotenv()

# --- Konfiguracja S3 ---
s3 = boto3.client(
    's3',
    region_name='fra1',
    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

BUCKET_NAME = "halfmarathon-model"
CSV_KEY = "halfmarathon/halfmarathon_wroclaw_2023__final.csv"
MODEL_KEY = "halfmarathon/model_polmaraton.pkl"

# --- Session state ---
if "czas_pred" not in st.session_state:
    st.session_state["czas_pred"] = None


# --- Ładowanie danych i modelu z S3 ---
@st.cache_resource
def load_data_and_model():
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_KEY)
        df_hist = pd.read_csv(obj['Body'], sep=';')

        def czas_na_sekundy(t):
            if pd.isna(t): return None
            if isinstance(t, (int, float)): return float(t)
            t = str(t).strip()
            try:
                parts = t.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    return m * 60 + s
                else:
                    return None
            except:
                return None

        df_hist['Czas_s'] = df_hist['Czas'].apply(czas_na_sekundy)
        df_hist = df_hist.dropna(subset=['Czas_s'])
        st.success(f"Wczytano {len(df_hist)} wyników dla zawodników z 2023")

        model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        model_bytes = model_obj['Body'].read()
        with open('temp_model.pkl', 'wb') as f:
            f.write(model_bytes)
        model = load_model('temp_model')
        os.remove('temp_model.pkl')

        return df_hist, model

    except Exception as e:
        st.error(f"Błąd: {e}")
        st.stop()


# --- Predykcja ---
def predykcja_dla_zawodnika(plec, kat_wiekowa, czas_5km_s, model, df_hist):
    dane = pd.DataFrame([[plec, kat_wiekowa, czas_5km_s]],
                        columns=['Płeć', 'Kategoria wiekowa', '5 km Czas'])
    czas_pred = predict_model(model, data=dane)['prediction_label'].iloc[0]

    miejsce_open = (df_hist['Czas_s'] < czas_pred).sum() + 1
    df_kat = df_hist[df_hist['Kategoria wiekowa'] == kat_wiekowa]
    miejsce_kat = (df_kat['Czas_s'] < czas_pred).sum() + 1

    h = int(czas_pred // 3600)
    m = int((czas_pred % 3600) // 60)
    s = int(czas_pred % 60)

    return {
        'czas_pred_s': float(czas_pred),
        'przewidywany_czas': f"{h}:{m:02d}:{s:02d}",
        'miejsce_open': int(miejsce_open),
        'miejsce_kat': int(miejsce_kat),
        'na_miejsc': len(df_hist),
        'w_kat_na_miejsc': len(df_kat),
    }


# --- Model danych Pydantic ---
class Person(BaseModel):
    imie: Optional[str] = None
    plec: Optional[int] = None
    kategoria_wiekowa: Optional[str] = None
    czas_5km_s: Optional[int] = None
    dane_kompletne: bool = False
    komunikat_bledu: Optional[str] = None


# --- Klient OpenAI przez Langfuse (drop-in replacement) ---
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
instructor_openai_client = instructor.from_openai(openai_client)


# --- Funkcja LLM opakowana w Langfuse trace ---
@observe(name="ekstrakcja-danych-biegacza")
def przetworz_zapytanie(text):
    return instructor_openai_client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        response_model=Person,
        messages=[{
            "role": "system",
            "content": """Wyciągnij z tekstu dane biegacza: imie, plec (0=kobieta, 1=mężczyzna), 
            kategoria_wiekowa (M20/M30/M40/M50/M60/M70 lub K20/K30/K40/K50/K60/K70), 
            czas_5km_s (przelicz MM:SS lub minuty na sekundy).

            Ustaw dane_kompletne=True TYLKO jeśli wszystkie 4 pola są jednoznacznie podane w tekście.
            Jeśli czegoś brakuje, ustaw dane_kompletne=False i w komunikat_bledu napisz 
            po polsku czego brakuje (np. 'Brakuje: wieku/kategorii wiekowej, czasu na 5 km').
            Nie zgaduj ani nie wymyślaj danych których nie ma w tekście."""
        }, {
            "role": "user",
            "content": text,
        }]
    )


# --- Ładowanie danych i modelu ---
df_hist, digital_model = load_data_and_model()

# --- UI ---
st.title("🏃 Predykcja czasu półmaratonu Wrocław")
st.write("Opowiedz nam o sobie i podaj swój czas na 5 km.")
text = st.text_input(
    "Opisz się",
    placeholder="np. Mam na imię Marek, mam 35 lat, jestem mężczyzną, 5 km biegnę w 25:30"
)

if st.button("Oblicz") and text:
    with st.spinner("Analizuję dane..."):
        res = przetworz_zapytanie(text)

    if not res.dane_kompletne:
        st.warning(
            f"Nie udało mi się zebrać wszystkich potrzebnych danych. "
            f"{res.komunikat_bledu or ''}\n\n"
            "Podaj proszę:\n"
            "- **Imię**\n"
            "- **Wiek** (lub kategorię wiekową, np. M35, K40)\n"
            "- **Płeć** (mężczyzna / kobieta)\n"
            "- **Czas na 5 km** (np. 25:30)"
        )
    else:
        wynik = predykcja_dla_zawodnika(
            res.plec, res.kategoria_wiekowa, res.czas_5km_s, digital_model, df_hist
        )
        st.session_state["czas_pred"] = wynik["czas_pred_s"]

        st.subheader(f"Cześć {res.imie}! Twój przewidywany wynik w kategorii {res.kategoria_wiekowa} to:")
        c1, c2, c3 = st.columns(3)
        c1.metric("Czas przebiegnięcia półmaratonu", wynik['przewidywany_czas'])
        c2.metric("Miejsce OPEN", f"{wynik['miejsce_open']} / {wynik['na_miejsc']}")
        c3.metric("Miejsce w kategorii wiekowej", f"{wynik['miejsce_kat']} / {wynik['w_kat_na_miejsc']}")

# --- Wykres ---
st.subheader("Rozkład wyników (wszyscy zawodnicy)")
fig, ax = plt.subplots()
ax.hist(df_hist['Czas_s'], bins=50)

if st.session_state["czas_pred"] is not None:
    ax.axvline(x=st.session_state["czas_pred"], color='red', linestyle='--', label='Twój wynik')
    ax.legend()

ax.set_xlabel("Czas (sekundy)")
ax.set_ylabel("Liczba zawodników")
st.pyplot(fig)
