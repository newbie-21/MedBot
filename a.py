import streamlit as st
import pandas as pd
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Load the CSV file for Kendra Locator
df = pd.read_csv('location.csv', encoding='Windows-1252')

# Initialize session state for selected service and chatbot history
if 'selected_service' not in st.session_state:
    st.session_state.selected_service = "Kendr Locator"
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''


st.set_page_config(layout="centered", initial_sidebar_state="expanded")

st.sidebar.title("KENDR LOCATOR")
st.sidebar.write("Find One Near You!")

display_option = st.sidebar.selectbox("Select:", ["Address", "Email"])
pin_code_input = st.sidebar.text_input("Enter Pin Code:")

if st.sidebar.button("Locate"):
    if pin_code_input:

        result = df[df['Pin'].astype(str) == pin_code_input]
        if not result.empty:
            st.sidebar.write(f"**Name**: {result['Name'].values[0]}")

            if display_option == "Address":
                st.sidebar.write(f"**Address**: {result['Address'].values[0]}")
            elif display_option == "Email":
                st.sidebar.write(f"**Email**: {result['Email'].values[0]}")
        else:
            st.sidebar.write("No results found.")
    else:
        st.sidebar.write("Please enter a pin code.")


llm = LlamaCpp(
    model_path="model.gguf",
    temperature=0.3,
    max_tokens=1024,
    top_p=1,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=False,
    stop=["###"]
)

template = """You are a knowledgeable, conversational assistant. Below is an Question that describes a query. Give out a Response that appropriately completes that query.

### Question:
{}

### Response:
{}"""

prompt = PromptTemplate.from_template(template)

PROFANE_WORDS = [
    "damn", "shit", "fuck", "bitch", "asshole", "dick", "piss", "crap", "cunt",
    "twat", "slut", "whore", "faggot", "nigger", "kike", "chink", "gook", "spic",
    "dyke", "suck", "cock", "pussy", "motherfucker", "bastard", "prick", "wanker",
    "bollocks", "arse", "bloody", "bugger", "tosser", "git", "slag", "pillock",
    "knob", "knobhead", "wazzock", "clit", "scrotum", "fanny", "ass", "freak",
    "bimbo", "dumbass", "jackass", "wimp", "idiot", "moron", "loser", "fool",
    "retard", "cocksucker", "shag", "shagger", "piss off", "go to hell",
    "dammit", "son of a bitch", "jerk", "puke", "chut", "chutiyah",
    "bhosdike", "bhenchod", "madarchod", "gandu", "gand", "bhancho",
    "saala", "kameena", "bhenji", "bhadwa", "kothi", "aankhmar", "launda",
    "bhikari", "sala", "bhosdika", "kothi", "sundar", "langda",
    "kaamchor", "gaddha", "bakra", "chudiya", "gando", "bhencod", "lanat",
    "bhoot", "chakkar", "chutak", "haramkhor", "bandar", "banda", "bakwas",
    "nikamma", "pagal", "nalayak", "pagal", "khota", "madharchod"
]


def contains_profanity(text):
    """Check if the text contains any profane words."""
    return any(word in text.lower() for word in PROFANE_WORDS)


def truncate_at_full_stop(text, max_length=1024):
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    print(f"Truncated text: {truncated}")

    last_period = truncated.rfind('.')
    print(f"Last period index: {last_period}")

    if last_period != -1:
        return truncated[:last_period + 1]

    return truncated


df1 = pd.read_csv(
    'med_name.csv', encoding='utf-8'
)

# Convert the 'Meds' column to a lowercase list
KNOWN_MEDICINES = df1['Meds   '].str.lower().tolist()


def contains_medicine_terms(output):
    """Check if the output contains terms that indicate a medicine name."""
    return any(term in output for term in [" IP ", " mg ", " ml ", " Mg ", " Ml ", "mg ", "ml ",
                                           " gm ", "gm ", " mcg ", "mcg "])


def is_valid_medicine_in_input(user_input):
    """Check if the user input contains any valid medicine name."""
    return any(med in user_input.lower() for med in KNOWN_MEDICINES)


if st.session_state.selected_service == "Kendr Locator":
    st.title("MedBot")

    user_input = st.text_input("Your Queries:", key='temp_user_input')

    if st.button("Ask Away"):
        if user_input:
            if contains_profanity(user_input):
                st.markdown("<span style='color: red;'>Mind The Language Dear!</span>", unsafe_allow_html=True)

            else:
                formatted_prompt = template.format(user_input, "")

                response = llm.invoke(formatted_prompt)

                if contains_medicine_terms(response):
                    # Generated response has medical terms
                    if is_valid_medicine_in_input(user_input):
                        truncated_response = truncate_at_full_stop(response)
                        st.markdown(f"**MedBot:** {truncated_response}", unsafe_allow_html=False)
                    else:
                        st.markdown("<span style='color: green;'>"
                                    "Please consult to a Pharmacist at you nearest Janaushadi Kendr""<br>"
                                    "Use Kendr Locator to locate one near you!"
                                    "</span>", unsafe_allow_html=True)

                else:
                    # No medicine-related terms, safe to display response
                    truncated_response = truncate_at_full_stop(response)
                    st.markdown(f"**MedBot:** {truncated_response}", unsafe_allow_html=False)

st.warning("Developer's notice : Responses are generated by AI and maybe inaccurate or inappropriate."
           "Any received medical or financial consult is not a substitute for professional advice.")
