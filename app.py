from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from vector_store import build_vector_store
from transcriber import transcribe_all
from audio_processor import process_input
from summarizer import generate_summary, generate_title
from extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from rag_engine import build_rag_chain, ask_question

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎥",
    layout="wide",
)

st.markdown("""
<style>
.block-container{padding-top:2rem;}
.main-title{
font-size:2.5rem;font-weight:700;color:#4F8BF9;
}
.small{color:#888;}
</style>
""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result=None
if "messages" not in st.session_state:
    st.session_state.messages=[]

st.sidebar.title("🎥 AI Meeting Assistant")
st.sidebar.success("Ready")

st.markdown('<div class="main-title">🎥 AI Meeting Assistant</div>', unsafe_allow_html=True)
st.caption("Summarize meetings, extract insights and chat with your transcript.")

source = st.text_input("YouTube URL or local audio path")
language = st.selectbox("Language",["english","hinglish"])

def run_pipeline(src,lang):
    progress=st.progress(0)
    status=st.empty()

    status.info("Processing audio...")
    progress.progress(10)
    chunks=process_input(src)

    status.info("Transcribing...")
    progress.progress(35)
    transcript=transcribe_all(chunks,language=lang)

    status.info("Building vector database...")
    progress.progress(55)
    build_vector_store(transcript)

    status.info("Generating summary...")
    progress.progress(70)
    summary=generate_summary(transcript)

    status.info("Generating title...")
    progress.progress(80)
    title=generate_title(summary)

    status.info("Extracting action items...")
    progress.progress(87)
    actions=extract_action_items(transcript)

    status.info("Extracting decisions...")
    progress.progress(92)
    decisions=extract_key_decisions(transcript)

    status.info("Extracting questions...")
    progress.progress(97)
    questions=extract_questions(transcript)

    rag=build_rag_chain()

    progress.progress(100)
    status.success("Pipeline completed successfully!")

    return {
        "title":title,
        "summary":summary,
        "actions":actions,
        "decisions":decisions,
        "questions":questions,
        "rag_chain":rag,
        "transcript":transcript
    }

if st.button("🚀 Analyze Meeting",use_container_width=True):
    if not source.strip():
        st.error("Enter a YouTube URL or local file path.")
    else:
        with st.spinner("Running AI pipeline..."):
            st.session_state.result=run_pipeline(source.strip(),language)
            st.session_state.messages=[]

if st.session_state.result:
    r=st.session_state.result

    c1,c2,c3=st.columns(3)
    c1.metric("Status","Completed")
    c2.metric("Language",language.title())
    c3.metric("Knowledge Base","Ready")

    st.header(r["title"])

    t1,t2,t3,t4=st.tabs([
        "📝 Summary",
        "✅ Action Items",
        "📌 Decisions",
        "❓ Questions"
    ])

    with t1:
        st.write(r["summary"])

    with t2:
        st.write(r["actions"])

    with t3:
        st.write(r["decisions"])

    with t4:
        st.write(r["questions"])

    with st.expander("Transcript"):
        st.write(r["transcript"])

    st.divider()
    st.subheader("💬 Chat with your meeting")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt=st.chat_input("Ask anything about the meeting...")

    if prompt:
        st.session_state.messages.append(
            {"role":"user","content":prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans=ask_question(r["rag_chain"],prompt)
                st.markdown(ans)

        st.session_state.messages.append(
            {"role":"assistant","content":ans}
        )
