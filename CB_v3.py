import streamlit as st  # Import after set_page_config
from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Conference Chatbot", page_icon="🤖")

# Load SBERT model for FAQ Matching (Fast Encoding)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Expanded FAQ Data
faqData = {
    # Submission and Paper Details
    "submission deadline": "The abstract submission deadline is July 1st, 2025, and the full paper deadline is August 1st, 2025.",
    "last date to submit paper": "The abstract submission deadline is July 1st, 2025, and the full paper deadline is August 1st, 2025.",
    "word limit": "The word limit for research papers is 6,000 words.",
    "citation format": "Follow the latest APA standards.",
    "submission fees": "No, there are no submission fees.",
    "review process": "Papers go through a double-blind peer review process.",
    "is the journal indexed": "Yes, our journal is indexed in Scopus and Web of Science.",
    
    # Conference Information
    "conference dates": "The event will be held from September 24-26, 2025.",
    "conference location": "The conference will be held in New York City.",
    "conference schedule": "The detailed conference schedule will be published on our website by August 15, 2025.",
    
    # Registration and Participation
    "registration process": "Register online through the conference website.",
    "is registration mandatory": "Yes, all participants must register to attend the conference.",
    "registration fees": "The registration fee is $200 for early bird and $300 for regular registration.",
    "is there a discount for students": "Yes, students get a 50% discount on registration fees.",
    "can i register on-site": "Yes, but on-site registration has a higher fee and is subject to availability.",
    
    # Presentation Formats
    "presentation format": "Presentations can be oral or poster format.",
    "how long is a paper presentation": "Each paper presentation is allocated 15 minutes, followed by a 5-minute Q&A session.",
    "how long is a poster presentation": "Poster presentations will be displayed throughout the conference with a dedicated Q&A session.",
    
    # Logistics and Travel
    "travel grants": "Limited travel grants are available for selected participants. Please check the website for eligibility.",
    "accommodation details": "We have partnered with nearby hotels. Discounted rates are available for registered participants.",
    "will meals be provided": "Yes, breakfast, lunch, and coffee breaks are included in the registration fee.",
    "is there an airport shuttle service": "Yes, shuttle services from the airport to the venue will be available for registered participants.",
    
    # Special Requests
    "can i request a visa invitation letter": "Yes, we provide visa invitation letters for registered international participants.",
    "is there a refund policy": "Yes, cancellations made before August 15th will receive a 50% refund. No refunds after this date.",
    "will there be virtual participation options": "Yes, there will be a hybrid option with virtual presentations and live streaming.",
    "can i change my presentation mode": "Requests for changing presentation mode should be made by August 1st.",
    
    # Miscellaneous
    "who are the keynote speakers": "The keynote speakers will be announced on our website by June 15, 2025.",
    "are there networking opportunities": "Yes, we have networking sessions and social events planned for attendees.",
    "how can i become a session chair": "Session chairs are selected based on expertise. You can apply via the conference website.",
    "how can i sponsor the conference": "Organizations interested in sponsorship can contact our sponsorship team through the website.",
    "will there be workshops": "Yes, we have pre-conference workshops on AI, data science, and research methodologies.",
    "is there a best paper award": "Yes, the best paper award will be given based on quality, originality, and impact.",
    "will conference proceedings be published": "Yes, all accepted papers will be published in the conference proceedings.",
    "who do i contact for more details": "For more information, email us at support@conference2025.com."
}

# Encode FAQ questions for fast matching
faq_questions = list(set(faqData.keys()))
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Cache user queries for faster retrieval
@lru_cache(maxsize=100)
def get_user_embedding(question):
    return model.encode(question, convert_to_tensor=True)

# Custom CSS for better UI
st.markdown("""
    <style>
        .chat-container {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            text-align: right;
            color: white;
            background-color: #0084ff;
            padding: 8px;
            border-radius: 10px;
            display: inline-block;
        }
        .bot-message {
            text-align: left;
            color: black;
            background-color: #e5e5ea;
            padding: 8px;
            border-radius: 10px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 JoRIE Chatbot")
st.markdown("Ask me anything about the **conference**, **submission process**, **registration**, or **participation details**.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.container():
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-container"><div class="user-message">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container"><div class="bot-message">{msg["content"]}</div></div>', unsafe_allow_html=True)

# User input field with "Enter" key functionality
user_input = st.chat_input("Type your question...")  # This enables ENTER key functionality

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Encode User Input
    user_embedding = get_user_embedding(user_input)

    # Find best match
    scores = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    best_match_idx = scores.argmax().item()
    best_match_score = scores[best_match_idx].item()

    # Get Response
    if best_match_score > 0.5:
        bot_reply = faqData[faq_questions[best_match_idx]]
    else:
        bot_reply = "I'm sorry, I don't have an answer for that. Please check the official guidelines."

    # Store Response
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Refresh UI
    st.rerun()
