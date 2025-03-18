from rag2 import get_chain
from rag2 import process_pdf
import streamlit as st



def answer_question(user_question):
    answer =chain.invoke(user_question)
    return answer

# Streamlit user interface
st.title("📄 Ask me question about your pdf")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
        with open("test.pdf", "wb") as f:
            f.write(uploaded_file.read())
        chunks=process_pdf("test.pdf")
        chain=get_chain(chunks)
        st.write("Ask a question, and the system will retrieve relevant information and generate an answer!")

        # Input field for the user to type a question
        user_question = st.text_input("Your Question:")

        if st.button("Get Answer"):
            if user_question:
                # Call the function to get the answer
                answer = answer_question(user_question)
                st.write(f"Answer: {answer}")
            else:
                st.write("Please enter a question.")





