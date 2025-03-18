File Instructions
1. rag2_final.py
This file contains essential functions for preprocessing and initializing the LLM model.
Do not execute this file directly.
Ensure it is stored in the same folder as the other project files.
2. app_final.py
This is the main script to execute in the terminal to launch the application.
To run it in VS Code terminal or any command line, use:
-streamlit run app_final.py
3. demo_2.ipynb
A Jupyter Notebook that allows you to explore each step of the process interactively.
Useful for debugging and understanding the workflow.
4. ideal_result.png
A screenshot of the expected Streamlit page layout.
Also includes an example answer to a query for reference.
5. Article_93.pdf
A test PDF used for validation.
In the Streamlit app, you can upload your own PDFs for processing.


Setup Instruction
1. Create a Virtual Environment
Before executing the script, create a virtual environment to manage dependencies and  install the necessary dependencies:
-pip install langchain langchain-community langchainhub chromadb faiss-cpu ollama pdfplumber streamlit
2,Install and Setup Ollama
-Make sure you have Ollama installed locally. If not, install it from: https://ollama.com
Once installed, download the llama3 model by running:ollama pull llama3



Future Improvements
When processing large PDFs (e.g., 500+ pages), we could  optimize retrieval by:
Splitting the document into chunks.
Calculating similarity between the user's query and stored document vectors.
Fetching only the top n most relevant chunks based on similarity score (n being adjustable as per user needs)
