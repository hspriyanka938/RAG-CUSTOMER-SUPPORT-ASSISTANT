🤖 RAG-Based Customer Support Assistant
This project is a Retrieval-Augmented Generation (RAG) system designed to build an intelligent customer support assistant. It uses a PDF-based knowledge base to retrieve relevant information and generate context-aware responses using a Large Language Model (Google Gemini).

The system is enhanced with LangGraph for workflow orchestration and includes a Human-in-the-Loop (HITL) mechanism for handling low-confidence queries.

🎯 Project Objective
The main objective of this project is to design and implement a scalable AI-based customer support system that:

Retrieves relevant information from a PDF knowledge base
Generates intelligent responses using an LLM
Uses workflow control for decision-making
Includes human intervention for uncertain cases(HITL)
Provides API documentation using Swagger
Offers a user-friendly interface
⭐ Key Features
📄 PDF-based knowledge ingestion
🔍 Semantic search using embeddings
🗂 Vector storage using ChromaDB
🤖 Response generation using Google Gemini
🔁 LangGraph-based workflow orchestration
🤝 Human-in-the-Loop (HITL) escalation system
🌐 Interactive UI (Frontend)
📘 Swagger API documentation for testing endpoints
Setup Instructions
1. Environment Setup
Create a Virtual Environment:
python -m venv venv
Activate the Virtual Environment:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate
Install Dependencies:
pip install -r requirements.txt
2. API Keys
Create a file named .env in the root directory (d:\innomatics\Rag-Project\.env) and add your Google API key:

GOOGLE_API_KEY=your-google-api-key-here
(Note: You can replace the Gemini models with other alternatives in app.py if needed).

3. Running the Application
Ingest a Document (Optional but Recommended):

Place a sample PDF in your project directory (e.g., sample_manual.pdf).
Open app.py, scroll to the bottom (if __name__ == "__main__":), uncomment the pdf_path and setup_vector_store() lines, and run it once to build the ChromaDB index.
Execute the LangGraph Workflow:

python app.py
4. How the Human-in-the-Loop (HITL) Simulation Works
When you run app.py, it simulates a workflow:

It asks a query: "How do I reset my password?"
If the vector DB doesn't have the answer, the LLM will grade itself with a 0.0 confidence score.
The LangGraph router intercepts this low score and pauses execution.
The script will prompt you in the terminal: Human Agent, provide the answer:
Type your response and hit enter. The graph resumes and delivers your human answer as the final output.
📘 Swagger API Documentation
After running the project, open:

http://localhost:8000/docs
👉 This provides interactive API testing for all backend endpoints.

🎥 Project Demo
Input: User query is asked in terminal
Output: Contextual answer from RAG system
HITL: Activated when confidence is low
👉 Example: User: What is refund policy? Bot: Refunds are allowed within 7 days...

🏗 System Architecture
image
🛠 Tech Stack
Python
LangChain
LangGraph
ChromaDB
Google Gemini API
PyPDF
FastAPI
Swagger UI
HTML, CSS, JavaScript
🎨 UI Features
Interactive chat interface
Real-time query-response system
Integrated backend communication
Improved user experience over CLI version
🚀 Future Enhancements
Multi-document knowledge base Authentication system Cloud deployment (AWS / GCP) Advanced intent classification Chat history memory

📸 Screenshots
image
Console version outputs
🧪 Query Response Example
Shows how the system retrieves relevant information from the PDF knowledge base and generates a response using Gemini LLM.
image
image
image
🤝 Human-in-the-Loop (HITL) Activation Example
Shows the system triggering HITL when confidence is low and asking for human intervention.
image
UI version outputs
Dashboard
image
🧪 Query Response Example
image
image
🤝 Human-in-the-Loop (HITL) Activation Example
image
Fast API SWAGGER Output
image
🎥 Demo Video
👉 Watch Full Demo Here: https://drive.google.com/drive/folders/1mV1dEcCnXME8kCKfzgECmMC3luI1k5d1?usp=drive_link

🔗 LinkedIn Post
https://www.linkedin.com/posts/tejasri-somarouthu-78aa83355_generativeai-artificialintelligence-rag-ugcPost-7453901211042480128-6qQC?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFidKJoBRLvumfdgwfmIDItuHoJpdXkBZuI

📝 Note
Due to ongoing semester examinations, this project was completed within limited time. A refined version with improved UI and explanation will be updated soon. The presentation PPT was created using the AI tool Gamma.

📌 Outcome
A fully functional AI-powered customer support system combining retrieval-based search with generative AI, making responses more accurate, scalable, and production-ready.

🙏 Acknowledgement
This project was developed as part of my Generative AI Internship at Innomatics Research Labs. It provided hands-on experience in building real-world AI systems using RAG architecture.

👨‍💻Author
Priyanka H S - IN126048902(Gen-AI intern)
