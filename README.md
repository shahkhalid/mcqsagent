# 🧠 MCQsAgent: Towards Human-Level MCQs Generation via Collaborative Multi-Agent AI framework

## 📖 Introduction

This repository presents **MCQsAgent**, a robust **Streamlit-based application** designed for the automated generation of high-quality **Multiple-Choice Questions (MCQs)**. Leveraging advanced **Large Language Models (LLMs)** from Google Gemini, Groq, and Mistral AI, MCQsAgent implements a multi-stage pipeline to produce contextually relevant, grammatically correct, and appropriately challenging MCQs.

This framework provides a practical and efficient solution for content creators, educators, and developers looking to automate and scale their assessment and learning material generation processes.

---

## 🎯 Purpose & Use Cases

**MCQsAgent** was developed to streamline the creation of diverse and effective multiple-choice questions — a task that traditionally requires significant manual effort and expertise.

By automating this process, MCQsAgent helps:

- **Content Developers**: Rapidly generate assessment materials for e-learning platforms, online courses, and educational content.
- **Educators**: Create quizzes and tests tailored to specific topics and difficulty levels, saving valuable time.
- **Developers**: Integrate intelligent question generation capabilities into their applications and platforms.
- **Businesses**: Develop training modules and knowledge checks efficiently.

This project demonstrates a practical application of advanced LLMs in automating complex content generation workflows, ensuring high standards of quality and relevance.

---

## 💡 Project Overview & Methodology

**MCQsAgent** employs a sophisticated, multi-step pipeline to ensure the generation of high-quality MCQs. Each step is orchestrated by calls to various **Large Language Models (LLMs)**, with intermediate processing and validation to refine the output.

### The Pipeline Comprises the Following Stages:

#### 1. **Question-Answer (QA) Pair Generation**

- **Objective**: Generate an initial set of concise question-answer pairs based on a user-defined topic and desired quantity.
- **LLM Interaction**: Prompt guides the LLM (e.g., Google Gemini 2.0 Flash) to create clear, unambiguous questions suitable for MCQs, with short and precise answers.
- **Constraints**: Avoid True/False questions, limit answer length, align with difficulty level.
- **Output Format**: CSV-like string:
  ```
  "Question","Answer"
  ```

---

#### 2. **QA Pair Verification**

- **Objective**: Validate QA pairs against criteria like clarity, conciseness, relevance, grammar, and accuracy.
- **LLM Interaction**: Groq evaluates each pair and returns a binary status ("0" = fail, "1" = pass).
- **Post-processing**: Pairs marked "0" are filtered out.
- **Output Format**: CSV-like string:
  ```
  "QA_Pair_Number","Status"
  ```

---

#### 3. **Distractor Generation**

- **Objective**: Generate 10 plausible but incorrect distractors per verified QA pair.
- **LLM Interaction**: Mistral AI generates diverse, conceptually related, subtly incorrect options.
- **Output Format**:
  ```json
  [["1", ["d1", ..., "d10"]], ...]
  ```

---

#### 4. **Distractor Selection**

- **Objective**: Select top 3 most suitable distractors per QA pair.
- **LLM Interaction**: Gemini reviews all 10 distractors and selects the best 3 based on plausibility, distinctiveness, and relevance.
- **Output Format**:
  ```json
  [["1", ["d1", "d2", "d3"]], ...]
  ```

---

#### 5. **MCQ Compilation**

- **Objective**: Combine verified QA pairs with selected distractors to form complete MCQ structures.
- **Logic**: Correct answer + 3 distractors → shuffled order (A–D), identify correct option letter.

---

#### 6. **MCQ Optimization**

- **Objective**: Final review and optimization of compiled MCQs.
- **LLM Interaction**: Gemini refines questions, options, and distractors, correcting inconsistencies or errors.
- **Structured Response**: Output formatted as JSON list of MCQ objects using Gemini’s schema support.

---

## ✨ Features

- **Customizable Generation**: Specify topic, number of questions, and difficulty level.
- **Robust Pipeline**: Multi-stage generation, verification, and refinement for high-quality output.
- **Multi-LLM Powered**: Leverages Google Gemini, Groq, and Mistral AI for diverse and intelligent content creation.
- **Structured Output**: Generates MCQs in a clear, readable format with downloadable CSV.
- **Secure API Handling**: Uses environment variables for API key management.

---

## 🚀 Getting Started

Follow these steps to set up and run the **MCQsAgent** application locally.

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mcqs-agent.git
cd mcqs-agent
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔐 API Key Configuration

This app requires valid API keys from **Google Gemini**, **Groq**, and **Mistral AI**. These should be stored securely using Streamlit's `secrets.toml`.

### Steps:

1. In your project root directory, create a folder named `.streamlit`.
2. Inside it, create a file called `secrets.toml`.
3. Add your keys like this:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
MISTRAL_API_KEY = "your_mistral_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

> ⚠️ Never share or commit this file publicly. Add it to your `.gitignore`.

### How to Get Your Keys:

| Service | Link |
|--------|------|
| **Google Gemini** | [Get API Key](https://aistudio.google.com/app/apikey) |
| **Groq** | [Manage API Keys](https://console.groq.com/keys) |
| **Mistral AI** | [API Keys Console](https://console.mistral.ai/api-keys/) |

> You must have appropriate access permissions (e.g., developer role) to manage API keys.

---

### 5. Run the Application

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

---

## 📂 Project Structure

```
mcqs-agent/
├── .env.example         # Example file for environment variables (copy to .env)
├── app.py               # Main Streamlit application file
├── requirements.txt     # Lists all Python dependencies
└── README.md            # Project documentation (this file)
```

For larger-scale development or more complex systems, modularization could include:

- `api_wrappers.py`: Handles direct interactions with LLM APIs.
- `utils.py`: Contains helper functions for parsing and merging data.
- `mcq_pipeline.py`: Encapsulates core multi-step logic.

---

## 🚀 Future Enhancements

We are continuously working to improve **MCQsAgent**. Potential future enhancements include:

- **Adaptive Difficulty**: Dynamic adjustment based on user performance or learning profiles.
- **Multimodal Input**: Generate MCQs from images, videos, or audio.
- **User Feedback Integration**: Fine-tune LLM behavior via human feedback.
- **Domain Specialization**: Accurate content generation for niche educational domains.
- **Performance Optimization**: Improve speed and resource usage.

---

## 🤝 Contributing

We welcome contributions to **MCQsAgent**! If you have suggestions for improvements, new features, or encounter any issues, please feel free to:

- Open an issue on this GitHub repository.
- Submit a pull request with your proposed changes.

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

## 📧 Contact

For any inquiries or collaborations, please contact:

**Muhammad Raheel Anwar - mranwar.mscs22seecs@seecs.edu.pk**

**Shah Khalid - shah.khalid@seecs.edu.pk**


## 📚 Cite This Paper

If you use **MCQsAgent** in your research or academic work, please consider citing it using the following format:

### 🔹 BibTeX:

```bibtex
@misc{mcqsagent,
  author       = {Muhammad Raheel Anwar, Shah Khalid},
  title        = {{MCQsAgent: An AI-Powered Framework for Automated Multiple-Choice Question Generation}},
  year         = {2025},
  note         = {\texttt{https://github.com/shahkhalid/mcqsagent}},
  howpublished = {\url{https://github.com/shahkhalid/mcqsagent}}
}
```

### 🔹 APA Style:

> Muhammad Raheel Anwar, Shah Khalid (2025). *MCQsAgent: Towards Human-Level MCQs Generation via Collaborative Multi-Agent AI framework*, Educational technology research and development. https://github.com/shahkhalid/mcqsagent
