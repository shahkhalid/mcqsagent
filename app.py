import streamlit as st
import google.generativeai as gemini
from mistralai import Mistral
from groq import Groq
from typing import Optional, List, Tuple
import concurrent.futures
import re, random
import ast
import json
from pydantic import BaseModel, TypeAdapter
from collections import defaultdict

# ====== Client Initialization ======
try:
    # Attempt to retrieve API keys from Streamlit secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
    gemini_api_key = st.secrets["GEMINI_API_KEY"]

    groq_client = Groq(api_key=groq_api_key)
    mistral_client = Mistral(api_key=mistral_api_key)
    gemini.configure(api_key=gemini_api_key) # Configure Gemini globally

except KeyError as e:
    st.error(f"Missing API key in Streamlit secrets: {e}. Please ensure GROQ_API_KEY, MISTRAL_API_KEY, and GEMINI_API_KEY are set.")
    st.stop() # Stop the app if initialization fails
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# ====== API Wrappers ======
def call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> Optional[str]:
    """Call Gemini API with prompt and model name"""
    try:
        model_instance = gemini.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text.strip() if response.text else None
    except Exception as e:
        st.error(f"Gemini error with model {model}: {e}")
        return None

def call_mistral(prompt: str, model: str = "open-mistral-7b") -> Optional[str]:
    """Call Mistral API with prompt and model name"""
    try:
        response = mistral_client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        st.error(f"Mistral error with model {model}: {e}")
        return None

def call_groq(prompt: str, model: str = "mixtral-8x7b-32768") -> Optional[str]:
    """Call Groq API with prompt and model name"""
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        st.error(f"Groq error with model {model}: {e}")
        return None

# Helper functions from original script (no changes needed here based on the request)
def merge_results(input_strings):
    """Merges a list of CSV-like strings, outputting a merged string.
    If all corresponding values are "1", the output is "1". Otherwise, it's "0".
    """
    data_dicts = []
    for s in input_strings:
        data = {}
        lines = s.strip().split('\n')
        for line in lines:
            match = re.match(r'^"([^"]+)","([^"]+)"$', line.strip())
            if match:
                key, value = match.groups()
                data[key] = value
            else:
                continue
        data_dicts.append(data)

    merged_data = {}
    all_keys = set()
    for d in data_dicts:
        all_keys.update(d.keys())

    output_string = ""
    for key in sorted(all_keys):
        all_ones = True
        for d in data_dicts:
            if key not in d or d[key] != "1":
                all_ones = False
                break
        merged_data[key] = "1" if all_ones else "0"
        output_string += f'"{key}","{merged_data[key]}"\n'
    return output_string.strip()

def parse_qa(input_string):
    """Parses a string of question-answer pairs into a numbered list."""
    qa_pairs = input_string.strip().split('\n')
    output = []

    for i, pair in enumerate(qa_pairs):
        parts = re.split(r'",\s*"', pair.strip('"'))
        if len(parts) == 2:
            question, answer = parts
            output.append([f'''"{i+1}"''', f'''"{question}"''', f'''"{answer}"'''])
        else:
            continue
    return output

def remove_zero_entries(parsed_data, zero_values_data):
    """Removes entries from parsed_data based on '0' values in zero_values_data."""
    filtered_data = []
    merged_zero_values = merge_results(zero_values_data)
    lines = merged_zero_values.strip().split('\n')

    verification_status = {}
    for line in lines:
        parts = line.split('","')
        if len(parts) == 2:
            number = parts[0].strip('"')
            status = parts[1].strip('"')
            verification_status[number] = status

    for entry in parsed_data:
        qa_number = entry[0].strip('"')
        if verification_status.get(qa_number) != "0":
            filtered_data.append(entry)
    return filtered_data

def parse_and_filter_distractors_list(input_string):
    """Parses a string of distractors into a list."""
    lines = input_string.strip().split('\n')
    distractors = [line.strip('"') for line in lines]
    return distractors

def extract_list_within_balanced_brackets(input_string):
    """
    Extracts the list within the outermost balanced square brackets.
    Handles incomplete or unbalanced structures and returns a list.
    """
    start_index = input_string.find("[")
    if start_index == -1:
        return None

    bracket_count = 0
    end_index = -1
    for i in range(start_index, len(input_string)):
        if input_string[i] == "[":
            bracket_count += 1
        elif input_string[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end_index = i
                break

    if end_index == -1:
        return None

    extracted_string = input_string[start_index : end_index+1]

    try:
        extracted_list = ast.literal_eval(extracted_string)
        return extracted_list
    except (SyntaxError, ValueError) as e:
        st.error(f"Error evaluating string as a list: {e}")
        return None

def parse_input(input_data):
    """Function to extract and parse the first valid list from each input string"""
    parsed_data = []
    for item in input_data:
        match = re.search(r"\[.*\]", item, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                parsed_list = ast.literal_eval(list_str)
                if isinstance(parsed_list, list):
                    parsed_data.append(parsed_list)
            except (SyntaxError, ValueError):
                continue
    return parsed_data

def merge_and_select(parsed_data):
    """Function to merge and select items"""
    merged = defaultdict(list)
    for data_list in parsed_data:
        for category, items in data_list:
            merged[category].extend(items)

    result = []
    for category, items in merged.items():
        frequency = {}
        for item in items:
            frequency[item] = frequency.get(item, 0) + 1
        sorted_items = sorted(frequency.keys(), key=lambda x: (-frequency[x], x))
        selected_items = sorted_items[:3]
        result.append([category, selected_items])

    return result

def parse_inputs(qa_pairs, distractors):
    """Function to parse QA pairs and distractors"""
    parsed_qa = {}
    for item in qa_pairs:
        category = item[0].strip('"')
        question = item[1].strip('"')
        correct_answer = item[2].strip('"')
        parsed_qa[category] = {"question": question, "correct_answer": correct_answer}

    parsed_distractors = {category: items for category, items in distractors}

    return parsed_qa, parsed_distractors

def merge_mcqs(parsed_qa, parsed_distractors):
    """Function to merge QA pairs with distractors"""
    final_mcqs = []

    for category, qa in parsed_qa.items():
        question = qa["question"]
        correct_answer = qa["correct_answer"]
        options = parsed_distractors.get(category, [])

        options_for_mcq = list(options)
        options_for_mcq.append(correct_answer)
        random.shuffle(options_for_mcq)

        while len(options_for_mcq) < 4:
            options_for_mcq.append(correct_answer)
            random.shuffle(options_for_mcq)

        options_for_mcq = options_for_mcq[:4]

        try:
            correct_option_letter = chr(65 + options_for_mcq.index(correct_answer))
        except ValueError:
            st.warning(f"Correct answer '{correct_answer}' not found in options for category {category}. Setting correct option to A.")
            correct_option_letter = 'A'

        final_mcqs.append({
            "category": category,
            "question": question,
            "options": options_for_mcq,
            "correct_option": correct_option_letter
        })

    return final_mcqs

# MCQs Generation Pipeline Implementations

def generate_qa_pairs(topic: str, count: int, difficulty: str):
    prompt = f'''
    Generate {count} Question and Answer pairs on the topic of "{topic}" following these guidelines:
    1. Maintain a difficulty level of {difficulty}.
    2. Ensure questions are clear, concise, and free from ambiguity or hints.
    3. Avoid True/False or Yes/No questions entirely.
    4. Avoid questions with lengthy answers or more than two items in the answer. For example, do not generate questions like:
        - "What are the four main phases of mitosis?","Prophase, Metaphase, Anaphase, Telophase"
    5. Ensure answers are short, precise, and directly related to the question.
    6. Focus on single-concept questions to maintain clarity and simplicity.
    7. Avoid overly complex phrasing or technical jargon unless necessary for the difficulty level.
    8. Ensure questions are well-suited for multiple-choice formats (MCQs) by avoiding open-ended or subjective questions.
    9. Verify that each question has only one correct and unambiguous answer.
    10. Align questions with Bloom's Taxonomy levels based on the difficulty:
            - Remembering (Basic): Focus on recall of facts, terms, or concepts. Example: "What is the capital of France?"
            - MCQ Example: "What is the chemical symbol for Gold? A) Ag B) Au C) Fe D) Pb"
            - Understanding (Intermediate): Test comprehension of ideas or concepts. Example: "Which of the following best describes Newton's First Law?"
            - MCQ Example: "Which law states that an object in motion stays in motion unless acted upon by an external force? A) Newton's First Law B) Newton's Second Law C) Newton's Third Law D) Law of Gravitation"
            - Applying (Intermediate): Use knowledge in new situations. Example: "If a force of 10N is applied to a 2kg object, what is its acceleration?"
            - MCQ Example: "What is the acceleration of a 5kg object when a 20N force is applied? A) 2 m/s² B) 4 m/s² C) 5 m/s² D) 10 m/s²"
            - Analyzing (Advanced): Break down information into parts and explore relationships. Example: "Which of the following processes is responsible for genetic diversity in meiosis?"
            - MCQ Example: "Which process during meiosis contributes to genetic variation? A) Crossing over B) DNA replication C) Cytokinesis D) Mitosis"
            - Evaluating (Advanced): Make judgments based on criteria and standards. Example: "Which theory best explains the observed phenomenon, and why?"
            - MCQ Example: "Which theory best explains the behavior of gases at high temperatures? A) Kinetic Theory B) Boyle's Law C) Charles's Law D) Ideal Gas Law"
            - Creating (Advanced): Produce new or original work. Example: "Design an experiment to test the effect of light on plant growth."
            - MCQ Example: "Which experimental setup would best test the effect of light intensity on plant growth? A) Varying light intensity with constant temperature B) Varying temperature with constant light intensity C) Using different soil types with constant light D) Changing water levels with constant light"
    11. Do not include any extra text, explanations, or deviations from the specified format.

    Output Format:
    "What is the chemical symbol for Gold?","Au"
    "Which law states that force equals mass times acceleration?","Newton's Second Law"

    Do not include any additional text or commentary in the output.'''

    response = call_gemini(prompt, "gemini-1.5-flash") # Changed to gemini-1.5-flash as per original reference
    return response if response else None


def verify_qa_pairs(qa_csv, count, difficulty):
    verification_prompt = f'''
        Verify the following {count} QA pairs with dificulty level of {difficulty} and output 0 or 1 for each pair based on the criteria below:
    {qa_csv}

    Verification Criteria:
    1. Clarity: The question must be clear, unambiguous, and precise.
    2. Conciseness: Avoid wordiness while retaining full meaning.
    3. Relevance: Ensure the question focuses on a meaningful concept.
    4. Grammatical Accuracy: No grammar or syntax errors.
    5. MCQ Suitability: The question must work well in a multiple-choice format.
    6. Answer Accuracy: The provided answer must be objectively correct.

    Output Rules:
    1. Use the format: `"<QA Pair Number>","<0 or 1>"`.
    2. Ensure each QA pair is evaluated only once.
    3. Do not repeat the same QA pair number in the output.
    4. Do not skip double quotes or add extra characters like `\n`.
    5. Follow the exact output format without deviations.
    6. Do not add any additional text, explanations, or deviations from the specified format.

    Output Format:
    "1","1"
    "2","1"
    "3","0"

    Important Note: Do not include any additional text like "Here is the output:\n", explanations, or deviations from the specified format.
    '''

    models = [
        ("gemini", "gemini-2.0-flash"),
        ("groq", "llama3-70b-8192"),
        ("mistral", "open-mistral-nemo")
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for service, model in models:
            if service == "gemini":
                futures.append(executor.submit(call_gemini, verification_prompt, model))
            elif service == "groq":
                futures.append(executor.submit(call_groq, verification_prompt, model))
            elif service == "mistral":
                futures.append(executor.submit(call_mistral, verification_prompt, model))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"Verification error: {e}")

    return results if results else None

def generate_distractors(qa_csv: str, difficulty: str):
    distractors_prompt = f'''
    Generate 10 high-quality plausible but incorrect distractors with difficulty level of {difficulty} for the given list of QA pairs (number, question, answer) following provided guidelines:
    QA Pairs:
    {qa_csv}

    Guidelines:
    1. Each option should seem like a valid answer.
    2. Conceptually related: distractors must belong to the same subject.
    3. Diverse and distinct: distractors should be plausible but incorrect and fully different, not just rewordings.
    4. No obvious errors: avoid illogical, humorous, or irrelevant choices, and all options should be phrased similarly.
    5. Not easily predictable: distractors should match the format and complexity of the correct answer.
    6. Conceptually close but subtly wrong: ensures test-takers engage critically.
    7. Vary numerical distractors realistically: numbers should be near the correct one but incorrect.

    Output Format for 3 QA Pairs:
    [["1", ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]], ["2", ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]],["3", ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]]]

    Do not skip double quotes, and follow the format strictly.
    This format is not json, this format is list of list, so avoid any additional text.
    The output must starts form [ and ends with ]
    Do not include any additional text, explanations, or deviations from the specified format.
    '''

    response = call_gemini(distractors_prompt, "gemini-2.0-flash-thinking-exp-01-21") # Using gemini-2.0-flash-thinking-exp-01-21 as per original reference
    return response if response else None

def get_final_distractors(qa_csv: str, distractors_csv, difficulty: str):
    final_distractors_prompt = f'''
    Based on the provided QA Pairs, select the best high-quality distractors from the list of distractors considering the dificulty level of {difficulty} given below:
    QA Pairs:
    {qa_csv}

    Distractors:
    {distractors_csv}

    Output Rules:
    1. Select 3 distractors for each QA pair.
    2. Ensure distractors are conceptually related, plausible, and distinct.
    3. Avoid distractors that are repetitive, irrelevant, or obviously incorrect.
    4. Follow the exact output format without deviations.

    Output Format:
    [["1", ["d1", "d2", "d3"]], ["2", ["d1", "d2", "d3"]], ["3", ["d1", "d2", "d3"]]]

    Do not skip double quotes, and follow the format strictly.
    This format is a list of lists, not JSON, so avoid any additional text.
    The output must start with `[` and end with `]`.
    Do not include any additional text, explanations, or deviations from the specified format.
    '''
    models = [
        ("gemini", "gemini-1.5-flash"),
        ("groq", "llama3-70b-8192"),
        ("mistral", "open-mistral-nemo")
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for service, model in models:
            if service == "gemini":
                futures.append(executor.submit(call_gemini, final_distractors_prompt, model))
            elif service == "groq":
                futures.append(executor.submit(call_groq, final_distractors_prompt, model))
            elif service == "mistral":
                futures.append(executor.submit(call_mistral, final_distractors_prompt, model))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"Distractor selection error: {e}")

    parsed_data = parse_input(results)
    final_result = merge_and_select(parsed_data)
    return final_result if final_result else None

class MCQs(BaseModel):
    category: str
    question: str
    options: List[str]
    correct_option: str


def optimize_mcqs(all_mcqs: List[dict], difficulty: str) -> Optional[List[MCQs]]:
    """
    Optimizes a list of MCQs using Gemini API with structured response.
    """
    optimization_prompt = f'''
    Review and optimize the following list of MCQs keeping difficulty level of {difficulty} to ensure they meet the criteria below. Output the refined MCQs in a structured JSON format.

    Optimization Criteria:
    1. Clarity: Ensure questions and options are clear, unambiguous, and free from hints.
    2. Alignment: Verify that questions align with their intended Bloom's Taxonomy level (Remembering, Understanding, Applying, Analyzing, Evaluating, Creating).
    3. Distractors: Ensure distractors are plausible, relevant, and distinct.
    4. Format: Correct any formatting issues (e.g., typos, inconsistent phrasing, or grammatical errors).

    Input MCQs:
    {json.dumps(all_mcqs, indent=2)}

    Output Rules:
    Do not add or remove questions unless they are invalid or redundant.
    Ensure the JSON structure is strictly followed.
    Optimize only where necessary; avoid over-optimization.
    '''

    # Corrected client initialization and API call for structured output
    model_instance = gemini.GenerativeModel(model_name='gemini-2.0-flash')

    # Define the response schema explicitly for a list of MCQs
    mcqs_list_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "correct_option": {"type": "string"}
            },
            "required": ["category", "question", "options", "correct_option"]
        }
    }

    try:
        response = model_instance.generate_content(
            contents=optimization_prompt,
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': mcqs_list_schema,
            },
        )
        # Parse the JSON string from response.text and validate with Pydantic
        generated_json_str = response.text
        parsed_data = json.loads(generated_json_str)
        generated_mcqs: List[MCQs] = TypeAdapter(List[MCQs]).validate_python(parsed_data)
        return generated_mcqs
    except Exception as e:
        st.error(f"MCQ optimization failed: {e}")
        return None


def compile_mcqs(filtered_output: List[List[str]], final_result: List[List[str]]) -> List[dict]:
    """Compiles the final MCQs from filtered QA pairs and selected distractors."""
    parsed_qa, parsed_distractors = parse_inputs(filtered_output, final_result)
    final_mcqs = merge_mcqs(parsed_qa, parsed_distractors)
    return final_mcqs


def generate_mcqs(topic: str, count: int, difficulty: str) -> Optional[List[MCQs]]:
    """
    Orchestrates the MCQ generation pipeline.
    """
    st.subheader("Generation Steps:")

    with st.spinner("Generating QA Pairs..."):
        qa_pairs = generate_qa_pairs(topic, count, difficulty)
        if not qa_pairs:
            st.error("Failed to generate QA pairs. Please try again.")
            return None
        st.success("QA Pairs Generated.")

    with st.spinner("Verifying QA Pairs..."):
        verification_results = verify_qa_pairs(qa_pairs, count, difficulty)
        if not verification_results:
            st.warning("Failed to verify QA pairs or no verification data returned. Proceeding with caution.")
            parsed_qa_pairs = parse_qa(qa_pairs)
            filtered_output = parsed_qa_pairs
        else:
            parsed_qa_pairs = parse_qa(qa_pairs)
            filtered_output = remove_zero_entries(parsed_qa_pairs, verification_results)

        if not filtered_output:
            st.warning("No valid QA pairs remaining after filtering. Adjust parameters or try again.")
            return None
        st.success(f"{len(filtered_output)} QA Pairs Verified and Filtered.")

    with st.spinner("Generating Distractors..."):
        distractors_raw = generate_distractors(filtered_output, difficulty)
        if not distractors_raw:
            st.error("Failed to generate initial distractors. Please try again.")
            return None
        cleaned_distractors = extract_list_within_balanced_brackets(distractors_raw)
        if not cleaned_distractors:
            st.error("Failed to parse distractors. Ensure the LLM output format is correct.")
            return None
        st.success("Distractors Generated.")

    with st.spinner("Selecting Final Distractors..."):
        final_distractors = get_final_distractors(filtered_output, cleaned_distractors, difficulty)
        if not final_distractors:
            st.error("Failed to select final distractors. Please try again.")
            return None
        st.success("Final Distractors Selected.")

    with st.spinner("Compiling MCQs..."):
        all_mcqs = compile_mcqs(filtered_output, final_distractors)
        if not all_mcqs:
            st.error("Failed to compile MCQs. Please try again.")
            return None
        st.success("MCQs Compiled.")

    with st.spinner("Optimizing MCQs..."):
        final_optimized_mcqs = optimize_mcqs(all_mcqs, difficulty)
        if not final_optimized_mcqs:
            st.error("Failed to optimize MCQs. Displaying unoptimized MCQs if available.")
            return None
        st.success("MCQs Optimized!")

    return final_optimized_mcqs

def view_mcqs_expanded(final_mcqs: List[MCQs], topic: str, difficulty: str):
    """Displays MCQs in an expanded, readable format."""
    st.header("Generated MCQs")
    for i, mcq in enumerate(final_mcqs):
        st.markdown(f"---")
        st.subheader(f"Question {i+1} (Category: {mcq.category})")
        st.write(f"**Question**: {mcq.question}")
        st.write("**Options**:")
        for j, option in enumerate(mcq.options):
            st.write(f" {chr(65+j)}. {option}")
        st.write(f"**Correct Option**: {mcq.correct_option}")
        st.write(f"**Difficulty**: {difficulty}")

def view_mcqs_csv(final_mcqs: List[MCQs], domain: str, topic: str, difficulty: str) -> str:
    """Returns MCQs in CSV format as a string."""
    csv_output = ""
    csv_output += '"Domain","Topic","Question","Option A","Option B","Option C","Option D","Correct Option","Difficulty"\n'
    for mcqs_item in final_mcqs:
        options_str = ""
        padded_options = list(mcqs_item.options) + [""] * (4 - len(mcqs_item.options))
        for option in padded_options[:4]:
            options_str += f'"{option.replace('"', '""')}",' # Escape double quotes within options
        options_str = options_str.rstrip(',')

        csv_output += f'''"{domain.replace('"', '""')}","{topic.replace('"', '""')}","{mcqs_item.question.replace('"', '""')}",{options_str},"{mcqs_item.correct_option.replace('"', '""')}","{difficulty.replace('"', '""')}"\n'''
    return csv_output

# Streamlit UI
st.set_page_config(page_title="MCQ Generator", layout="centered")

st.title("🧠 MCQsAgent")
st.markdown("Generate multiple-choice questions on any topic with custom difficulty levels.")

# Input fields
with st.sidebar:
    st.header("Configuration")
    domain = st.text_input("Domain (e.g., Programming, Science)", "Programming")
    topic = st.text_input("Topic (e.g., Python Pytorch, Quantum Physics)", "Python Pytorch")
    num_mcqs = st.slider("Number of MCQs to Generate", min_value=1, max_value=10, value=3)
    difficulty_options = ["Basic", "Intermediate", "Advanced", "College Standard"]
    difficulty = st.selectbox("Difficulty Level", difficulty_options, index=difficulty_options.index("College Standard"))

# Main area for generation button and results
if st.button("Generate MCQs"):
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        st.info("Generating MCQs, this might take a moment...")
        final_mcqs_data = generate_mcqs(topic, num_mcqs, difficulty)

        if final_mcqs_data:
            st.success("MCQ generation complete!")

            display_format = st.radio("Choose Display Format:", ("Expanded View", "CSV Format"))

            if display_format == "Expanded View":
                view_mcqs_expanded(final_mcqs_data, topic, difficulty)
            else:
                csv_output = view_mcqs_csv(final_mcqs_data, domain, topic, difficulty)
                st.text_area("CSV Output", csv_output, height=300)

                st.download_button(
                    label="Download MCQs as CSV",
                    data=csv_output,
                    file_name=f"mcqs_{topic.replace(' ', '_').lower()}_{difficulty.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )
        else:
            st.error("MCQ generation failed. Please check the logs for details or try again with different parameters.")

st.markdown("---")
st.markdown("Developed using Google's Gemini API, Mistral API, and Groq API.")
