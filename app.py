# app.py (Main Streamlit Application File)

import streamlit as st
import google.generativeai as gemini
from typing import Optional, List, Tuple
import concurrent.futures
import re, random
import ast
import json
from pydantic import BaseModel, TypeAdapter
from collections import defaultdict
import os
from dotenv import load_dotenv # Used for loading environment variables

# --- Configuration and Initialization ---
# Load environment variables from .env file
# This should be at the very top of your main application file.
load_dotenv()

# ====== Client Initialization ======
# Gemini API configuration. The API key will be automatically provided by the environment.
try:
    # Retrieve API key from environment variables
    # It's crucial to set GEMINI_API_KEY in your .env file
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        st.stop() # Stop the app if API key is missing

    gemini.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Gemini API initialization error: {e}")
    st.stop() # Stop the app if initialization fails

# --- api_wrappers.py (Separate file for API interaction) ---

def call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> Optional[str]:
    """
    Calls the Gemini API with a given prompt and model name.

    Args:
        prompt (str): The text prompt to send to the Gemini model.
        model (str): The name of the Gemini model to use (default: "gemini-2.0-flash").

    Returns:
        Optional[str]: The generated text response from the model, or None if an error occurs.
    """
    try:
        model_instance = gemini.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        # Check if response.text exists and is not empty before stripping
        return response.text.strip() if response.text else None
    except Exception as e:
        st.error(f"Gemini API call error: {e}")
        return None

# --- utils.py (Separate file for utility functions and Pydantic models) ---

class MCQs(BaseModel):
    """Pydantic model for a single Multiple Choice Question."""
    category: str
    question: str
    options: List[str]
    correct_option: str

def merge_results(input_strings: List[str]) -> str:
    """
    Merges a list of CSV-like strings, typically from verification steps.
    If all corresponding values for a key are "1", the output for that key is "1".
    Otherwise, it's "0".

    Args:
        input_strings (List[str]): A list of strings, each representing CSV-like data
                                    (e.g., '"1","1"\n"2","0"').

    Returns:
        str: A merged CSV-like string where each key's value is "1" only if all
             corresponding input values were "1".
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
            # else:
                # print(f"Skipping malformed line in merge_results: {line}") # Debugging
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

def parse_qa(input_string: str) -> List[List[str]]:
    """
    Parses a string of question-answer pairs into a numbered list of lists.
    Expected input format: '"Question text","Answer text"\n"Q2","A2"'

    Args:
        input_string (str): A string containing QA pairs.

    Returns:
        List[List[str]]: A list where each inner list contains
                         [QA_Number_as_String, Question_String, Answer_String].
    """
    qa_pairs = input_string.strip().split('\n')
    output = []

    for i, pair in enumerate(qa_pairs):
        # Split by '","' to handle quoted strings correctly
        parts = re.split(r'",\s*"', pair.strip('"'))
        if len(parts) == 2:
            question, answer = parts
            output.append([f'''"{i+1}"''', f'''"{question}"''', f'''"{answer}"'''])
        # else:
            # print(f"Skipping malformed QA pair in parse_qa: {pair}") # Debugging
    return output

def remove_zero_entries(parsed_data: List[List[str]], zero_values_data: List[str]) -> List[List[str]]:
    """
    Removes entries from parsed_data based on '0' values in zero_values_data.
    'zero_values_data' typically comes from the verification step.

    Args:
        parsed_data (List[List[str]]): List of QA pairs (e.g., [['"1"', '"Q1"', '"A1"']]).
        zero_values_data (List[str]): List of verification results (e.g., ['"1","1"\n"2","0"']).

    Returns:
        List[List[str]]: Filtered list of QA pairs where verification status was not '0'.
    """
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
        # Only include if status is not '0' (i.e., '1' or missing, assuming missing means valid)
        if verification_status.get(qa_number) != "0":
            filtered_data.append(entry)
    return filtered_data

def parse_and_filter_distractors_list(input_string: str) -> List[str]:
    """
    Parses a string of distractors into a list of strings.
    Expected input: '"d1"\n"d2"\n...'

    Args:
        input_string (str): A string containing distractors, one per line, quoted.

    Returns:
        List[str]: A list of parsed distractor strings.
    """
    lines = input_string.strip().split('\n')
    distractors = [line.strip('"') for line in lines]
    return distractors

def extract_list_within_balanced_brackets(input_string: str) -> Optional[List]:
    """
    Extracts and safely evaluates the list within the outermost balanced square brackets
    from a given string. Handles incomplete or unbalanced structures.

    Args:
        input_string (str): The string potentially containing a list.

    Returns:
        Optional[List]: The extracted Python list, or None if no valid list is found/parsed.
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
        # ast.literal_eval is safer than eval() for evaluating string literals
        extracted_list = ast.literal_eval(extracted_string)
        return extracted_list
    except (SyntaxError, ValueError) as e:
        st.error(f"Error evaluating string as a list in extract_list_within_balanced_brackets: {e}")
        return None

def parse_input(input_data: List[str]) -> List[List]:
    """
    Parses a list of strings, each expected to contain a list-like structure,
    into a list of Python lists.
    This is used to parse the raw distractor outputs from LLMs.

    Args:
        input_data (List[str]): A list of strings, each potentially containing a list.

    Returns:
        List[List]: A list of parsed Python lists.
    """
    parsed_data = []
    for item in input_data:
        # Use regex to find the entire JSON-like structure
        match = re.search(r"\[.*\]", item, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                parsed_list = ast.literal_eval(list_str)
                if isinstance(parsed_list, list):
                    parsed_data.append(parsed_list)
            except (SyntaxError, ValueError):
                # print(f"Could not parse list string: {list_str}") # Debugging
                continue
    return parsed_data

def merge_and_select(parsed_data: List[List]) -> List[List]:
    """
    Merges multiple lists of distractors (grouped by category) and selects the top 3
    most frequent distractors for each category.

    Args:
        parsed_data (List[List]): A list of lists, where each inner list is
                                  [category, [item1, item2, ...]].

    Returns:
        List[List]: A list where each inner list is [category, [top_3_distractors]].
    """
    merged = defaultdict(list)
    for data_list in parsed_data:
        for category, items in data_list:
            merged[category].extend(items)

    result = []
    for category, items in merged.items():
        frequency = {}
        for item in items:
            frequency[item] = frequency.get(item, 0) + 1
        # Sort by frequency in descending order, then alphabetically for ties
        sorted_items = sorted(frequency.keys(), key=lambda x: (-frequency[x], x))
        selected_items = sorted_items[:3] # Select top 3
        result.append([category, selected_items])

    return result

def parse_inputs(qa_pairs_data: List[List[str]], distractors_data: List[List[str]]) -> Tuple[dict, dict]:
    """
    Parses QA pairs and distractor data into structured dictionaries.

    Args:
        qa_pairs_data (List[List[str]]): List of QA pairs from `parse_qa`.
        distractors_data (List[List[str]]): List of selected distractors from `merge_and_select`.

    Returns:
        Tuple[dict, dict]: A tuple containing two dictionaries:
                          - parsed_qa: {category: {"question": str, "correct_answer": str}}
                          - parsed_distractors: {category: List[str]}
    """
    parsed_qa = {}
    for item in qa_pairs_data:
        # item is like ['"1"', '"What is X?"', '"Y"']
        category = item[0].strip('"')
        question = item[1].strip('"')
        correct_answer = item[2].strip('"')
        parsed_qa[category] = {"question": question, "correct_answer": correct_answer}

    parsed_distractors = {category: items for category, items in distractors_data}

    return parsed_qa, parsed_distractors

def merge_mcqs(parsed_qa: dict, parsed_distractors: dict) -> List[dict]:
    """
    Merges parsed QA pairs with selected distractors to form complete MCQ structures.

    Args:
        parsed_qa (dict): Dictionary of QA pairs.
        parsed_distractors (dict): Dictionary of distractors.

    Returns:
        List[dict]: A list of dictionaries, each representing a complete MCQ.
    """
    final_mcqs = []

    for category, qa in parsed_qa.items():
        question = qa["question"]
        correct_answer = qa["correct_answer"]
        options = parsed_distractors.get(category, [])

        options_for_mcq = list(options) # Create a copy to avoid modifying original list
        options_for_mcq.append(correct_answer)
        random.shuffle(options_for_mcq)

        # Ensure there are exactly 4 options by padding with correct answer if needed
        while len(options_for_mcq) < 4:
            options_for_mcq.append(correct_answer)
            random.shuffle(options_for_mcq)

        # Trim to exactly 4 options if more were generated (shouldn't happen with current logic)
        options_for_mcq = options_for_mcq[:4]

        # Determine the correct option (A, B, C, or D)
        try:
            correct_option_letter = chr(65 + options_for_mcq.index(correct_answer))
        except ValueError:
            st.warning(f"Correct answer '{correct_answer}' not found in options for category {category}. Setting correct option to A as a fallback.")
            correct_option_letter = 'A' # Fallback if correct answer somehow isn't in options

        final_mcqs.append({
            "category": category,
            "question": question,
            "options": options_for_mcq,
            "correct_option": correct_option_letter
        })

    return final_mcqs

# --- mcq_pipeline.py (Separate file for core MCQ generation logic) ---

def generate_qa_pairs(topic: str, count: int, difficulty: str) -> Optional[str]:
    """
    Generates Question and Answer pairs using the Gemini API.

    Args:
        topic (str): The subject topic for the QA pairs.
        count (int): The number of QA pairs to generate.
        difficulty (str): The desired difficulty level.

    Returns:
        Optional[str]: A string of generated QA pairs in CSV-like format, or None on failure.
    """
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

    response = call_gemini(prompt, "gemini-2.0-flash")
    return response

def verify_qa_pairs(qa_csv: str, count: int, difficulty: str) -> Optional[List[str]]:
    """
    Verifies the quality of generated QA pairs using the Gemini API.

    Args:
        qa_csv (str): A string containing the QA pairs in CSV-like format.
        count (int): The number of QA pairs being verified.
        difficulty (str): The difficulty level of the QA pairs.

    Returns:
        Optional[List[str]]: A list of verification results (e.g., ['"1","1"']), or None on failure.
    """
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
    results = []
    # Using ThreadPoolExecutor for potential future parallel calls, currently single call
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(call_gemini, verification_prompt, "gemini-2.0-flash")]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"Verification step error: {e}")
    return results if results else None

def generate_distractors(qa_csv: List[List[str]], difficulty: str) -> Optional[str]:
    """
    Generates a list of plausible but incorrect distractors for given QA pairs.

    Args:
        qa_csv (List[List[str]]): The list of QA pairs (e.g., [['"1"', '"Q1"', '"A1"']]).
        difficulty (str): The desired difficulty level for distractors.

    Returns:
        Optional[str]: A string containing the generated distractors in a list-of-lists format,
                       or None on failure.
    """
    # Convert list of lists to the expected string format for the prompt
    formatted_qa_pairs = ""
    for qa_pair in qa_csv:
        # Assuming qa_pair is ['"1"', '"Question"', '"Answer"']
        formatted_qa_pairs += f'{qa_pair[0]}, {qa_pair[1]}, {qa_pair[2]}\n'

    distractors_prompt = f'''
    Generate 10 high-quality plausible but incorrect distractors with difficulty level of {difficulty} for the given list of QA pairs (number, question, answer) following provided guidelines:
    QA Pairs:
    {formatted_qa_pairs.strip()}

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

    response = call_gemini(distractors_prompt, "gemini-2.0-flash")
    return response

def get_final_distractors(qa_csv_data: List[List[str]], distractors_raw_data: List[List], difficulty: str) -> Optional[List[List]]:
    """
    Selects the best 3 distractors from a larger list of generated distractors for each QA pair.

    Args:
        qa_csv_data (List[List[str]]): The list of QA pairs.
        distractors_raw_data (List[List]): The raw list of 10 distractors per category.
        difficulty (str): The difficulty level.

    Returns:
        Optional[List[List]]: A list of lists, where each inner list contains
                              [category, [selected_distractor1, selected_distractor2, selected_distractor3]],
                              or None on failure.
    """
    # Convert list of lists to the expected string format for the prompt
    formatted_qa_pairs = ""
    for qa_pair in qa_csv_data:
        formatted_qa_pairs += f'{qa_pair[0]}, {qa_pair[1]}, {qa_pair[2]}\n'

    final_distractors_prompt = f'''
    Based on the provided QA Pairs, select the best high-quality distractors from the list of distractors considering the dificulty level of {difficulty} given below:
    QA Pairs:
    {formatted_qa_pairs.strip()}

    Distractors:
    {json.dumps(distractors_raw_data, indent=2)}

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
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(call_gemini, final_distractors_prompt, "gemini-2.0-flash")]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"Final distractor selection error: {e}")

    # Parse the input
    parsed_data = parse_input(results)

    # Merge and select items
    final_result = merge_and_select(parsed_data)
    return final_result

def optimize_mcqs(all_mcqs: List[dict], difficulty: str) -> Optional[List[MCQs]]:
    """
    Optimizes a list of MCQs using the Gemini API with a structured JSON response.

    Args:
        all_mcqs (List[dict]): A list of compiled MCQs in dictionary format.
        difficulty (str): The difficulty level to consider during optimization.

    Returns:
        Optional[List[MCQs]]: A list of optimized MCQs as Pydantic models, or None on failure.
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

    client = gemini.GenerativeModel(model_name='gemini-2.0-flash')

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
        response = client.generate_content(
            contents=optimization_prompt,
            generation_config={
                'response_mime_type': 'application/json',
                'response_schema': mcqs_list_schema,
            },
        )
        generated_json_str = response.text
        parsed_data = json.loads(generated_json_str)
        # Validate the parsed JSON against the Pydantic model
        generated_mcqs: List[MCQs] = TypeAdapter(List[MCQs]).validate_python(parsed_data)
        return generated_mcqs
    except Exception as e:
        st.error(f"MCQ optimization failed: {e}")
        return None

def compile_mcqs(filtered_output: List[List[str]], final_result: List[List[str]]) -> List[dict]:
    """
    Compiles the final MCQs from filtered QA pairs and selected distractors.

    Args:
        filtered_output (List[List[str]]): The list of verified and filtered QA pairs.
        final_result (List[List[str]]): The list of selected distractors for each category.

    Returns:
        List[dict]: A list of compiled MCQs in dictionary format.
    """
    parsed_qa, parsed_distractors = parse_inputs(filtered_output, final_result)
    final_mcqs = merge_mcqs(parsed_qa, parsed_distractors)
    return final_mcqs

def generate_mcqs(topic: str, count: int, difficulty: str) -> Optional[List[MCQs]]:
    """
    Orchestrates the entire MCQ generation pipeline, including QA generation,
    verification, distractor generation, selection, compilation, and optimization.

    Args:
        topic (str): The topic for the MCQs.
        count (int): The desired number of MCQs.
        difficulty (str): The difficulty level for the MCQs.

    Returns:
        Optional[List[MCQs]]: A list of optimized MCQs as Pydantic models, or None if any step fails.
    """
    st.subheader("Generation Steps:")

    # Step 1: Generate QA Pairs
    with st.spinner("Generating QA Pairs..."):
        qa_pairs = generate_qa_pairs(topic, count, difficulty)
        if not qa_pairs:
            st.error("Failed to generate QA pairs. Please try again.")
            return None
        st.success("QA Pairs Generated.")

    # Step 2: Verify QA Pairs
    with st.spinner("Verifying QA Pairs..."):
        verification_results = verify_qa_pairs(qa_pairs, count, difficulty)
        if not verification_results:
            st.warning("Failed to verify QA pairs or no verification data returned. Proceeding with caution.")
            parsed_qa_pairs = parse_qa(qa_pairs) # Still parse even if no verification
            filtered_output = parsed_qa_pairs
        else:
            parsed_qa_pairs = parse_qa(qa_pairs)
            filtered_output = remove_zero_entries(parsed_qa_pairs, verification_results)

        if not filtered_output:
            st.warning("No valid QA pairs remaining after filtering. Adjust parameters or try again.")
            return None
        st.success(f"{len(filtered_output)} QA Pairs Verified and Filtered.")

    # Step 3: Generate Distractors
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

    # Step 4: Select Final Distractors
    with st.spinner("Selecting Final Distractors..."):
        final_distractors = get_final_distractors(filtered_output, cleaned_distractors, difficulty)
        if not final_distractors:
            st.error("Failed to select final distractors. Please try again.")
            return None
        st.success("Final Distractors Selected.")

    # Step 5: Compile MCQs
    with st.spinner("Compiling MCQs..."):
        all_mcqs = compile_mcqs(filtered_output, final_distractors)
        if not all_mcqs:
            st.error("Failed to compile MCQs. Please try again.")
            return None
        st.success("MCQs Compiled.")

    # Step 6: Optimize MCQs
    with st.spinner("Optimizing MCQs..."):
        final_optimized_mcqs = optimize_mcqs(all_mcqs, difficulty)
        if not final_optimized_mcqs:
            st.error("Failed to optimize MCQs. Displaying unoptimized MCQs if available.")
            return None # Return None if optimization fails
        st.success("MCQs Optimized!")

    return final_optimized_mcqs

# --- app.py (Continued - UI and Display Functions) ---

def view_mcqs_expanded(final_mcqs: List[MCQs], topic: str, difficulty: str):
    """
    Displays MCQs in an expanded, readable format within the Streamlit UI.

    Args:
        final_mcqs (List[MCQs]): A list of optimized MCQs (Pydantic models).
        topic (str): The topic of the MCQs.
        difficulty (str): The difficulty level of the MCQs.
    """
    st.header("Generated MCQs")
    if not final_mcqs:
        st.info("No MCQs to display. Please generate them first.")
        return

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
    """
    Generates a CSV formatted string of the MCQs.

    Args:
        final_mcqs (List[MCQs]): A list of optimized MCQs (Pydantic models).
        domain (str): The domain of the MCQs.
        topic (str): The topic of the MCQs.
        difficulty (str): The difficulty level of the MCQs.

    Returns:
        str: A string containing the MCQs in CSV format.
    """
    csv_output = ""
    # Header for CSV
    csv_output += '"Domain","Topic","Question","Option A","Option B","Option C","Option D","Correct Option","Difficulty"\n'
    for mcqs_item in final_mcqs:
        options_str = ""
        # Ensure there are always 4 options for CSV consistency
        # Pad with empty strings if less than 4 options (though typically 4 are generated)
        padded_options = list(mcqs_item.options) + [""] * (4 - len(mcqs_item.options))
        for option in padded_options[:4]: # Take only first 4 options
            options_str += f'"{option}",'
        options_str = options_str.rstrip(',') # Remove trailing comma

        csv_output += f'''"{domain}","{topic}","{mcqs_item.question}",{options_str},"{mcqs_item.correct_option}","{difficulty}"\n'''
    return csv_output

# --- Streamlit UI Layout ---
st.set_page_config(page_title="MCQ Generator", layout="centered")

st.title("🧠 MCQsAgent")
st.markdown("Generate multiple-choice questions on any topic with custom difficulty levels.")

# Input fields in the sidebar
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
        # Call the main generation pipeline
        final_mcqs_data = generate_mcqs(topic, num_mcqs, difficulty)

        if final_mcqs_data:
            st.success("MCQ generation complete!")

            # Display expanded view of MCQs
            view_mcqs_expanded(final_mcqs_data, topic, difficulty)

            st.markdown("---")
            st.subheader("MCQs in CSV Format")
            # Generate CSV output
            csv_output = view_mcqs_csv(final_mcqs_data, domain, topic, difficulty)
            st.text_area("CSV Output", csv_output, height=300)

            # Provide download button for CSV
            st.download_button(
                label="Download MCQs as CSV",
                data=csv_output,
                file_name=f"mcqs_{topic.replace(' ', '_').lower()}_{difficulty.lower()}.csv",
                mime="text/csv",
            )
        else:
            st.error("Failed to generate MCQs. Please review the errors above or try different parameters.")

