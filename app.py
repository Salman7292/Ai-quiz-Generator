from flask import Flask, render_template, request, jsonify, make_response, send_file, redirect, url_for, session, flash
from typing import List, Optional
from pydantic import BaseModel, ValidationError
import json
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from fpdf import FPDF
from io import BytesIO
import io
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mysqldb import MySQL
from functools import wraps
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or 'your-secret-key-here'

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'quiz_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# ================ DATA MODELS ================
class MCQOption(BaseModel):
    """Represents a single multiple-choice option"""
    text: str
    is_correct: bool

class Question(BaseModel):
    """Represents a quiz question with its options"""
    question_text: str
    options: List[MCQOption]
    difficulty: str
    topic: str

class Quiz(BaseModel):
    """Represents a complete quiz with metadata"""
    title: str
    questions: List[Question]
    total_questions: int
    difficulty_level: str
    topic: str
    id: int = None

class QuizResult(BaseModel):
    """Represents quiz results with user answers"""
    quiz_data: dict
    user_answers: dict
    score: int
    total_questions: int
    correct_answers: List[int]
    wrong_answers: List[int]

class ExamQuestion(BaseModel):
    """Represents an exam question (MCQ or descriptive)"""
    question_text: str
    question_type: str  # 'mcq' or 'descriptive'
    options: Optional[List[str]] = None  # Only for MCQs
    correct_answer: Optional[str] = None  # For MCQs, the correct option index
    difficulty: str
    topic: str
    max_marks: int = 5  # Each question worth 5 marks

class ExamPaper(BaseModel):
    """Represents a complete exam paper"""
    title: str
    questions: List[ExamQuestion]
    total_questions: int
    mcq_count: int
    descriptive_count: int
    difficulty_level: str
    topic: str
    total_marks: int

class ExamEvaluation(BaseModel):
    """Represents exam evaluation results"""
    exam_data: dict
    user_answers: dict
    score: float
    total_marks: int
    question_feedback: List[dict]
    topic_analysis: dict
    difficulty_analysis: dict

# ================ HELPER FUNCTIONS ================
def get_user_by_username(username):
    """Retrieve user by username from database"""
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    return user

def create_user(username, email, password):
    """Create a new user in the database"""
    hashed_password = generate_password_hash(password)
    cur = mysql.connection.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        mysql.connection.commit()
        return cur.lastrowid
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Error creating user: {str(e)}")
        return None
    finally:
        cur.close()

def validate_quiz_structure(quiz_data):
    """Validate quiz structure matches database requirements"""
    required_root_fields = ['title', 'topic', 'difficulty_level', 'total_questions', 'questions']
    for field in required_root_fields:
        if field not in quiz_data:
            raise ValueError(f"Quiz missing required field: {field}")
    
    if not isinstance(quiz_data['questions'], list):
        raise ValueError("Questions must be a list")
    
    for i, question in enumerate(quiz_data['questions'], 1):
        required_question_fields = ['question_text', 'options', 'difficulty', 'topic']
        for field in required_question_fields:
            if field not in question:
                raise ValueError(f"Question {i} missing required field: {field}")
        
        if not isinstance(question['options'], list):
            raise ValueError(f"Question {i} options must be a list")
        
        for j, option in enumerate(question['options'], 1):
            if 'text' not in option:
                raise ValueError(f"Question {i} option {j} missing text")
            if 'is_correct' not in option:
                option['is_correct'] = False  # Default to False if not specified

def save_quiz_to_db(quiz_data, user_id):
    """Save generated quiz to database"""
    try:
        # Validate quiz structure first
        validate_quiz_structure(quiz_data)
        
        cur = mysql.connection.cursor()
        
        # Insert quiz metadata
        cur.execute(
            """INSERT INTO quizzes 
            (title, topic, difficulty_level, total_questions, created_by) 
            VALUES (%s, %s, %s, %s, %s)""",
            (quiz_data['title'], 
             quiz_data['topic'],
             quiz_data['difficulty_level'],
             quiz_data['total_questions'],
             user_id)
        )
        quiz_id = cur.lastrowid
        
        # Insert questions and options
        for i, question in enumerate(quiz_data['questions'], 1):
            cur.execute(
                """INSERT INTO questions 
                (quiz_id, question_text, difficulty, topic, question_order) 
                VALUES (%s, %s, %s, %s, %s)""",
                (quiz_id,
                 question['question_text'],
                 question.get('difficulty', quiz_data['difficulty_level']),
                 question.get('topic', quiz_data['topic']),
                 i)
            )
            question_id = cur.lastrowid
            
            for j, option in enumerate(question['options'], 1):
                cur.execute(
                    """INSERT INTO options 
                    (question_id, option_text, is_correct, option_order) 
                    VALUES (%s, %s, %s, %s)""",
                    (question_id,
                     option['text'],
                     option.get('is_correct', False),
                     j)
                )
        
        mysql.connection.commit()
        return quiz_id
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Error saving quiz to database: {str(e)}")
        logger.error(f"Quiz data that failed: {json.dumps(quiz_data, indent=2)}")
        raise ValueError(f"Database error: {str(e)}") from e
    finally:
        cur.close()

def get_quiz_by_id(quiz_id):
    """Retrieve quiz by ID from database"""
    cur = mysql.connection.cursor()
    
    try:
        # Get quiz metadata
        cur.execute("SELECT * FROM quizzes WHERE id = %s", (quiz_id,))
        quiz = cur.fetchone()
        
        if quiz:
            # Get questions
            cur.execute("SELECT * FROM questions WHERE quiz_id = %s ORDER BY question_order", (quiz_id,))
            questions = cur.fetchall()
            
            quiz['questions'] = []
            for question in questions:
                # Get options for each question
                cur.execute(
                    "SELECT * FROM options WHERE question_id = %s ORDER BY option_order", 
                    (question['id'],))
                options = cur.fetchall()
                question['options'] = [{'text': opt['option_text'], 'is_correct': bool(opt['is_correct'])} for opt in options]
                quiz['questions'].append(question)
        
        return quiz
    except Exception as e:
        logger.error(f"Error getting quiz by ID: {str(e)}")
        return None
    finally:
        cur.close()

def get_user_quizzes(user_id):
    """Retrieve all quizzes created by a user"""
    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            SELECT id, title, topic, difficulty_level, total_questions, created_at 
            FROM quizzes 
            WHERE created_by = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        return cur.fetchall()
    except Exception as e:
        logger.error(f"Error getting user quizzes: {str(e)}")
        return []
    finally:
        cur.close()

def get_user_quiz_stats(user_id):
    """Get user statistics including quizzes created, completed, and average score"""
    cur = mysql.connection.cursor()
    try:
        # Get quizzes created count
        cur.execute("SELECT COUNT(*) as count FROM quizzes WHERE created_by = %s", (user_id,))
        quizzes_created = cur.fetchone()['count']
        
        # Get quiz results stats
        cur.execute("""
            SELECT 
                COUNT(*) as quizzes_completed,
                AVG(score) as average_score
            FROM quiz_results 
            WHERE user_id = %s
        """, (user_id,))
        result_stats = cur.fetchone()
        
        # Get user creation date
        cur.execute("SELECT created_at FROM users WHERE id = %s", (user_id,))
        user_info = cur.fetchone()
        
        # Get recent quizzes
        cur.execute("""
            SELECT q.id, q.title, q.topic, q.difficulty_level, q.created_at 
            FROM quizzes q
            WHERE q.created_by = %s
            ORDER BY q.created_at DESC
            LIMIT 3
        """, (user_id,))
        recent_quizzes = cur.fetchall()
        
        return {
            'quizzes_created': quizzes_created,
            'quizzes_completed': result_stats['quizzes_completed'] if result_stats['quizzes_completed'] else 0,
            'average_score': float(result_stats['average_score']) * 100 if result_stats['average_score'] else 0,
            'created_at': user_info['created_at'],
            'recent_quizzes': recent_quizzes
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return None
    finally:
        cur.close()

def save_quiz_result(quiz_id, user_id, score, total_questions):
    """Save quiz results to database"""
    cur = mysql.connection.cursor()
    try:
        cur.execute(
            """INSERT INTO quiz_results 
            (quiz_id, user_id, score, total_questions) 
            VALUES (%s, %s, %s, %s)""",
            (quiz_id, user_id, score, total_questions)
        )
        mysql.connection.commit()
        return cur.lastrowid
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Error saving quiz result: {str(e)}")
        return None
    finally:
        cur.close()

def is_authenticated():
    """Check if user is logged in"""
    return 'username' in session

def login_required(f):
    """Decorator to ensure user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def generate_subtopics(topic: str, llm) -> List[str]:
    """Generate a list of subtopics for a given general topic using Gemini API"""
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="""
        Given the general topic '{topic}', please break it down into 5-10 specific subtopics 
        that would be suitable for creating quizzes. Return ONLY a Python list of strings 
        (no additional text or markdown formatting).

        Example output for "Science":
        ["Biology", "Chemistry", "Physics", "Astronomy", "Geology"]

        Important:
        - Return ONLY a valid Python list
        - Each subtopic should be distinct and specific
        - Subtopics should be suitable for quiz questions
        - Do not include any additional text or explanations
        """
    )

    chain = (RunnablePassthrough() | prompt | llm)
    result = chain.invoke({"topic": topic})

    try:
        content = result.content if hasattr(result, 'content') else str(result)
        # Clean the response to extract just the Python list
        clean_content = content.replace("```python", "").replace("```", "").strip()
        subtopics = eval(clean_content)  # Safely evaluate the string as Python list
        
        if not isinstance(subtopics, list) or not all(isinstance(x, str) for x in subtopics):
            raise ValueError("Generated subtopics are not in the correct format")
            
        return subtopics
    except Exception as e:
        logger.error(f"Error generating subtopics: {str(e)}")
        raise ValueError(f"Failed to generate subtopics: {str(e)}")

def generate_quiz(topic: str, llm, num_questions: int = 5, difficulty_level: str = "medium") -> dict:
    """Generate quiz with loading state handling"""
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")
    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 20:
        raise ValueError("Number of questions must be between 1 and 20")
    if difficulty_level not in ["easy", "medium", "hard"]:
        raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

    # Check if the topic is in the format "subtopic in general_topic"
    if " in " in topic:
        # If it's already in the correct format, use it as is
        formatted_topic = topic
    else:
        # Otherwise, use just the topic (for backward compatibility)
        formatted_topic = topic

    prompt = PromptTemplate(
        input_variables=["topic", "num_questions", "difficulty_level"],
        template="""
        Create a {num_questions}-question quiz about {topic} where ALL questions are {difficulty_level} difficulty.
        Each question must have exactly 4 options (one correct). Format as valid JSON:
        {{
            "title": "Quiz about {topic} ({difficulty_level})",
            "questions": [
                {{
                    "question_text": "...",
                    "options": [
                        {{"text": "...", "is_correct": false}},
                        {{"text": "...", "is_correct": true}},
                        {{"text": "...", "is_correct": false}},
                        {{"text": "...", "is_correct": false}}
                    ],
                    "difficulty": "{difficulty_level}",
                    "topic": "{topic}"
                }}
            ],
            "total_questions": {num_questions},
            "difficulty_level": "{difficulty_level}",
            "topic": "{topic}"
        }}
        Important:
        - Only respond with valid JSON (no markdown)
        - Ensure exactly one correct answer per question
        - Make the correct answers randomized in position (not always the same option number)
        """
    )

    chain = (RunnablePassthrough() | prompt | llm)
    result = chain.invoke({
        "topic": formatted_topic,  # Use the formatted topic here
        "num_questions": num_questions,
        "difficulty_level": difficulty_level
    })

    try:
        content = result.content if hasattr(result, 'content') else str(result)
        clean_content = content.replace("```json", "").replace("```", "").strip()
        quiz_data = json.loads(clean_content)
        
        # Ensure consistent structure
        quiz_data['topic'] = formatted_topic  # Use the formatted topic here
        quiz_data['difficulty_level'] = difficulty_level
        
        # Ensure each question has topic and difficulty
        for question in quiz_data['questions']:
            question['topic'] = question.get('topic', formatted_topic)
            question['difficulty'] = question.get('difficulty', difficulty_level)
        
        validated_quiz = Quiz(**quiz_data)
        return validated_quiz.model_dump()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise ValueError("Failed to parse quiz data. Please try again.")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise ValueError("Invalid quiz format received. Please try again.")
    except Exception as e:
        logger.error(f"Quiz generation error: {str(e)}")
        raise ValueError(f"Quiz generation failed: {str(e)}")

def generate_pdf(quiz_data: dict):
    """Generate compact PDF from quiz data"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 6, txt=quiz_data['title'], ln=1, align='C')
    pdf.set_font("Arial", size=9)
    pdf.cell(200, 5, txt=f"Total Questions: {quiz_data['total_questions']} | Difficulty: {quiz_data['difficulty_level'].capitalize()}", ln=1, align='C')
    pdf.ln(4)
    
    for i, question in enumerate(quiz_data['questions'], 1):
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 5, txt=f"{i}. {question['question_text']}")
        pdf.set_font("Arial", size=9)
        
        for option in question['options']:
            prefix = "✓" if option['is_correct'] else "•"
            pdf.cell(5, 4, txt=prefix)
            pdf.multi_cell(0, 4, txt=f" {option['text']}")
        pdf.ln(2)
    
    return pdf

def calculate_results(quiz_data: dict, user_answers: dict) -> dict:
    """Calculate quiz results based on user answers"""
    score = 0
    correct_answers = []
    wrong_answers = []
    
    for i, question in enumerate(quiz_data['questions']):
        question_index = str(i)
        if question_index in user_answers:
            selected_option_index = user_answers[question_index]
            correct_option_index = next((j for j, opt in enumerate(question['options']) if opt['is_correct']), None)
            
            if selected_option_index == correct_option_index:
                score += 1
                correct_answers.append(i)
            else:
                wrong_answers.append(i)
    
    result = QuizResult(
        quiz_data=quiz_data,
        user_answers=user_answers,
        score=score,
        total_questions=len(quiz_data['questions']),
        correct_answers=correct_answers,
        wrong_answers=wrong_answers
    )
    
    return result.model_dump()

def generate_exam_paper(topic: str, llm, mcq_count: int = 5, descriptive_count: int = 5, 
                       difficulty_level: str = "medium", paper_style: str = "balanced") -> dict:
    """Generate an exam paper with MCQs and descriptive questions"""
    if not topic or not isinstance(topic, str):
        raise ValueError("Topic must be a non-empty string")
    
    if mcq_count < 0 or descriptive_count < 0 or (mcq_count + descriptive_count) < 1:
        raise ValueError("Must have at least one question")
    
    if difficulty_level not in ["easy", "medium", "hard"]:
        raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
    
    if paper_style not in ["simple", "balanced", "technical"]:
        raise ValueError("Paper style must be 'simple', 'balanced', or 'technical'")

    prompt = PromptTemplate(
        input_variables=["topic", "mcq_count", "descriptive_count", "difficulty_level", "paper_style"],
        template="""
        Create an exam paper about {topic} with:
        - {mcq_count} multiple-choice questions (MCQs)
        - {descriptive_count} descriptive questions
        - All questions should be {difficulty_level} difficulty
        - Paper style should be {paper_style}
        - Each question is worth 5 marks
        
        Format as valid JSON:
        {{
            "title": "Exam Paper on {topic}",
            "questions": [
                {{
                    "question_text": "...",
                    "question_type": "mcq",
                    "options": ["option1", "option2", "option3", "option4"],
                    "correct_answer": "option_index",  // e.g., "0" for first option
                    "difficulty": "{difficulty_level}",
                    "topic": "{topic}",
                    "max_marks": 5
                }},
                {{
                    "question_text": "...",
                    "question_type": "descriptive",
                    "difficulty": "{difficulty_level}",
                    "topic": "{topic}",
                    "max_marks": 5
                }}
            ],
            "total_questions": {total_questions},
            "mcq_count": {mcq_count},
            "descriptive_count": {descriptive_count},
            "difficulty_level": "{difficulty_level}",
            "topic": "{topic}",
            "total_marks": {total_marks}
        }}
        
        Important:
        - Only respond with valid JSON (no markdown)
        - For MCQs: provide exactly 4 options and one correct answer
        - For descriptive questions: don't include options
        - Make questions appropriate for the {paper_style} style
        - Ensure questions match the {difficulty_level} difficulty
        - Each question is worth 5 marks (max_marks: 5)
        """
    ).partial(
        total_questions=mcq_count + descriptive_count,
        total_marks=(mcq_count + descriptive_count) * 5
    )

    chain = (RunnablePassthrough() | prompt | llm)
    result = chain.invoke({
        "topic": topic,
        "mcq_count": mcq_count,
        "descriptive_count": descriptive_count,
        "difficulty_level": difficulty_level,
        "paper_style": paper_style
    })

    try:
        content = result.content if hasattr(result, 'content') else str(result)
        clean_content = content.replace("```json", "").replace("```", "").strip()
        exam_data = json.loads(clean_content)
        
        # Validate the structure
        validated_exam = ExamPaper(**exam_data)
        return validated_exam.model_dump()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise ValueError("Failed to parse exam data. Please try again.")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise ValueError("Invalid exam format received. Please try again.")
    except Exception as e:
        logger.error(f"Exam generation error: {str(e)}")
        raise ValueError(f"Exam generation failed: {str(e)}")

def evaluate_descriptive_answers(exam_data: dict, user_answers: dict, llm) -> List[dict]:
    """Evaluate descriptive answers using LLM with detailed 5-mark grading"""
    feedback = []
    
    for i, question in enumerate(exam_data['questions']):
        if question['question_type'] == 'descriptive':
            user_answer = user_answers.get(str(i), "").strip()
            if not user_answer:
                feedback.append({
                    "question_index": i,
                    "marks": 0,
                    "feedback": "No answer provided",
                    "model_answer": "N/A"
                })
                continue
                
            evaluation_prompt = PromptTemplate(
                input_variables=["question", "answer", "max_marks"],
                template="""
                Evaluate the following exam answer out of {max_marks} marks. Provide:
                1. Marks awarded (0-{max_marks})
                2. Detailed feedback on strengths and weaknesses
                3. A model answer for comparison
                
                Question: {question}
                Student Answer: {answer}
                
                Respond ONLY with JSON format:
                {{
                    "marks": 0-{max_marks},
                    "feedback": "detailed feedback here",
                    "model_answer": "ideal answer here"
                }}
                
                Grading Criteria:
                - Accuracy and completeness (2 marks)
                - Clarity and organization (1 mark)
                - Depth of analysis (1 mark)
                - Originality (1 mark)
                """
            )
            
            chain = (RunnablePassthrough() | evaluation_prompt | llm)
            try:
                eval_result = chain.invoke({
                    "question": question['question_text'],
                    "answer": user_answer,
                    "max_marks": question['max_marks']
                })
                
                eval_content = eval_result.content if hasattr(eval_result, 'content') else str(eval_result)
                eval_data = json.loads(eval_content.replace("```json", "").replace("```", "").strip())
                
                feedback.append({
                    "question_index": i,
                    "marks": eval_data['marks'],
                    "feedback": eval_data['feedback'],
                    "model_answer": eval_data['model_answer']
                })
            except Exception as e:
                logger.error(f"Error evaluating answer: {str(e)}")
                feedback.append({
                    "question_index": i,
                    "marks": 0,
                    "feedback": "Could not evaluate answer",
                    "model_answer": "N/A"
                })
    
    return feedback

def analyze_results(exam_data: dict, user_answers: dict, question_feedback: List[dict]) -> dict:
    """Analyze results by topic and difficulty"""
    topic_stats = {}
    difficulty_stats = {
        "easy": {"total": 0, "correct": 0, "marks": 0},
        "medium": {"total": 0, "correct": 0, "marks": 0},
        "hard": {"total": 0, "correct": 0, "marks": 0}
    }
    
    for i, question in enumerate(exam_data['questions']):
        # Initialize topic stats if not exists
        if question['topic'] not in topic_stats:
            topic_stats[question['topic']] = {
                "total": 0,
                "correct": 0,
                "marks": 0
            }
        
        # Update topic stats
        topic_stats[question['topic']]['total'] += 1
        if question['question_type'] == 'mcq':
            # Convert to string for consistent comparison
            user_answer = str(user_answers.get(str(i)))
            correct_answer = str(question['correct_answer'])
            is_correct = user_answer == correct_answer
            marks = question['max_marks'] if is_correct else 0
            topic_stats[question['topic']]['correct'] += int(is_correct)
            topic_stats[question['topic']]['marks'] += marks
        else:
            # For descriptive questions, use the evaluated marks
            feedback = next((f for f in question_feedback if f['question_index'] == i), None)
            marks = feedback['marks'] if feedback else 0
            topic_stats[question['topic']]['marks'] += marks
            topic_stats[question['topic']]['correct'] += int(marks >= question['max_marks'] * 0.8)  # 80% threshold
        
        # Update difficulty stats
        difficulty = question['difficulty']
        difficulty_stats[difficulty]['total'] += 1
        if question['question_type'] == 'mcq':
            user_answer = str(user_answers.get(str(i)))
            correct_answer = str(question['correct_answer'])
            is_correct = user_answer == correct_answer
            marks = question['max_marks'] if is_correct else 0
            difficulty_stats[difficulty]['correct'] += int(is_correct)
            difficulty_stats[difficulty]['marks'] += marks
        else:
            # For descriptive questions, use the evaluated marks
            feedback = next((f for f in question_feedback if f['question_index'] == i), None)
            marks = feedback['marks'] if feedback else 0
            difficulty_stats[difficulty]['marks'] += marks
            difficulty_stats[difficulty]['correct'] += int(marks >= question['max_marks'] * 0.8)  # 80% threshold
    
    return {
        "topic_analysis": topic_stats,
        "difficulty_analysis": difficulty_stats
    }

def generate_charts(analysis_data: dict) -> dict:
    """Generate Chart.js compatible chart configurations for results visualization"""
    # Topic Performance Chart
    topics = list(analysis_data['topic_analysis'].keys())
    topic_accuracy = [
        (analysis_data['topic_analysis'][t]['correct'] / analysis_data['topic_analysis'][t]['total']) * 100 
        if analysis_data['topic_analysis'][t]['total'] > 0 else 0
        for t in topics
    ]
    
    topic_chart = {
        "type": "bar",
        "data": {
            "labels": topics,
            "datasets": [{
                "label": "Accuracy (%)",
                "data": topic_accuracy,
                "backgroundColor": '#636EFA',
                "borderColor": '#636EFA',
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Performance by Topic",
                    "font": {"size": 16}
                },
                "legend": {"display": False}
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 100,
                    "title": {
                        "display": True,
                        "text": "Accuracy (%)"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Topic"
                    }
                }
            }
        }
    }
    
    # Difficulty Performance Chart
    difficulties = ['easy', 'medium', 'hard']
    difficulty_accuracy = [
        (analysis_data['difficulty_analysis'][d]['correct'] / analysis_data['difficulty_analysis'][d]['total']) * 100 
        if analysis_data['difficulty_analysis'][d]['total'] > 0 else 0
        for d in difficulties
    ]
    
    difficulty_chart = {
        "type": "bar",
        "data": {
            "labels": difficulties,
            "datasets": [{
                "label": "Accuracy (%)",
                "data": difficulty_accuracy,
                "backgroundColor": ['#00CC96', '#AB63FA', '#FFA15A'],
                "borderColor": ['#00CC96', '#AB63FA', '#FFA15A'],
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Performance by Difficulty",
                    "font": {"size": 16}
                },
                "legend": {"display": False}
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 100,
                    "title": {
                        "display": True,
                        "text": "Accuracy (%)"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Difficulty Level"
                    }
                }
            }
        }
    }

    # Marks Distribution Chart
    total_marks = sum([analysis_data['topic_analysis'][t]['marks'] for t in topics])
    total_possible = sum([analysis_data['topic_analysis'][t]['total'] * 5 for t in topics])
    
    marks_chart = None
    if total_possible > 0:
        marks_chart = {
            "type": "doughnut",
            "data": {
                "labels": ["Correct", "Incorrect"],
                "datasets": [{
                    "data": [total_marks, total_possible - total_marks],
                    "backgroundColor": ['#00CC96', '#EF553B'],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Marks Distribution",
                        "font": {"size": 16}
                    }
                },
                "cutout": "40%"
            }
        }
    
    return {
        "topic_chart": topic_chart,
        "difficulty_chart": difficulty_chart,
        "marks_chart": marks_chart
    }

# ================ ROUTES ================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_authenticated():
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please fill in all fields', 'danger')
            return redirect(url_for('login'))
        
        user = get_user_by_username(username)
        if not user or not check_password_hash(user['password_hash'], password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        session['username'] = user['username']
        session['user_id'] = user['id']
        # flash('Logged in successfully!', 'success')
        
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if is_authenticated():
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields', 'danger')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        
        if get_user_by_username(username):
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))
        
        user_id = create_user(username, email, password)
        if user_id:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Error creating account. Please try again.', 'danger')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    # flash('Logged out successfully', 'success')
    return redirect(url_for('Home'))

@app.route('/', methods=['GET'])
@login_required
def index():
    """Main quiz creation page with optional quiz loading"""
    quiz_to_load = None
    load_quiz_id = request.args.get('load_quiz')
    
    if load_quiz_id:
        try:
            quiz_to_load = get_quiz_by_id(load_quiz_id)
            if quiz_to_load and quiz_to_load['created_by'] != session['user_id']:
                flash('You are not authorized to load this quiz', 'danger')
                quiz_to_load = None
        except Exception as e:
            logger.error(f"Error loading quiz: {str(e)}")
            flash('Error loading quiz', 'danger')
    
    return render_template('index.html', generated=False, quiz_to_load=quiz_to_load)

@app.route('/generate-subtopics', methods=['POST'])
@login_required
def generate_subtopics_api():
    """API endpoint for generating subtopics from a general topic"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.debug(f"Received subtopic request data: {data}")
        
        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("Google API key not configured")
            return jsonify({'error': 'API key not configured'}), 500
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.5,  # Lower temperature for more focused subtopics
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            subtopics = generate_subtopics(topic=topic, llm=llm)
            logger.debug(f"Generated subtopics: {subtopics}")
            return jsonify({'subtopics': subtopics})
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating subtopics: {str(e)}")
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in generate_subtopics_api: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/generate', methods=['POST'])
@login_required
def generate_api():
    """API endpoint for AJAX quiz generation"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
            
        try:
            num_questions = int(data.get('num_questions', 5))
            if num_questions < 1 or num_questions > 20:
                return jsonify({'error': 'Number of questions must be between 1 and 20'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid number of questions'}), 400
            
        difficulty = data.get('difficulty', 'medium')
        if difficulty not in ['easy', 'medium', 'hard']:
            return jsonify({'error': 'Invalid difficulty level'}), 400
        
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("Google API key not configured")
            return jsonify({'error': 'API key not configured'}), 500
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            quiz = generate_quiz(
                topic=topic,
                llm=llm,
                num_questions=num_questions,
                difficulty_level=difficulty
            )
            
            # Save to database
            user_id = session['user_id']
            quiz_id = save_quiz_to_db(quiz, user_id)
            quiz['id'] = quiz_id
            
            logger.debug("Quiz generated successfully")
            return jsonify(quiz)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in generate_api: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/submit-quiz', methods=['POST'])
@login_required
def submit_quiz():
    """Endpoint to submit quiz answers and get results"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        quiz_data = data.get('quiz_data')
        user_answers = data.get('user_answers')
        
        if not quiz_data or not user_answers:
            return jsonify({'error': 'Missing quiz data or answers'}), 400
        
        results = calculate_results(quiz_data, user_answers)
        
        # Save results to database
        user_id = session['user_id']
        save_quiz_result(
            quiz_id=quiz_data['id'],
            user_id=user_id,
            score=results['score'],
            total_questions=results['total_questions']
        )
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in submit_quiz: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-quiz-pdf', methods=['POST'])
@login_required
def download_quiz_pdf():
    """Endpoint to download quiz as PDF"""
    try:
        quiz_data = request.get_json()
        if not quiz_data:
            return jsonify({'error': 'No quiz data provided'}), 400

        pdf = generate_pdf(quiz_data)
        
        # Create a bytes buffer for the PDF
        pdf_bytes = io.BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)
        
        return send_file(
            pdf_bytes,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{quiz_data['title'].replace(' ', '_')}.pdf"
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load-quiz')
@login_required
def load_quiz():
    """Page to load previous quizzes"""
    try:
        user_id = session['user_id']
        quizzes = get_user_quizzes(user_id)
        return render_template('load_quiz.html', quizzes=quizzes)
    except Exception as e:
        logger.error(f"Error in load_quiz: {str(e)}")
        flash('Error loading quizzes', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/my-quizzes')
@login_required
def my_quizzes():
    """Display all quizzes created by the user"""
    try:
        user_id = session['user_id']
        quizzes = get_user_quizzes(user_id)
        return render_template('my_quizzes.html', quizzes=quizzes)
    except Exception as e:
        logger.error(f"Error in my_quizzes: {str(e)}")
        flash('Error loading quizzes', 'danger')
        return redirect(url_for('index'))

@app.route('/quiz/<int:quiz_id>')
@login_required
def view_quiz(quiz_id):
    """View a specific quiz"""
    try:
        quiz = get_quiz_by_id(quiz_id)
        if not quiz:
            flash('Quiz not found', 'danger')
            return redirect(url_for('my_quizzes'))
        
        if quiz['created_by'] != session['user_id']:
            flash('You are not authorized to view this quiz', 'danger')
            return redirect(url_for('my_quizzes'))
        
        return render_template('view_quiz.html', quiz=quiz)
    except Exception as e:
        logger.error(f"Error in view_quiz: {str(e)}")
        flash('Error loading quiz', 'danger')
        return redirect(url_for('my_quizzes'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with statistics"""
    try:
        user_id = session['user_id']
        user_stats = get_user_quiz_stats(user_id)
        
        if not user_stats:
            flash('Error loading dashboard data', 'danger')
            return render_template('dashboard.html', user_stats=None, recent_quizzes=[])
        
        return render_template(
            'dashboard.html',
            user_stats=user_stats,
            recent_quizzes=user_stats['recent_quizzes']
        )
        
    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        flash('Error loading dashboard data', 'danger')
        return render_template('dashboard.html', user_stats=None, recent_quizzes=[])

@app.route('/Home')
def Home():
    return render_template('Home.html')

@app.route('/About')
def About():
    return render_template('about.html')

@app.route('/analytics')
@login_required
def analytics():
    """User analytics page with performance data"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # 1. Basic stats
        cur.execute("""
            SELECT 
                COUNT(*) AS quizzes_completed,
                AVG(score) AS average_score
            FROM quiz_results 
            WHERE user_id = %s
        """, (user_id,))
        stats = cur.fetchone()
        
        # 2. Score progress over time
        cur.execute("""
            SELECT DATE(completed_at) AS date, AVG(score) AS avg_score
            FROM quiz_results 
            WHERE user_id = %s
            GROUP BY DATE(completed_at)
            ORDER BY DATE(completed_at) ASC
        """, (user_id,))
        score_progress = cur.fetchall()
        
        # 3. Topic mastery
        cur.execute("""
            SELECT q.topic, AVG(qr.score) AS avg_score, COUNT(qr.id) AS quiz_count
            FROM quiz_results qr
            JOIN quizzes q ON qr.quiz_id = q.id
            WHERE qr.user_id = %s
            GROUP BY q.topic
            ORDER BY avg_score DESC
            LIMIT 6
        """, (user_id,))
        topic_mastery = cur.fetchall()
        
        # 4. Difficulty distribution
        cur.execute("""
            SELECT q.difficulty_level, COUNT(*) AS count
            FROM quiz_results qr
            JOIN quizzes q ON qr.quiz_id = q.id
            WHERE qr.user_id = %s
            GROUP BY q.difficulty_level
        """, (user_id,))
        difficulty_dist = cur.fetchall()
        
        # 5. Time of day performance
        cur.execute("""
            SELECT 
                CASE 
                    WHEN HOUR(completed_at) BETWEEN 6 AND 11 THEN 'Morning'
                    WHEN HOUR(completed_at) BETWEEN 12 AND 17 THEN 'Afternoon'
                    WHEN HOUR(completed_at) BETWEEN 18 AND 22 THEN 'Evening'
                    ELSE 'Night'
                END AS time_of_day,
                AVG(score) AS avg_score
            FROM quiz_results
            WHERE user_id = %s
            GROUP BY time_of_day
            ORDER BY FIELD(time_of_day, 'Morning', 'Afternoon', 'Evening', 'Night')
        """, (user_id,))
        time_performance = cur.fetchall()
        
        # 6. Personalized recommendations
        cur.execute("""
            SELECT 
                q.topic, 
                COUNT(*) AS total_questions,
                SUM(CASE WHEN o.is_correct = TRUE THEN 1 ELSE 0 END) AS correct_answers
            FROM quiz_results qr
            JOIN quizzes q ON qr.quiz_id = q.id
            JOIN questions qu ON q.id = qu.quiz_id
            JOIN options o ON qu.id = o.question_id
            WHERE qr.user_id = %s
            GROUP BY q.topic
            HAVING (SUM(CASE WHEN o.is_correct = TRUE THEN 1 ELSE 0 END) / COUNT(*)) < 0.7
            ORDER BY (SUM(CASE WHEN o.is_correct = TRUE THEN 1 ELSE 0 END) / COUNT(*)) ASC
            LIMIT 3
        """, (user_id,))
        recommendations = cur.fetchall()
        
        # Prepare data for charts
        score_data = {
            'dates': [str(row['date']) for row in score_progress],
            'scores': [float(row['avg_score']) for row in score_progress]
        }
        
        topic_data = {
            'topics': [row['topic'] for row in topic_mastery],
            'scores': [float(row['avg_score']) for row in topic_mastery]
        }
        
        difficulty_data = {
            'difficulties': [row['difficulty_level'] for row in difficulty_dist],
            'counts': [row['count'] for row in difficulty_dist]
        }
        
        time_data = {
            'times': [row['time_of_day'] for row in time_performance],
            'scores': [float(row['avg_score']) for row in time_performance]
        }
        
        # Format recommendations
        formatted_recs = []
        for rec in recommendations:
            accuracy = (rec['correct_answers'] / rec['total_questions']) * 100
            formatted_recs.append({
                'topic': rec['topic'],
                'accuracy': round(accuracy, 1),
                'question_count': rec['total_questions']
            })
        
        # Calculate overall stats
        quizzes_completed = stats['quizzes_completed'] or 0
        average_score = float(stats['average_score'] or 0)
        
        analytics_data = {
            'quizzes_completed': quizzes_completed,
            'average_score': round(average_score, 1),
            'score_progress': score_data,
            'topic_mastery': topic_data,
            'difficulty_dist': difficulty_data,
            'time_performance': time_data,
            'recommendations': formatted_recs
        }
        
        return render_template('analytics.html', analytics_data=analytics_data)
        
    except Exception as e:
        logger.error(f"Error in analytics: {str(e)}")
        flash('Error loading analytics data', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/digital-exam')
@login_required
def digital_exam():
    return render_template('digital.html')

@app.route('/digital-exam/generate-exam', methods=['POST'])
@login_required
def digital_generate_exam():
    """API endpoint for generating exam papers"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.debug(f"Received exam request data: {data}")
        
        topic = data.get('topic', '').strip()
        mcq_count = int(data.get('mcq_count', 5))
        descriptive_count = int(data.get('descriptive_count', 5))
        difficulty = data.get('difficulty', 'medium')
        paper_style = data.get('paper_style', 'balanced')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
            
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("Google API key not configured")
            return jsonify({'error': 'API key not configured'}), 500
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            exam = generate_exam_paper(
                topic=topic,
                llm=llm,
                mcq_count=mcq_count,
                descriptive_count=descriptive_count,
                difficulty_level=difficulty,
                paper_style=paper_style
            )
            
            # Save exam to database
            user_id = session['user_id']
            cur = mysql.connection.cursor()
            
            # Insert exam metadata
            cur.execute(
                """INSERT INTO exam_papers 
                (title, topic, difficulty_level, total_questions, mcq_count, descriptive_count, total_marks, created_by) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (exam['title'], 
                 exam['topic'],
                 exam['difficulty_level'],
                 exam['total_questions'],
                 exam['mcq_count'],
                 exam['descriptive_count'],
                 exam['total_marks'],
                 user_id)
            )
            exam_id = cur.lastrowid
            
            # Insert questions and options
            for i, question in enumerate(exam['questions'], 1):
                cur.execute(
                    """INSERT INTO exam_questions 
                    (exam_id, question_text, question_type, difficulty, topic, max_marks, correct_answer, question_order) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (exam_id,
                     question['question_text'],
                     question['question_type'],
                     question['difficulty'],
                     question['topic'],
                     question['max_marks'],
                     str(question.get('correct_answer', '')) if question['question_type'] == 'mcq' else None,
                     i)
                )
                question_id = cur.lastrowid
                
                if question['question_type'] == 'mcq' and question.get('options'):
                    for j, option in enumerate(question['options'], 1):
                        cur.execute(
                            """INSERT INTO exam_question_options 
                            (question_id, option_text, option_order) 
                            VALUES (%s, %s, %s)""",
                            (question_id,
                             option,
                             j)
                        )
            
            mysql.connection.commit()
            exam['id'] = exam_id
            
            logger.debug("Exam generated and saved successfully")
            return jsonify(exam)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            mysql.connection.rollback()
            logger.error(f"Error generating exam: {str(e)}")
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in generate_exam: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
    


@app.route('/digital-exam/evaluate-exam', methods=['POST'])
@login_required
def digital_evaluate_exam():
    """API endpoint for evaluating exam papers"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.debug(f"Received evaluation request data: {data}")
        
        exam_data = data.get('exam_data')
        user_answers = data.get('user_answers')
        
        if not exam_data or not user_answers:
            return jsonify({'error': 'Missing exam data or answers'}), 400
            
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("Google API key not configured")
            return jsonify({'error': 'API key not configured'}), 500
            
        # First calculate MCQ results
        mcq_marks = 0
        question_feedback = []
        
        for i, question in enumerate(exam_data['questions']):
            if question['question_type'] == 'mcq':
                # Convert to string for consistent comparison
                user_answer = str(user_answers.get(str(i)))
                correct_answer = str(question['correct_answer'])
                is_correct = user_answer == correct_answer
                marks = question['max_marks'] if is_correct else 0
                mcq_marks += marks
                
                feedback = {
                    "question_index": i,
                    "marks": marks,
                    "feedback": "Correct!" if is_correct else 
                              f"Incorrect. The correct answer was: {question['options'][int(question['correct_answer'])]}",
                    "model_answer": question['options'][int(question['correct_answer'])]
                }
                question_feedback.append(feedback)
        
        # Then evaluate descriptive answers with LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,  # Lower temperature for more consistent evaluations
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        descriptive_feedback = evaluate_descriptive_answers(exam_data, user_answers, llm)
        descriptive_marks = sum(f['marks'] for f in descriptive_feedback)
        
        # Combine all feedback
        question_feedback.extend(descriptive_feedback)
        question_feedback.sort(key=lambda x: x['question_index'])
        
        # Calculate total score
        total_marks = mcq_marks + descriptive_marks
        
        # Analyze results by topic and difficulty
        analysis = analyze_results(exam_data, user_answers, question_feedback)
        
        # Generate visualization charts
        charts = generate_charts(analysis)
        
        # Save exam results to database
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        try:
            # Insert exam result
            cur.execute(
                """INSERT INTO exam_results 
                (exam_id, user_id, score, total_marks) 
                VALUES (%s, %s, %s, %s)""",
                (exam_data['id'],
                 user_id,
                 total_marks,
                 exam_data['total_marks'])
            )
            result_id = cur.lastrowid
            
            # Insert user answers for each question
            for i, question in enumerate(exam_data['questions']):
                answer_data = user_answers.get(str(i))
                feedback = next((f for f in question_feedback if f['question_index'] == i), None)
                
                # Get the question_id from the database
                cur.execute(
                    """SELECT id FROM exam_questions 
                    WHERE exam_id = %s AND question_order = %s""",
                    (exam_data['id'], i+1)
                )
                question_row = cur.fetchone()
                
                if question_row:
                    question_id = question_row['id']
                    
                    # Insert answer record
                    cur.execute(
                        """INSERT INTO exam_user_answers 
                        (result_id, question_id, answer_text, selected_option, marks_awarded, feedback) 
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                        (result_id,
                         question_id,
                         answer_data if question['question_type'] == 'descriptive' else None,
                         int(answer_data) if question['question_type'] == 'mcq' and answer_data is not None else None,
                         feedback['marks'] if feedback else 0,
                         feedback['feedback'] if feedback else '')
                    )
            
            mysql.connection.commit()
            
            logger.debug(f"Saved exam results to database. Result ID: {result_id}")
        except Exception as e:
            mysql.connection.rollback()
            logger.error(f"Error saving exam results: {str(e)}")
            # Continue to return results even if saving fails
        
        # Prepare final results
        result = {
            "exam_data": exam_data,
            "user_answers": user_answers,
            "score": total_marks,
            "total_marks": exam_data['total_marks'],
            "question_feedback": question_feedback,
            "topic_analysis": analysis['topic_analysis'],
            "difficulty_analysis": analysis['difficulty_analysis'],
            "topic_chart": charts['topic_chart'],
            "difficulty_chart": charts['difficulty_chart'],
            "marks_chart": charts['marks_chart']
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in evaluate_exam: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/digital-exam/stats')
@login_required
def digital_exam_stats():
    """Get statistics for digital exams"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # Get basic stats
        cur.execute("""
            SELECT 
                COUNT(*) as exams_completed,
                AVG(score) as average_score,
                SUM(score) as total_marks_earned,
                SUM(total_marks) as total_possible_marks
            FROM exam_results 
            WHERE user_id = %s
        """, (user_id,))
        stats = cur.fetchone()
        
        # Get recent exams
        cur.execute("""
            SELECT er.id, ep.title, ep.topic, er.score, er.total_marks, 
                   er.completed_at, ep.difficulty_level
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            WHERE er.user_id = %s
            ORDER BY er.completed_at DESC
            LIMIT 3
        """, (user_id,))
        recent_exams = cur.fetchall()
        
        # Get topic performance
        cur.execute("""
            SELECT ep.topic, 
                   AVG(er.score) as avg_score,
                   COUNT(er.id) as exam_count
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            WHERE er.user_id = %s
            GROUP BY ep.topic
            ORDER BY avg_score DESC
            LIMIT 5
        """, (user_id,))
        topic_performance = cur.fetchall()
        
        # Calculate percentages
        if stats and stats['total_possible_marks'] and stats['total_possible_marks'] > 0:
            avg_percentage = (stats['average_score'] / stats['total_possible_marks']) * 100 * stats['exams_completed']
        else:
            avg_percentage = 0
        
        exam_stats = {
            'exams_completed': stats['exams_completed'] if stats else 0,
            'average_score': round(float(stats['average_score']), 1) if stats and stats['average_score'] else 0,
            'average_percentage': round(avg_percentage, 1) if stats else 0,
            'recent_exams': recent_exams,
            'topic_performance': topic_performance,
            'created_at': datetime.now()  # You might want to get user creation date from users table
        }
        
        return render_template('digital_exam_dashboard.html', exam_stats=exam_stats)
    except Exception as e:
        logger.error(f"Error getting digital exam stats: {str(e)}")
        flash('Error loading exam statistics', 'danger')
        return redirect(url_for('dashboard'))
    


# ================ DATA MODELS ================
# (Keep your existing models and add these new ones)

class ExamResult(BaseModel):
    id: int
    exam_id: int
    user_id: int
    score: float
    total_marks: int
    completed_at: datetime
    username: Optional[str] = None
    title: Optional[str] = None

class ExamAnswer(BaseModel):
    id: int
    result_id: int
    question_id: int
    answer_text: Optional[str] = None
    selected_option: Optional[int] = None
    marks_awarded: float
    feedback: Optional[str] = None
    question_text: str
    question_type: str
    max_marks: int
    options: Optional[List[str]] = None

# ================ HELPER FUNCTIONS ================
# (Keep your existing helper functions and add these new ones)

def get_user_exams(user_id):
    """Retrieve all exams created by a user"""
    cur = mysql.connection.cursor()
    try:
        cur.execute("""
            SELECT id, title, topic, difficulty_level, total_questions, 
                   mcq_count, descriptive_count, total_marks, created_at 
            FROM exam_papers 
            WHERE created_by = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        return cur.fetchall()
    except Exception as e:
        logger.error(f"Error getting user exams: {str(e)}")
        return []
    finally:
        cur.close()

def get_exam_by_id(exam_id):
    """Retrieve exam by ID with all questions and options"""
    cur = mysql.connection.cursor()
    try:
        # Get exam metadata
        cur.execute("SELECT * FROM exam_papers WHERE id = %s", (exam_id,))
        exam = cur.fetchone()
        
        if exam:
            # Get questions
            cur.execute("""
                SELECT * FROM exam_questions 
                WHERE exam_id = %s 
                ORDER BY question_order
            """, (exam_id,))
            questions = cur.fetchall()
            
            exam['questions'] = []
            for question in questions:
                # Get options for MCQ questions
                if question['question_type'] == 'mcq':
                    cur.execute("""
                        SELECT * FROM exam_question_options 
                        WHERE question_id = %s 
                        ORDER BY option_order
                    """, (question['id'],))
                    options = [opt['option_text'] for opt in cur.fetchall()]
                    question['options'] = options
                
                exam['questions'].append(question)
        
        return exam
    except Exception as e:
        logger.error(f"Error getting exam by ID: {str(e)}")
        return None
    finally:
        cur.close()

def generate_exam_pdf(exam_data):
    """Generate PDF from exam data"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=exam_data['title'], ln=1, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 6, txt=f"Topic: {exam_data['topic']}", ln=1, align='C')
    pdf.cell(200, 6, txt=f"Difficulty: {exam_data['difficulty_level'].capitalize()}", ln=1, align='C')
    pdf.cell(200, 6, txt=f"Total Questions: {exam_data['total_questions']} (MCQ: {exam_data['mcq_count']}, Descriptive: {exam_data['descriptive_count']})", ln=1, align='C')
    pdf.cell(200, 6, txt=f"Total Marks: {exam_data['total_marks']}", ln=1, align='C')
    pdf.ln(10)
    
    # Questions
    for i, question in enumerate(exam_data['questions'], 1):
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 5, txt=f"Q{i}. {question['question_text']}")
        
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, txt=f"Type: {question['question_type'].upper()} | Marks: {question['max_marks']} | Difficulty: {question['difficulty']}", ln=1)
        
        if question['question_type'] == 'mcq':
            for j, option in enumerate(question['options'], 1):
                prefix = "✓" if str(j-1) == question['correct_answer'] else "•"
                pdf.cell(5, 4, txt=prefix)
                pdf.multi_cell(0, 4, txt=f" {option}")
        else:
            pdf.multi_cell(0, 5, txt="[Descriptive Answer]")
            if question.get('correct_answer'):
                pdf.set_font("Arial", 'I', 9)
                pdf.multi_cell(0, 4, txt=f"Model Answer: {question['correct_answer']}")
                pdf.set_font("Arial", size=10)
        
        pdf.ln(5)
    
    return pdf

# ================ EXAM ROUTES ================

@app.route('/my-exams')
@login_required
def my_exams():
    """Display all exams created by the user"""
    try:
        user_id = session['user_id']
        exams = get_user_exams(user_id)
        return render_template('my_exams.html', exams=exams)
    except Exception as e:
        logger.error(f"Error in my_exams: {str(e)}")
        flash('Error loading exams', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/exam/<int:exam_id>')
@login_required
def view_exam(exam_id):
    """View a specific exam"""
    try:
        exam = get_exam_by_id(exam_id)
        if not exam:
            flash('Exam not found', 'danger')
            return redirect(url_for('my_exams'))
        
        if exam['created_by'] != session['user_id']:
            flash('You are not authorized to view this exam', 'danger')
            return redirect(url_for('my_exams'))
        
        return render_template('view_exam.html', exam=exam)
    except Exception as e:
        logger.error(f"Error in view_exam: {str(e)}")
        flash('Error loading exam', 'danger')
        return redirect(url_for('my_exams'))

@app.route('/load-exam')
@login_required
def load_exam():
    """Page to load previous exams"""
    try:
        user_id = session['user_id']
        exams = get_user_exams(user_id)
        return render_template('load_exam.html', exams=exams)
    except Exception as e:
        logger.error(f"Error in load_exam: {str(e)}")
        flash('Error loading exams', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/download-exam-pdf', methods=['POST'])
@login_required
def download_exam_pdf():
    """Endpoint to download exam as PDF"""
    try:
        exam_data = request.get_json()
        if not exam_data:
            return jsonify({'error': 'No exam data provided'}), 400

        pdf = generate_exam_pdf(exam_data)
        
        # Create a bytes buffer for the PDF
        pdf_bytes = io.BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)
        
        return send_file(
            pdf_bytes,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{exam_data['title'].replace(' ', '_')}.pdf"
        )
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete-exam/<int:exam_id>', methods=['DELETE'])
@login_required
def delete_exam(exam_id):
    """Delete an exam and all its questions"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # Verify the exam belongs to the user
        cur.execute("SELECT created_by FROM exam_papers WHERE id = %s", (exam_id,))
        exam = cur.fetchone()
        
        if not exam:
            return jsonify({'error': 'Exam not found'}), 404
            
        if exam['created_by'] != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Delete will cascade to questions and options
        cur.execute("DELETE FROM exam_papers WHERE id = %s", (exam_id,))
        
        mysql.connection.commit()
        return jsonify({'success': True})
    except Exception as e:
        mysql.connection.rollback()
        logger.error(f"Error deleting exam: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()

@app.route('/exam-results/<int:exam_id>')
@login_required
def exam_results(exam_id):
    """View results for a specific exam"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # Verify exam ownership
        cur.execute("SELECT created_by FROM exam_papers WHERE id = %s", (exam_id,))
        exam = cur.fetchone()
        
        if not exam:
            flash('Exam not found', 'danger')
            return redirect(url_for('my_exams'))
            
        if exam['created_by'] != user_id:
            flash('You are not authorized to view these results', 'danger')
            return redirect(url_for('my_exams'))
        
        # Get all results for this exam
        cur.execute("""
            SELECT er.*, u.username 
            FROM exam_results er
            JOIN users u ON er.user_id = u.id
            WHERE er.exam_id = %s
            ORDER BY er.completed_at DESC
        """, (exam_id,))
        results = cur.fetchall()
        
        return render_template('exam_results.html', exam_id=exam_id, results=results)
    except Exception as e:
        logger.error(f"Error in exam_results: {str(e)}")
        flash('Error loading exam results', 'danger')
        return redirect(url_for('my_exams'))
    finally:
        cur.close()

@app.route('/exam-result-detail/<int:result_id>')
@login_required
def exam_result_detail(result_id):
    """View detailed results for a specific exam attempt"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # Get result details
        cur.execute("""
            SELECT er.*, ep.title, ep.total_marks, u.username, ep.created_by
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            JOIN users u ON er.user_id = u.id
            WHERE er.id = %s
        """, (result_id,))
        result = cur.fetchone()
        
        if not result:
            flash('Result not found', 'danger')
            return redirect(url_for('dashboard'))
            
        # Verify authorization (either exam creator or the user who took the exam)
        if result['created_by'] != user_id and result['user_id'] != user_id:
            flash('You are not authorized to view this result', 'danger')
            return redirect(url_for('dashboard'))
        
        # Get all answers for this result
        cur.execute("""
            SELECT eua.*, eq.question_text, eq.question_type, eq.max_marks
            FROM exam_user_answers eua
            JOIN exam_questions eq ON eua.question_id = eq.id
            WHERE eua.result_id = %s
            ORDER BY eq.question_order
        """, (result_id,))
        answers = cur.fetchall()
        
        # For MCQ questions, get the options
        for answer in answers:
            if answer['question_type'] == 'mcq':
                cur.execute("""
                    SELECT option_text FROM exam_question_options
                    WHERE question_id = %s
                    ORDER BY option_order
                """, (answer['question_id'],))
                answer['options'] = [opt['option_text'] for opt in cur.fetchall()]
        
        # Calculate percentage score
        result['percentage'] = round((result['score'] / result['total_marks']) * 100, 1)
        
        return render_template('exam_result_detail.html', result=result, answers=answers)
    except Exception as e:
        logger.error(f"Error in exam_result_detail: {str(e)}")
        flash('Error loading exam result details', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        cur.close()






@app.route('/digital-exam/analytics')
@login_required
def digital_exam_analytics():
    """Digital exam analytics dashboard"""
    try:
        user_id = session['user_id']
        cur = mysql.connection.cursor()
        
        # 1. Basic stats
        cur.execute("""
            SELECT 
                COUNT(*) AS exams_completed,
                AVG(score) AS average_score,
                SUM(score) AS total_marks_earned,
                SUM(total_marks) AS total_possible_marks
            FROM exam_results 
            WHERE user_id = %s
        """, (user_id,))
        stats = cur.fetchone()
        
        # 2. Score progress over time
        cur.execute("""
            SELECT DATE(completed_at) AS date, AVG(score) AS avg_score
            FROM exam_results 
            WHERE user_id = %s
            GROUP BY DATE(completed_at)
            ORDER BY DATE(completed_at) ASC
        """, (user_id,))
        score_progress = cur.fetchall()
        
        # 3. Topic mastery
        cur.execute("""
            SELECT ep.topic, 
                   AVG(er.score / er.total_marks * 100) AS avg_score_percentage,
                   COUNT(er.id) AS exam_count
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            WHERE er.user_id = %s
            GROUP BY ep.topic
            ORDER BY avg_score_percentage DESC
            LIMIT 6
        """, (user_id,))
        topic_mastery = cur.fetchall()
        
        # 4. Difficulty distribution
        cur.execute("""
            SELECT ep.difficulty_level, COUNT(*) AS count
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            WHERE er.user_id = %s
            GROUP BY ep.difficulty_level
        """, (user_id,))
        difficulty_dist = cur.fetchall()
        
        # 5. Time of day performance
        cur.execute("""
            SELECT 
                CASE 
                    WHEN HOUR(completed_at) BETWEEN 6 AND 11 THEN 'Morning'
                    WHEN HOUR(completed_at) BETWEEN 12 AND 17 THEN 'Afternoon'
                    WHEN HOUR(completed_at) BETWEEN 18 AND 22 THEN 'Evening'
                    ELSE 'Night'
                END AS time_of_day,
                AVG(score / total_marks * 100) AS avg_score_percentage
            FROM exam_results
            WHERE user_id = %s
            GROUP BY time_of_day
            ORDER BY FIELD(time_of_day, 'Morning', 'Afternoon', 'Evening', 'Night')
        """, (user_id,))
        time_performance = cur.fetchall()
        
        # 6. Personalized recommendations
        cur.execute("""
            SELECT 
                ep.topic, 
                COUNT(*) AS total_questions,
                SUM(CASE WHEN eua.marks_awarded >= eq.max_marks * 0.8 THEN 1 ELSE 0 END) AS correct_answers
            FROM exam_results er
            JOIN exam_papers ep ON er.exam_id = ep.id
            JOIN exam_user_answers eua ON er.id = eua.result_id
            JOIN exam_questions eq ON eua.question_id = eq.id
            WHERE er.user_id = %s
            GROUP BY ep.topic
            HAVING (SUM(CASE WHEN eua.marks_awarded >= eq.max_marks * 0.8 THEN 1 ELSE 0 END) / COUNT(*)) < 0.7
            ORDER BY (SUM(CASE WHEN eua.marks_awarded >= eq.max_marks * 0.8 THEN 1 ELSE 0 END) / COUNT(*)) ASC
            LIMIT 3
        """, (user_id,))
        recommendations = cur.fetchall()
        
        # Prepare data for charts
        score_data = {
            'dates': [str(row['date']) for row in score_progress],
            'scores': [float(row['avg_score']) for row in score_progress]
        }
        
        topic_data = {
            'topics': [row['topic'] for row in topic_mastery],
            'scores': [float(row['avg_score_percentage']) for row in topic_mastery]
        }
        
        difficulty_data = {
            'difficulties': [row['difficulty_level'] for row in difficulty_dist],
            'counts': [row['count'] for row in difficulty_dist]
        }
        
        time_data = {
            'times': [row['time_of_day'] for row in time_performance],
            'scores': [float(row['avg_score_percentage']) for row in time_performance]
        }
        
        # Format recommendations
        formatted_recs = []
        for rec in recommendations:
            accuracy = (rec['correct_answers'] / rec['total_questions']) * 100 if rec['total_questions'] > 0 else 0
            formatted_recs.append({
                'topic': rec['topic'],
                'accuracy': round(accuracy, 1),
                'question_count': rec['total_questions']
            })
        
        # Calculate overall stats
        exams_completed = stats['exams_completed'] or 0
        average_score = float(stats['average_score'] or 0)
        total_marks_earned = float(stats['total_marks_earned'] or 0)
        total_possible_marks = float(stats['total_possible_marks'] or 1)  # Avoid division by zero
        
        analytics_data = {
            'exams_completed': exams_completed,
            'average_score': round(average_score, 1),
            'average_percentage': round((total_marks_earned / total_possible_marks) * 100, 1),
            'score_progress': score_data,
            'topic_mastery': topic_data,
            'difficulty_dist': difficulty_data,
            'time_performance': time_data,
            'recommendations': formatted_recs
        }
        
        return render_template('digital_exam_anlytics.html', analytics_data=analytics_data)
        
    except Exception as e:
        logger.error(f"Error in digital_exam_analytics: {str(e)}")
        flash('Error loading digital exam analytics data', 'danger')
        return redirect(url_for('dashboard'))



if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable not set")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)



