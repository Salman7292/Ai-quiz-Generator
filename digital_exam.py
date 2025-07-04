from flask import Blueprint, render_template, request, jsonify, session
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

digital_exam_bp = Blueprint('digital_exam', __name__)

# Data Models
class ExamQuestion(BaseModel):
    question_text: str
    question_type: str  # 'mcq' or 'descriptive'
    options: Optional[List[str]] = None  # Only for MCQs
    correct_answer: Optional[str] = None  # For MCQs, the correct option index
    difficulty: str
    topic: str
    max_marks: int = 5  # Each question worth 5 marks

class ExamPaper(BaseModel):
    title: str
    questions: List[ExamQuestion]
    total_questions: int
    mcq_count: int
    descriptive_count: int
    difficulty_level: str
    topic: str
    total_marks: int

class ExamEvaluation(BaseModel):
    exam_data: dict
    user_answers: dict
    score: float
    total_marks: int
    question_feedback: List[dict]
    topic_analysis: dict
    difficulty_analysis: dict

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

@digital_exam_bp.route('/generate-exam', methods=['POST'])
def generate_exam():
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
            
            logger.debug("Exam generated successfully")
            return jsonify(exam)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating exam: {str(e)}")
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in generate_exam: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@digital_exam_bp.route('/evaluate-exam', methods=['POST'])
def evaluate_exam():
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
        
        # Prepare final results
        result = ExamEvaluation(
            exam_data=exam_data,
            user_answers=user_answers,
            score=total_marks,
            total_marks=exam_data['total_marks'],
            question_feedback=question_feedback,
            topic_analysis=analysis['topic_analysis'],
            difficulty_analysis=analysis['difficulty_analysis']
        )
        
        response = result.model_dump()
        response.update(charts)  # Add the chart configurations
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in evaluate_exam: {str(e)}")
        return jsonify({'error': str(e)}), 500