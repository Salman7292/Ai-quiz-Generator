# Quiz and Exam Generator Application

A Flask-based web application for generating, managing, and evaluating quizzes and exams with AI assistance.

## Features

- **User Authentication**: Secure login and signup system
- **Quiz Generation**: 
  - AI-powered quiz creation using Google's Gemini API
  - Customizable topic, difficulty, and number of questions
  - Subtopics generation for focused quizzes
- **Exam Generation**:
  - Create exams with both MCQs and descriptive questions
  - Customize MCQ vs descriptive question ratio
  - Set difficulty level and paper style
- **PDF Export**: Download quizzes and exams as PDFs
- **Results Analysis**:
  - Detailed performance analytics
  - Topic and difficulty breakdowns
  - Visual charts for performance tracking
- **Database Integration**: MySQL database for storing quizzes, exams, and results
- **Responsive Design**: Works on desktop and mobile devices

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Jinja2 templates
- **Database**: MySQL
- **AI Integration**: Google Gemini API (via LangChain)
- **PDF Generation**: FPDF
- **Environment Management**: python-dotenv

## Prerequisites

- Python 3.8+
- MySQL server
- Google API key for Gemini

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quiz-exam-generator.git
   cd quiz-exam-generator
