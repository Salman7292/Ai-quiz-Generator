# AI-Powered Quiz & Exam Generator 🚀

[![Demo Video](https://img.youtube.com/vi/6gH_RAfH3Ik/maxresdefault.jpg)](https://www.youtube.com/watch?v=6gH_RAfH3Ik)

*A full-stack web application that generates, administers, and evaluates quizzes/exams using AI*

## ✨ Features

| Feature Category | Highlights |
|-----------------|------------|
| **AI Generation** | Create quizzes/exams with Google Gemini API |
| **Smart Evaluation** | Auto-grade descriptive answers with AI feedback |
| **Comprehensive Analytics** | Performance breakdowns by topic/difficulty |
| **PDF Export** | Download quizzes/exams as printable PDFs |
| **User Management** | Secure authentication & progress tracking |

## 🎥 Video Demo

[![Watch the full demo](https://img.shields.io/badge/▶_Watch_Full_Demo-FF0000?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=6gH_RAfH3Ik)

**Timestamps:**
- 00:00 - Introduction & Overview
- 01:15 - Quiz Generation Demo
- 03:40 - Exam Evaluation Process
- 05:25 - Analytics Dashboard
- 07:10 - PDF Export Feature

## 🛠️ Tech Stack

**Backend:**
- Python Flask
- Google Gemini API
- MySQL Database
- FPDF (PDF generation)

**Frontend:**
- HTML5, CSS3, JavaScript
- Chart.js (Visualizations)
- Responsive Design

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MySQL Server
- Google API Key

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/quiz-exam-generator.git
cd quiz-exam-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
