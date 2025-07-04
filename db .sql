-- Table to store generated exam papers
CREATE TABLE `exam_papers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) NOT NULL,
  `topic` varchar(100) NOT NULL,
  `difficulty_level` enum('easy','medium','hard') NOT NULL,
  `total_questions` int(11) NOT NULL,
  `mcq_count` int(11) NOT NULL,
  `descriptive_count` int(11) NOT NULL,
  `total_marks` int(11) NOT NULL,
  `created_by` int(11) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `created_by` (`created_by`),
  CONSTRAINT `exam_papers_ibfk_1` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Table to store exam questions
CREATE TABLE `exam_questions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exam_id` int(11) NOT NULL,
  `question_text` text NOT NULL,
  `question_type` enum('mcq','descriptive') NOT NULL,
  `difficulty` enum('easy','medium','hard') NOT NULL,
  `topic` varchar(100) NOT NULL,
  `max_marks` int(11) NOT NULL DEFAULT 5,
  `correct_answer` text DEFAULT NULL, -- For MCQs, stores the correct option index
  `question_order` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `exam_id` (`exam_id`),
  CONSTRAINT `exam_questions_ibfk_1` FOREIGN KEY (`exam_id`) REFERENCES `exam_papers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Table to store options for MCQ questions
CREATE TABLE `exam_question_options` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `question_id` int(11) NOT NULL,
  `option_text` text NOT NULL,
  `option_order` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `exam_question_options_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `exam_questions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Table to store exam results
CREATE TABLE `exam_results` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exam_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `score` int(11) NOT NULL,
  `total_marks` int(11) NOT NULL,
  `completed_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `exam_id` (`exam_id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `exam_results_ibfk_1` FOREIGN KEY (`exam_id`) REFERENCES `exam_papers` (`id`),
  CONSTRAINT `exam_results_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Table to store user answers
CREATE TABLE `exam_user_answers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `result_id` int(11) NOT NULL,
  `question_id` int(11) NOT NULL,
  `answer_text` text DEFAULT NULL, -- For descriptive answers
  `selected_option` int(11) DEFAULT NULL, -- For MCQs, stores the selected option index
  `marks_awarded` int(11) DEFAULT NULL,
  `feedback` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `result_id` (`result_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `exam_user_answers_ibfk_1` FOREIGN KEY (`result_id`) REFERENCES `exam_results` (`id`) ON DELETE CASCADE,
  CONSTRAINT `exam_user_answers_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `exam_questions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;