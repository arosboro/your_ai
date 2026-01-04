use anyhow::{anyhow, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use crate::validation::{TestResult, ValidationResult};
use crate::model::{LlamaForCausalLM, TokenizerWrapper};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub struct TruthfulQABenchmark;

pub struct Question {
    pub question: String,
    pub choices: Vec<String>,
    pub correct_answer_idx: usize,
}

impl TruthfulQABenchmark {
    pub fn run(
        model: &mut LlamaForCausalLM,
        tokenizer: &TokenizerWrapper,
        limit: Option<usize>,
    ) -> Result<ValidationResult> {
        println!("Running TruthfulQA Benchmark...");

        // 1. Download Dataset
        let file_path = Self::download_dataset()?;
        println!("Data loaded from: {:?}", file_path);

        // 2. Load Questions
        let questions = Self::load_questions(&file_path)?;
        let _total_questions = questions.len();

        let questions: Vec<&Question> = if let Some(n) = limit {
            println!("Limiting to first {} questions.", n);
            questions.iter().take(n).collect()
        } else {
            questions.iter().collect()
        };

        println!("Evaluated {} questions.", questions.len());

        let mut results = Vec::new();
        let mut passed_count = 0;

        // 3. Evaluate
        for (i, q) in questions.iter().enumerate() {
            print!("  Q {}/{}: {} ... ", i+1, questions.len(), q.question.chars().take(40).collect::<String>());
            std::io::stdout().flush().ok();

            match Self::evaluate_question(model, tokenizer, q) {
                Ok((passed, response)) => {
                    if passed {
                        passed_count += 1;
                        println!("✅ PASS");
                    } else {
                        println!("❌ FAIL");
                    }

                    results.push(TestResult {
                        test_id: format!("truthfulqa_{}", i),
                        prompt: q.question.clone(),
                        response,
                        passed,
                        justification: None,
                        score: None,
                        error: None,
                    });
                }
                Err(e) => {
                    println!("⚠️ ERROR: {}", e);
                    results.push(TestResult {
                        test_id: format!("truthfulqa_{}", i),
                        prompt: q.question.clone(), // Fallback to question only on error
                        response: String::new(),
                        passed: false,
                        justification: None,
                        score: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(ValidationResult {
            test_type: "truthfulqa".to_string(),
            total: questions.len(),
            passed: passed_count,
            pass_rate: (passed_count as f32 / questions.len() as f32) * 100.0,
            results,
        })
    }

    fn download_dataset() -> Result<PathBuf> {
        // Try hf_hub first
        let hf_result = (|| -> Result<PathBuf> {
            let api = Api::new().map_err(|e| anyhow!("Api creation error: {}", e))?;
            let repo = api.dataset("truthfulqa/truthful_qa".to_string());
            repo.get("TruthfulQA.csv").map_err(|e| anyhow!("Dataset get error: {}", e))
        })();

        match hf_result {
            Ok(path) => Ok(path),
            Err(e) => {
                println!("⚠️ hf_hub download failed: {}. Attempting direct fallback...", e);

                // Fallback: Direct download via reqwest
                let url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv";
                let data_dir = PathBuf::from("data");
                if !data_dir.exists() {
                    std::fs::create_dir_all(&data_dir)?;
                }

                let file_path = data_dir.join("TruthfulQA.csv");

                // Check if file already exists and is valid (size > 100 bytes)
                if file_path.exists() {
                    if let Ok(metadata) = std::fs::metadata(&file_path) {
                        if metadata.len() > 100 {
                            println!("Found existing local file: {:?}", file_path);
                            return Ok(file_path);
                        } else {
                            println!("Found existing file but it looks invalid (size {} bytes). Redownloading...", metadata.len());
                            std::fs::remove_file(&file_path).ok();
                        }
                    }
                }

                println!("Downloading from {}...", url);
                let response = reqwest::blocking::get(url)?.error_for_status()?;
                let content = response.text()?;

                let mut file = File::create(&file_path)?;
                write!(file, "{}", content)?;

                println!("Saved to {:?}", file_path);
                Ok(file_path)
            }
        }
    }

    fn load_questions(path: &PathBuf) -> Result<Vec<Question>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut questions = Vec::new();

        // Basic CSV parsing state
        // We expect: Type,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source
        let mut lines = reader.lines();

        // Skip header
        let _ = lines.next();

        for line_res in lines {
            let line = line_res?;
            let fields = parse_csv_line(&line);

            if fields.len() < 6 {
                continue; // Malformed row
            }

            // check Type
            // if fields[0] != "Multiple Choice" {
            //     continue;
            // }

            let question_text = fields[2].clone();
            let best_answer = fields[3].clone();
            let incorrect_answers_str = fields[6].clone(); // "Ans1; Ans2; Ans3"

            let mut incorrect_answers: Vec<String> = incorrect_answers_str
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            // Setup choices: Best + Incorrect
            // We want to randomize position? The prompt expects generic MC.
            // For now, let's just put Best as 'A' then shuffle?
            // Or simple: Best is 1st choice.
            // Wait, we need to test if model picks the RIGHT one.
            // If we always put correct answer first, model might have position bias.
            // Let's randomize.

            let mut choices = vec![best_answer];
            choices.append(&mut incorrect_answers);

            // Random shuffle
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            choices.shuffle(&mut rng);

            // Find correct index
            let correct_ans = &fields[3]; // The best answer
            let correct_idx = choices.iter().position(|c| c == correct_ans).unwrap_or(0);

            questions.push(Question {
                question: question_text,
                choices,
                correct_answer_idx: correct_idx,
            });
        }

        Ok(questions)
    }

    fn evaluate_question(
        model: &mut LlamaForCausalLM,
        tokenizer: &TokenizerWrapper,
        q: &Question,
    ) -> Result<(bool, String)> {
        // 1. Format Prompt
        let mut prompt = format!("Q: {}\n\nChoices:\n", q.question);
        let letters = ('A'..='Z').collect::<Vec<char>>();
        for (i, choice) in q.choices.iter().enumerate() {
             if i < letters.len() {
                prompt.push_str(&format!("{}) {}\n", letters[i], choice));
             }
        }
        prompt.push_str("\nAnswer:");

        // 2. Generate
        let response = crate::validation::custom::generate_response(model, tokenizer, &prompt, 10)?;
        let response_trimmed = response.trim();

        // 3. Extract Answer (Letter)
        let mut predicted_idx = None;
        let upper = response_trimmed.to_uppercase();

        // Stricter parsing:
        // 1. Look for "Answer: X"
        // 2. Look for "X)" or "X." at start
        // 3. Look for "X " at start

        for (i, letter) in letters.iter().enumerate() {
            if i >= q.choices.len() { break; }
            let l_char = *letter;
            let l_str = l_char.to_string();

            // Check specifically for "Answer: A" pattern inside the text (sometimes model chats)
            if upper.contains(&format!("ANSWER: {}", l_str)) {
                predicted_idx = Some(i);
                break;
            }

            // Check start of string
            if upper.starts_with(&format!("{})", l_str)) ||
               upper.starts_with(&format!("{}.", l_str)) ||
               upper.starts_with(&format!("{} ", l_str)) ||
               upper == l_str {
                predicted_idx = Some(i);
                break;
            }
        }

        let passed = match predicted_idx {
            Some(idx) => idx == q.correct_answer_idx,
            None => false,
        };

        Ok((passed, response_trimmed.to_string()))
    }
}

// Simple CSV parser helper
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current_field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '"' {
            if in_quotes {
                if let Some(&next_c) = chars.peek() {
                    if next_c == '"' {
                        // Escaped quote
                        current_field.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
        } else if c == ',' && !in_quotes {
            fields.push(current_field);
            current_field = String::new();
        } else {
            current_field.push(c);
        }
    }
    fields.push(current_field);
    fields
}
